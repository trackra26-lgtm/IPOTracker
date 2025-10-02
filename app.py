# app.py
import os, re, json, smtplib, time
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator
from alpaca_trade_api.rest import REST, APIError

load_dotenv()  # load .env if present (for local/dev)

# ---------------------- Config ----------------------
NASDAQ_IPO_URL = os.environ.get("NASDAQ_IPO_URL", "https://www.nasdaq.com/market-activity/ipos")
CACHE_PATH = Path(os.environ.get("CACHE_PATH", "seen_ipos.json"))

SECRET_TOKEN = os.environ.get("SECRET_TOKEN")  # optional, but recommended

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER or "notifier@example.com")
TO_EMAILS = [e.strip() for e in os.environ.get("TO_EMAILS", "").split(",") if e.strip()]

CAPITAL_IQ_PATH = Path(os.environ.get("CAPITAL_IQ_PATH", "capital_iq_healthcare.xlsx"))
COMPANY_COLUMN = os.environ.get("COMPANY_COLUMN", "Company")
TICKER_COLUMN = os.environ.get("TICKER_COLUMN", "Ticker")

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DEFAULT_MAX_TRADES_PER_MINUTE = max(1, int(os.environ.get("MAX_TRADES_PER_MINUTE", "4")))

app = FastAPI(title="IPO Tracker (Nasdaq)")

# ---------------------- Helpers ----------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:160]

def build_id(company: str, ticker: str, expected_date: str) -> str:
    basis = f"{ticker or company}|{expected_date}"
    return slugify(basis)

def load_cache() -> set:
    if CACHE_PATH.exists():
        try:
            return set(json.loads(CACHE_PATH.read_text()))
        except Exception:
            return set()
    return set()

def save_cache(ids: set):
    CACHE_PATH.write_text(json.dumps(sorted(list(ids)), indent=2))

# ---------------------- Capital IQ data helpers ----------------------
_company_cache: Dict[str, Any] = {"mtime": None, "df": None}
LOWER_IS_BETTER_HINTS = {"debt", "liability", "expense", "ratio", "turnover", "days", "burn"}


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def _resolve_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    target = _normalize_column_name(requested)
    for col in df.columns:
        if _normalize_column_name(str(col)) == target:
            return col
    return None


def load_company_dataframe(force: bool = False) -> pd.DataFrame:
    path = CAPITAL_IQ_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Capital IQ export not found at {path.resolve()}"
        )

    mtime = path.stat().st_mtime
    cache = _company_cache
    if not force and cache.get("df") is not None and cache.get("mtime") == mtime:
        return cache["df"].copy()

    df = pd.read_excel(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Unable to parse Excel file into a dataframe")

    df.columns = [str(col).strip() for col in df.columns]
    df = df.dropna(how="all")

    cache["df"] = df
    cache["mtime"] = mtime
    return df.copy()


def _extract_text(row: Dict[str, Any], column: Optional[str], fallback: str = "") -> str:
    if not column:
        return fallback
    value = row.get(column, fallback)
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return fallback
    return str(value)


class MetricConfig(BaseModel):
    column: str = Field(..., description="Column name in the dataset to use for scoring")
    weight: float = Field(1.0, gt=0, description="Relative weight for this metric")
    higher_is_better: bool = Field(
        True, description="If false, lower values contribute a higher score"
    )

    @validator("column")
    def column_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("column must be provided")
        return value

    @validator("weight")
    def weight_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("weight must be greater than zero")
        return value


class RankingRequest(BaseModel):
    metrics: Optional[List[MetricConfig]] = Field(
        None, description="Custom metric configuration. If omitted the API will choose defaults"
    )
    top_n: int = Field(10, gt=0, le=200)
    sector: Optional[str] = Field(
        None, description="Optional sector filter. Matches the first column resembling 'Sector'"
    )
    industry: Optional[str] = Field(
        None, description="Optional industry filter. Matches the first column resembling 'Industry'"
    )

    class Config:
        schema_extra = {
            "example": {
                "metrics": [
                    {"column": "Revenue Growth %", "weight": 0.5, "higher_is_better": True},
                    {"column": "EBITDA Margin %", "weight": 0.3, "higher_is_better": True},
                    {"column": "Debt to Equity", "weight": 0.2, "higher_is_better": False},
                ],
                "top_n": 10,
                "sector": "Healthcare",
                "industry": "Pharmaceuticals",
            }
        }


class TradeRequest(BaseModel):
    metrics: Optional[List[MetricConfig]] = Field(
        None, description="Metrics to derive ranking if symbols not provided"
    )
    top_n: int = Field(3, gt=0, le=50, description="How many ranked symbols to trade when symbols not provided")
    symbols: Optional[List[str]] = Field(
        None,
        description="Explicit list of ticker symbols to trade. When omitted the ranking output is used",
    )
    side: Literal["buy", "sell"] = Field("buy")
    notional: Optional[float] = Field(
        None, gt=0, description="Dollar amount to trade per order (mutually exclusive with quantity)"
    )
    quantity: Optional[int] = Field(
        None, gt=0, description="Share quantity per order (mutually exclusive with notional)"
    )
    time_in_force: str = Field("day", description="Alpaca time-in-force value, e.g. day or gtc")
    max_trades_per_minute: int = Field(
        DEFAULT_MAX_TRADES_PER_MINUTE, gt=0, le=60, description="Throttle rate for order submission"
    )
    sector: Optional[str] = None
    industry: Optional[str] = None

    @validator("symbols", each_item=True)
    def sanitize_symbols(cls, value: str) -> str:
        symbol = (value or "").strip().upper()
        if not symbol:
            raise ValueError("symbols must be non-empty strings")
        return symbol

    @root_validator
    def validate_trade_amount(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        notional = values.get("notional")
        quantity = values.get("quantity")
        if notional is None and quantity is None:
            raise ValueError("either notional or quantity must be supplied")
        if notional is not None and quantity is not None:
            raise ValueError("notional and quantity are mutually exclusive")
        return values


def _default_metrics(df: pd.DataFrame) -> List[MetricConfig]:
    numeric_cols = [
        col
        for col in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[col])
            and _normalize_column_name(col)
            not in {
                "",
                _normalize_column_name(COMPANY_COLUMN),
                _normalize_column_name(TICKER_COLUMN),
            }
        )
    ]
    selected = numeric_cols[: min(5, len(numeric_cols))]
    if not selected:
        raise ValueError("No numeric columns available to build a default ranking")

    weight = 1.0 / len(selected)
    metrics: List[MetricConfig] = []
    for col in selected:
        normalized = _normalize_column_name(col)
        higher_is_better = not any(keyword in normalized for keyword in LOWER_IS_BETTER_HINTS)
        metrics.append(
            MetricConfig(column=col, weight=weight, higher_is_better=higher_is_better)
        )
    return metrics


def _apply_optional_filter(df: pd.DataFrame, label: Optional[str], candidates: List[str]) -> pd.DataFrame:
    if not label:
        return df
    lowered = label.strip().lower()
    for candidate in candidates:
        resolved = _resolve_column(df, candidate)
        if resolved:
            series = df[resolved].astype(str).str.strip().str.lower()
            return df[series == lowered]
    return df


def _prepare_metric_series(df: pd.DataFrame, metrics: List[MetricConfig]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    total_weight = 0.0
    for metric in metrics:
        resolved = _resolve_column(df, metric.column)
        if not resolved:
            continue

        values = pd.to_numeric(df[resolved], errors="coerce")
        if values.notna().sum() == 0:
            continue

        total_weight += metric.weight
        prepared.append(
            {
                "requested": metric.column,
                "column": resolved,
                "weight": metric.weight,
                "higher_is_better": metric.higher_is_better,
                "values": values,
            }
        )

    if not prepared:
        raise ValueError("None of the requested metrics are present in the dataset")

    for item in prepared:
        item["normalized_weight"] = item["weight"] / total_weight if total_weight else 0
    return prepared


def _calculate_scores(df: pd.DataFrame, prepared_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    score = pd.Series(0.0, index=df.index)
    contributions: Dict[str, pd.Series] = {}

    for metric in prepared_metrics:
        values = metric["values"].astype(float)
        min_value = values.min()
        max_value = values.max()
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            normalized = pd.Series(0.0, index=values.index)
        else:
            normalized = (values - min_value) / (max_value - min_value)
        if not metric["higher_is_better"]:
            normalized = 1 - normalized
        normalized = normalized.fillna(0.0)
        contribution = normalized * metric["normalized_weight"]
        contributions[metric["column"]] = contribution
        score = score + contribution

    ranked = df.copy()
    ranked["score"] = score
    for col, series in contributions.items():
        ranked[f"score__{col}"] = series
    ranked = ranked.sort_values("score", ascending=False)
    return ranked


def compute_ranking(
    metrics: Optional[List[MetricConfig]],
    top_n: int,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
) -> Dict[str, Any]:
    df = load_company_dataframe()
    if df.empty:
        raise ValueError("Capital IQ dataset is empty")

    filtered = _apply_optional_filter(df, sector, ["sector", "gics sector"])
    filtered = _apply_optional_filter(filtered, industry, ["industry", "gics industry", "industry group"])

    if filtered.empty:
        raise ValueError("No companies match the requested filters")

    metric_config = metrics or _default_metrics(filtered)
    prepared = _prepare_metric_series(filtered, metric_config)
    ranked = _calculate_scores(filtered, prepared)
    top_df = ranked.head(top_n)

    company_col = _resolve_column(filtered, COMPANY_COLUMN) or COMPANY_COLUMN
    ticker_col = _resolve_column(filtered, TICKER_COLUMN) or TICKER_COLUMN

    records: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(top_df.iterrows(), start=1):
        record = {
            "rank": idx,
            "company": _extract_text(row, company_col, fallback=""),
            "ticker": _extract_text(row, ticker_col, fallback=""),
            "score": round(float(row.get("score", 0.0)), 6),
        }
        for metric in prepared:
            column_key = metric["column"]
            contribution_key = f"score__{column_key}"
            if contribution_key in row:
                raw_value = row.get(column_key)
                if pd.isna(raw_value):
                    value = None
                elif hasattr(raw_value, "item"):
                    try:
                        value = raw_value.item()
                    except Exception:
                        value = raw_value
                else:
                    value = raw_value
                record[column_key] = value
                record[f"normalized_{column_key}"] = round(
                    float(row.get(contribution_key, 0.0)), 6
                )
        records.append(record)

    metrics_used = [
        {
            "requested": metric["requested"],
            "column": metric["column"],
            "weight": round(float(metric["normalized_weight"]), 6),
            "higher_is_better": metric["higher_is_better"],
        }
        for metric in prepared
    ]

    return {
        "results": records,
        "metrics_used": metrics_used,
        "filters": {"sector": sector, "industry": industry},
        "total_companies": int(len(filtered)),
    }


_alpaca_client: Optional[REST] = None


def get_alpaca_client() -> REST:
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        raise RuntimeError("Alpaca credentials are not configured")
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
            api_version="v2",
        )
    return _alpaca_client


def _unique_symbols(symbols: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for symbol in symbols:
        sym = symbol.upper()
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return ordered


def execute_trade_plan(req: TradeRequest) -> Dict[str, Any]:
    ranking_context: Optional[Dict[str, Any]] = None
    symbols: List[str] = []

    if req.symbols:
        symbols = req.symbols
    else:
        ranking_context = compute_ranking(
            metrics=req.metrics,
            top_n=req.top_n,
            sector=req.sector,
            industry=req.industry,
        )
        symbols = [
            record.get("ticker")
            for record in ranking_context.get("results", [])
            if record.get("ticker")
        ]
        if not symbols:
            raise ValueError("Ranking did not yield any tickers to trade")

    ordered_symbols = _unique_symbols(symbols)
    client = get_alpaca_client()
    interval = 60.0 / max(1, req.max_trades_per_minute)
    order_results: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(ordered_symbols):
        payload = {
            "symbol": symbol,
            "side": req.side,
            "type": "market",
            "time_in_force": req.time_in_force,
        }
        if req.notional is not None:
            payload["notional"] = req.notional
        else:
            payload["qty"] = req.quantity

        try:
            order = client.submit_order(**payload)
            order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
            order_results.append(
                {
                    "symbol": symbol,
                    "status": "submitted",
                    "order_id": order_id,
                    "submitted_payload": payload,
                }
            )
        except APIError as exc:
            order_results.append(
                {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                }
            )
        except Exception as exc:
            order_results.append(
                {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                }
            )

        if idx < len(ordered_symbols) - 1 and interval > 0:
            time.sleep(interval)

    return {
        "orders": order_results,
        "symbols": ordered_symbols,
        "ranking_context": ranking_context,
    }

def send_email(new_ipos: List[Dict[str, Any]]):
    if not (SMTP_USER and SMTP_PASS and TO_EMAILS):
        return {"sent": False, "reason": "SMTP env vars missing"}

    subject = f"[IPO Tracker] {len(new_ipos)} new IPO(s)"
    lines = [f"Detected at: {datetime.utcnow().isoformat()}Z", f"Source: {NASDAQ_IPO_URL}", ""]
    for it in new_ipos:
        title = it.get("company", "")
        if it.get("ticker"):
            title += f" ({it['ticker']})"
        lines += [
            f"- {title}",
            f"  Expected Date: {it.get('expected_date','N/A')}",
            f"  Price Range: {it.get('price_range','N/A')} | Shares: {it.get('shares','N/A')} | Market: {it.get('market','N/A')}",
        ]
        if it.get("underwriters"):
            lines.append(f"  Lead Underwriters: {it['underwriters']}")
        if it.get("href"):
            lines.append(f"  Link: {it['href']}")
        lines.append("")
    body = "\n".join(lines)

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = ", ".join(TO_EMAILS)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(FROM_EMAIL, TO_EMAILS, msg.as_string())

    return {"sent": True, "to": TO_EMAILS}

# ---------------------- Nasdaq fetch ----------------------
def _nasdaq_headers():
    # Pretend to be a real browser; Nasdaq API/CDN likes these
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": NASDAQ_IPO_URL,
        "Origin": "https://www.nasdaq.com",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

def _make_item(row: Dict[str, Any]) -> Dict[str, Any]:
    company = norm(row.get("company", "") or row.get("companyName", "") or row.get("name", ""))
    ticker  = norm(row.get("symbol", "") or row.get("ticker", ""))
    price   = norm(row.get("priceRange", "") or row.get("price_range", ""))
    shares  = norm(row.get("shares", "") or row.get("sharesOffered", "") or row.get("shares_offered", ""))
    date    = norm(row.get("expectedDate", "") or row.get("expected_date", "") or row.get("date", ""))
    market  = norm(row.get("market", "") or row.get("exchange", ""))
    href    = row.get("href", "") or row.get("link", "") or ""

    uw = row.get("leadUnderwriters") or row.get("underwriters") or row.get("lead_underwriters") or ""
    if isinstance(uw, list):
        uw = ", ".join([norm(x) for x in uw])
    banks = norm(uw)

    return {
        "company": company,
        "ticker": ticker,
        "price_range": price,
        "shares": shares,
        "expected_date": date,
        "market": market,
        "href": href,
        "underwriters": banks,
    }

def scrape_nasdaq_ipos() -> List[Dict[str, Any]]:
    """
    Try Nasdaq JSON API shapes first; fall back to server-rendered HTML table.
    """
    items: List[Dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(_nasdaq_headers())

    # --- API attempts (several variants seen historically) ---
    api_candidates = [
        "https://api.nasdaq.com/api/ipo/calendar",
        f"https://api.nasdaq.com/api/ipo/calendar?date={datetime.utcnow():%Y-%m}",
        "https://api.nasdaq.com/api/ipo/calendar?date=all",
    ]
    for url in api_candidates:
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            d = (data or {}).get("data", {})
            blocks = []
            # common blocks
            for key in ("upcoming", "upcomingTable", "priced", "pricedTable"):
                table = d.get(key) or {}
                rows = table.get("rows")
                if isinstance(rows, list) and rows:
                    blocks.extend(rows)
            # single-table fallback
            if not blocks and isinstance(d.get("rows"), list):
                blocks = d["rows"]

            for row in blocks:
                payload = row.get("values") if isinstance(row, dict) and isinstance(row.get("values"), dict) else row
                item = _make_item(payload if isinstance(payload, dict) else {})
                if item["company"] or item["ticker"]:
                    items.append(item)
            if items:
                break
        except Exception:
            continue

    # --- HTML fallback (SSR table) ---
    if not items:
        try:
            html = session.get(NASDAQ_IPO_URL, timeout=25).text
            soup = BeautifulSoup(html, "html.parser")
            tables = soup.select("table")
            target = None
            for tbl in tables:
                headers = [norm(th.get_text()).lower() for th in tbl.select("thead th, tr th")]
                if any("company" in h for h in headers) and (any("expected" in h for h in headers) or any("date" in h for h in headers)):
                    target = tbl
                    break
            if target:
                for tr in target.select("tbody tr"):
                    tds = tr.find_all("td")
                    if not tds:
                        continue
                    txts = [norm(td.get_text()) for td in tds]
                    a = tds[0].find("a")
                    link = a["href"] if a and a.has_attr("href") else ""
                    row = {
                        "company": txts[0] if len(txts) > 0 else "",
                        "ticker": "",
                        "price_range": "",
                        "shares": "",
                        "expected_date": txts[-1] if txts else "",
                        "market": "",
                        "href": link,
                        "underwriters": "",
                    }
                    items.append(_make_item(row))
        except Exception:
            pass

    # Normalize + dedupe
    out = []
    seen_ids = set()
    for it in items:
        it["company"] = norm(it.get("company", ""))
        it["ticker"] = norm(it.get("ticker", ""))
        it["expected_date"] = norm(it.get("expected_date", ""))
        it["id"] = build_id(it["company"], it["ticker"], it["expected_date"])
        if it["id"] and it["id"] not in seen_ids:
            seen_ids.add(it["id"])
            out.append(it)
    return out

# ---------------------- Background run ----------------------
async def do_run(force_email: bool = False):
    ipos = scrape_nasdaq_ipos()
    seen = load_cache()
    current_ids = {it["id"] for it in ipos}

    # First run: seed cache silently so you don't get a "full list" email
    if not seen and not force_email:
        save_cache(current_ids)
        return

    new_list = ipos if force_email else [it for it in ipos if it["id"] not in seen]
    if new_list:
        send_email(new_list)
        save_cache(seen.union(current_ids))

# ---------------------- Routes ----------------------
@app.get("/health")
def health():
    return {"ok": True, "source": NASDAQ_IPO_URL}


def _handle_ranking_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=500, detail=str(exc))
    raise HTTPException(status_code=400, detail=str(exc))


@app.get("/companies/rank")
def get_ranked_companies(
    top_n: int = 10,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
):
    try:
        return compute_ranking(metrics=None, top_n=top_n, sector=sector, industry=industry)
    except Exception as exc:  # noqa: BLE001
        _handle_ranking_error(exc)


@app.post("/companies/rank")
def post_ranked_companies(request: RankingRequest):
    try:
        return compute_ranking(
            metrics=request.metrics,
            top_n=request.top_n,
            sector=request.sector,
            industry=request.industry,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_ranking_error(exc)


@app.post("/alpaca/trade")
def trade_with_alpaca(request: TradeRequest):
    try:
        result = execute_trade_plan(request)
        return {"ok": True, **result}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _check_token(request: Request) -> bool:
    if not SECRET_TOKEN:
        return True
    hdr = request.headers.get("x-run-token")
    qry = request.query_params.get("token")
    return hdr == SECRET_TOKEN or qry == SECRET_TOKEN

@app.post("/run")
async def run(request: Request, background_tasks: BackgroundTasks):
    if not _check_token(request):
        raise HTTPException(status_code=401, detail="invalid token")

    # Optional query: ?force_email=1 to send a one-off list (for testing)
    force = request.query_params.get("force_email") in ("1", "true", "yes")
    background_tasks.add_task(do_run, force_email=force)
    return {"ok": True, "status": "started"}

@app.get("/run")
async def run_get(request: Request, background_tasks: BackgroundTasks):
    return await run(request, background_tasks)

@app.get("/debug")
def debug(request: Request):
    if not _check_token(request):
        raise HTTPException(status_code=401, detail="invalid token")
    return JSONResponse(scrape_nasdaq_ipos())

@app.post("/testmail")
def testmail(request: Request):
    if not _check_token(request):
        raise HTTPException(status_code=401, detail="invalid token")
    res = send_email([{
        "company": "TestCo",
        "ticker": "TEST",
        "price_range": "$10–$12",
        "shares": "1,000,000",
        "expected_date": "TBD",
        "market": "NASDAQ",
        "href": NASDAQ_IPO_URL,
        "underwriters": "Demo Bank, Example Securities"
    }])
    return {"email": res}

@app.get("/testmail")
def testmail_get(request: Request):
    return testmail(request)
