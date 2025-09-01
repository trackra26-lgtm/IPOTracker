import os, re, json, smtplib, asyncio
from email.mime.text import MIMEText
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup


# ---------- Config ----------
NASDAQ_IPO_URL = os.environ.get("NASDAQ_IPO_URL", "https://www.nasdaq.com/market-activity/ipos")
CACHE_PATH = Path("seen_ipos.json")
SECRET_TOKEN = os.environ.get("SECRET_TOKEN")  # set in Render

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER or "notifier@example.com")
TO_EMAILS = [e.strip() for e in os.environ.get("TO_EMAILS", "").split(",") if e.strip()]

# Detail-page scraping caps (keep runs snappy)
DETAIL_TIMEOUT_MS = int(os.environ.get("DETAIL_TIMEOUT_MS", "7000"))  # 7s per detail page
MAX_DETAIL_PAGES = int(os.environ.get("MAX_DETAIL_PAGES", "12"))      # limit in case calendar is huge

app = FastAPI(title="IPO Tracker (Nasdaq)")

# ---------- Helpers ----------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:160]

def load_cache() -> set:
    if CACHE_PATH.exists():
        try:
            return set(json.loads(CACHE_PATH.read_text()))
        except Exception:
            return set()
    return set()

def save_cache(ids: set):
    CACHE_PATH.write_text(json.dumps(sorted(list(ids)), indent=2))

def build_id(company: str, ticker: str, expected_date: str) -> str:
    # Ticker (if present) + expected date make this robust; fall back to company+date
    basis = f"{ticker or company}|{expected_date}"
    return slugify(basis)

def send_email(new_ipos: list):
    if not (SMTP_USER and SMTP_PASS and TO_EMAILS):
        return {"sent": False, "reason": "SMTP env vars missing"}

    subject = f"[IPO Tracker] {len(new_ipos)} new IPO(s) added"
    lines = [f"Detected at: {datetime.utcnow().isoformat()}Z", f"Source: {NASDAQ_IPO_URL}", ""]
    for it in new_ipos:
        title = f"{it.get('company','')}" + (f" ({it.get('ticker')})" if it.get("ticker") else "")
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

def _nasdaq_headers():
    # Nasdaq’s API/CDN likes a real browser UA + referer
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/market-activity/ipos",
        "Origin": "https://www.nasdaq.com",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

def _norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "").strip())

def _make_item(row):
    # Build a unified dict from various possible shapes
    company = _norm(row.get("company", "") or row.get("companyName", "") or row.get("name", ""))
    ticker  = _norm(row.get("symbol", "") or row.get("ticker", ""))
    price   = _norm(row.get("priceRange", "") or row.get("price_range", ""))
    shares  = _norm(row.get("shares", "") or row.get("sharesOffered", "") or row.get("shares_offered", ""))
    date    = _norm(row.get("expectedDate", "") or row.get("expected_date", "") or row.get("date", ""))
    market  = _norm(row.get("market", "") or row.get("exchange", ""))
    href    = row.get("href", "") or row.get("link", "") or ""
    # Underwriters may be a string or list
    uw = row.get("leadUnderwriters") or row.get("underwriters") or row.get("lead_underwriters") or ""
    if isinstance(uw, list):
        uw = ", ".join([_norm(x) for x in uw])
    banks = _norm(uw)

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

def scrape_nasdaq_ipos():
    """
    Try Nasdaq's JSON API first (fast & reliable), then fall back to parsing HTML table
    (server-side content) if the API format changes.
    """
    items = []

    # --- Attempt 1: Nasdaq JSON API variants ---
    session = requests.Session()
    session.headers.update(_nasdaq_headers())
    api_candidates = [
        # common endpoint (often returns upcoming & priced blocks)
        "https://api.nasdaq.com/api/ipo/calendar",
        # monthly slices (try current month)
        f"https://api.nasdaq.com/api/ipo/calendar?date={datetime.utcnow():%Y-%m}",
        # generic "all" (some deployments accept it)
        "https://api.nasdaq.com/api/ipo/calendar?date=all",
    ]

    for url in api_candidates:
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            # Known shapes seen historically:
            # 1) data["upcoming"]["rows"] / data["priced"]["rows"]
            # 2) data["upcomingTable"]["rows"] / data["pricedTable"]["rows"]
            # 3) data["data"]["rows"] (single table)
            blocks = []
            d = data.get("data", {})
            for key in ("upcoming", "upcomingTable", "priced", "pricedTable"):
                table = d.get(key) or {}
                rows = table.get("rows")
                if isinstance(rows, list) and rows:
                    blocks.extend(rows)
            # single-table fallback
            if not blocks and isinstance(d.get("rows"), list):
                blocks = d["rows"]

            for row in blocks:
                # Rows may contain "values" dict or be flat; flatten common shapes
                payload = {}
                if isinstance(row, dict):
                    if "values" in row and isinstance(row["values"], dict):
                        payload = row["values"]
                    else:
                        payload = row
                item = _make_item(payload)
                # Ensure we have at least company + date
                if item["company"] or item["ticker"]:
                    items.append(item)
            if items:
                break
        except Exception:
            continue

    # --- Attempt 2: Fallback HTML parse (no JS) ---
    if not items:
        try:
            html = session.get("https://www.nasdaq.com/market-activity/ipos", timeout=25).text
            soup = BeautifulSoup(html, "html.parser")
            # Find a table with columns "Company" and "Expected Date" (or similar)
            tables = soup.select("table")
            target = None
            for tbl in tables:
                headers = [(_norm(th.get_text())).lower() for th in tbl.select("thead th, tr th")]
                if any("company" in h for h in headers) and (any("expected" in h for h in headers) or any("date" in h for h in headers)):
                    target = tbl
                    break
            if target:
                for tr in target.select("tbody tr"):
                    tds = tr.find_all("td")
                    if not tds:
                        continue
                    txts = [_norm(td.get_text()) for td in tds]
                    a = tds[0].find("a")
                    link = a["href"] if a and a.has_attr("href") else ""
                    row = {
                        "company": txts[0] if len(txts) > 0 else "",
                        "ticker": "",  # sometimes not in the table; could be elsewhere
                        "price_range": "",
                        "shares": "",
                        "expected_date": txts[-1] if txts else "",
                        "market": "",
                        "href": link,
                        "underwriters": "",  # not usually present in SSR table
                    }
                    items.append(_make_item(row))
        except Exception:
            pass

    # Final normalization + dedupe by our existing ID function
    out = []
    seen_ids = set()
    for it in items:
        it["company"] = _norm(it.get("company", ""))
        it["ticker"] = _norm(it.get("ticker", ""))
        it["expected_date"] = _norm(it.get("expected_date", ""))
        it["id"] = build_id(it["company"], it["ticker"], it["expected_date"])
        if it["id"] and it["id"] not in seen_ids:
            seen_ids.add(it["id"])
            out.append(it)
    return out


# ---------- Background job ----------
async def do_run():
    ipos = await scrape_nasdaq_ipos()
    seen = load_cache()

    # Silent-seed on first run to avoid "email all" noise
    if not seen:
        save_cache({it["id"] for it in ipos})
        return

    current_ids = {it["id"] for it in ipos}
    new_list = [it for it in ipos if it["id"] not in seen]

    if new_list:
        send_email(new_list)
        save_cache(seen.union(current_ids))

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "source": NASDAQ_IPO_URL}

def _check_token(request: Request):
    if not SECRET_TOKEN:
        return True
    return request.headers.get("x-run-token") == SECRET_TOKEN or request.query_params.get("token") == SECRET_TOKEN

@app.post("/run")
async def run(request: Request, background_tasks: BackgroundTasks):
    if not _check_token(request):
        raise HTTPException(status_code=401, detail="invalid token")
    background_tasks.add_task(do_run)
    return {"ok": True, "status": "started"}

@app.get("/run")
async def run_get(request: Request, background_tasks: BackgroundTasks):
    return await run(request, background_tasks)

@app.post("/testmail")
async def testmail(request: Request):
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
async def testmail_get(request: Request):
    return await testmail(request)
