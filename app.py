# app.py
import os, re, json, smtplib
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

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
