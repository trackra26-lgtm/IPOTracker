import os, re, json, smtplib, asyncio
from email.mime.text import MIMEText
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from pathlib import Path
from datetime import datetime

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

# ---------- Scraper ----------
async def scrape_nasdaq_ipos():
    """
    Scrape Nasdaq IPO calendar.
    For each row: company, ticker, price range, shares, expected date, market, link.
    Then open each detail link (capped / timeboxed) to extract lead underwriters if available.
    Returns: list of IPO dicts.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()

        await page.goto(NASDAQ_IPO_URL, wait_until="networkidle", timeout=120000)

        # The calendar is a table. We'll query rows via JS to be resilient to minor DOM shifts.
        rows = await page.evaluate("""
        () => {
          const data = [];
          // Look for any table with the IPO columns (Company, Symbol, Price Range, Shares, Expected, Market)
          const tables = Array.from(document.querySelectorAll('table'));
          const target = tables.find(tbl => {
            const ths = Array.from(tbl.querySelectorAll('thead th, tr th')).map(th => th.textContent.trim().toLowerCase());
            return ths.some(h => h.includes('company')) &&
                   ths.some(h => h.includes('symbol')) &&
                   (ths.some(h => h.includes('expected')) || ths.some(h => h.includes('date')));
          });
          if (!target) return data;

          const headerRow = target.querySelector('thead tr') || target.querySelector('tr');
          const headers = Array.from(headerRow.querySelectorAll('th')).map(th => th.textContent.trim());
          const idx = {}; headers.forEach((h,i)=> idx[h.toLowerCase()] = i);

          function colIdx(opts) {
            const keys = Object.keys(idx);
            for (const opt of opts) {
              const k = keys.find(k => k.includes(opt));
              if (k != null) return idx[k];
            }
            return -1;
          }

          const iCompany = colIdx(['company']);
          const iSymbol  = colIdx(['symbol','ticker']);
          const iPrice   = colIdx(['price range','price']);
          const iShares  = colIdx(['shares']);
          const iDate    = colIdx(['expected','date']);
          const iMarket  = colIdx(['market','exchange']);

          const trs = Array.from(target.querySelectorAll('tbody tr'));
          for (const tr of trs) {
            const tds = Array.from(tr.querySelectorAll('td'));
            if (!tds.length) continue;

            const txt = i => (i>=0 && i<tds.length) ? (tds[i].textContent||'').trim() : '';

            let href = '';
            if (iCompany >= 0 && iCompany < tds.length) {
              const a = tds[iCompany].querySelector('a');
              if (a && a.href) href = a.href;
            }

            data.push({
              company: txt(iCompany),
              ticker:  txt(iSymbol),
              price_range: txt(iPrice),
              shares: txt(iShares),
              expected_date: txt(iDate),
              market: txt(iMarket),
              href
            });
          }
          return data;
        }
        """)

        # Visit detail pages (capped) to extract lead underwriters
        # We timebox each detail to keep the whole run under cron time budgets.
        detailed = []
        for i, row in enumerate(rows):
            if i >= MAX_DETAIL_PAGES:
                detailed.append({**row, "underwriters": ""})
                continue
            href = row.get("href", "")
            underwriters = ""
            if href:
                try:
                    dp = await ctx.new_page()
                    await dp.goto(href, wait_until="domcontentloaded", timeout=DETAIL_TIMEOUT_MS)
                    # Try common labels that appear on Nasdaq detail pages
                    # We'll search for elements whose text contains 'Underwriter' and read nearby content.
                    # Fallbacks try different patterns to be robust.
                    underwriters = await dp.evaluate("""
                    () => {
                      const txt = (el) => (el && el.textContent || '').trim();
                      // Strategy 1: find any dt/dd pair like "Lead underwriters" / value
                      const dts = Array.from(document.querySelectorAll('dt,th'));
                      for (const dt of dts) {
                        const t = txt(dt).toLowerCase();
                        if (t.includes('underwriter')) {
                          // try next sibling (dd/td) or parent row's next cell
                          let val = '';
                          const dd = dt.nextElementSibling;
                          if (dd) val = txt(dd);
                          if (!val && dt.parentElement) {
                              const tds = dt.parentElement.querySelectorAll('td,dd');
                              if (tds.length >= 2) val = txt(tds[1]);
                          }
                          if (val) return val;
                        }
                      }
                      // Strategy 2: look for label blocks
                      const labels = Array.from(document.querySelectorAll('*')).filter(el => /underwriter/i.test(txt(el)));
                      for (const el of labels) {
                        // check siblings
                        const sib = el.nextElementSibling;
                        if (sib && txt(sib).length > 2) return txt(sib);
                      }
                      return '';
                    }
                    """)
                    await dp.close()
                except PWTimeout:
                    underwriters = ""
                except Exception:
                    underwriters = ""
            detailed.append({**row, "underwriters": norm(underwriters)})

        await browser.close()

    # Normalize and dedupe
    items = []
    for r in detailed:
        company = norm(r.get("company",""))
        ticker  = norm(r.get("ticker",""))
        price   = norm(r.get("price_range",""))
        shares  = norm(r.get("shares",""))
        date    = norm(r.get("expected_date",""))
        market  = norm(r.get("market",""))
        href    = r.get("href","").strip()
        banks   = norm(r.get("underwriters",""))

        # Build item
        items.append({
            "id": build_id(company, ticker, date),
            "company": company,
            "ticker": ticker,
            "price_range": price,
            "shares": shares,
            "expected_date": date,
            "market": market,
            "href": href,
            "underwriters": banks
        })

    # Deduplicate by id
    uniq = {}
    for it in items:
        uniq[it["id"]] = it
    return list(uniq.values())

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
