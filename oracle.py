"""
polygon_alpaca_stat_arb.py - Cloud-Ready Stat Arb Bot
Uses Polygon.io for real-time data (free crypto)
Uses Alpaca for order execution
Perfect for Oracle Cloud deployment!
"""

import os
import time
from collections import deque
from datetime import datetime

import alpaca_trade_api as tradeapi
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CloudStatArbBot:
    def __init__(self):
        """Initialize with API keys from .env file"""

        # Load from environment
        polygon_key = os.getenv("POLYGON_API_KEY")
        alpaca_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.polygon_crypto_prefix = os.getenv("POLYGON_CRYPTO_PREFIX", "X:")

        # Validate keys
        if not polygon_key:
            raise ValueError("POLYGON_API_KEY not found in .env file")
        if not alpaca_key or not alpaca_secret:
            raise ValueError("Alpaca keys not found in .env file")

        # Polygon for data (free crypto)
        self.polygon_key = polygon_key
        self.polygon_base = "https://api.polygon.io"

        # Alpaca for trading
        base_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )
        self.alpaca = tradeapi.REST(
            alpaca_key, alpaca_secret, base_url, api_version="v2"
        )

        # Trading parameters (can override from .env)
        self.lookback = 60
        self.entry_z = float(os.getenv("ENTRY_THRESHOLD", "2.0"))
        self.exit_z = float(os.getenv("EXIT_THRESHOLD", "0.5"))
        self.position_size = float(os.getenv("POSITION_SIZE", "0.40"))

        # Crypto pairs (Polygon has free real-time crypto)
        self.pairs = [
            ("BTC", "ETH"),
            ("ETH", "LTC"),
        ]

        self.price_buffers = {
            "BTC": deque(maxlen=200),
            "ETH": deque(maxlen=200),
            "LTC": deque(maxlen=200),
        }

        self.active_positions = {}

        print("✅ Bot initialized")
        print("   Data: Polygon.io (free crypto)")
        print(f"   Execution: Alpaca ({'paper' if paper else 'live'})")

    def _polygon_get(self, endpoint, params):
        try:
            response = requests.get(
                f"{self.polygon_base}{endpoint}",
                params=params,
                timeout=10,
            )
        except requests.RequestException as exc:
            print(f"   Error contacting Polygon: {exc}")
            return None
        return response

    def _parse_polygon_price(self, response):
        if not response or response.status_code != 200:
            return None

        data = response.json()
        if "last" in data and "price" in data["last"]:
            return data["last"]["price"]
        if "results" in data and data["results"]:
            first = data["results"][0]
            if "c" in first:
                return first["c"]
        return None

    def get_crypto_price(self, symbol):
        """Get real-time crypto price from Polygon"""
        endpoints = [
            f"/v2/last/crypto/{symbol}/USD",
            f"/v1/last/crypto/{symbol}/USD",
            f"/v2/aggs/ticker/{self.polygon_crypto_prefix}{symbol}USD/prev",
        ]
        params = {"apiKey": self.polygon_key}

        for endpoint in endpoints:
            response = self._polygon_get(endpoint, params)
            price = self._parse_polygon_price(response)
            if price is not None:
                self.price_buffers[symbol].append(
                    {"time": datetime.now(), "price": price}
                )
                return price

            if response is not None:
                body = response.text.strip()
                if body:
                    body = body[:200]
                print(
                    "   Polygon error "
                    f"({response.status_code}) on {endpoint}: {body}"
                )

        return 0

    def calc_zscore(self, s1, s2):
        """Calculate z-score from price buffers"""
        buf1 = self.price_buffers[s1]
        buf2 = self.price_buffers[s2]

        if len(buf1) < self.lookback or len(buf2) < self.lookback:
            return 0, 0

        # Get recent prices
        p1 = np.array([x["price"] for x in list(buf1)[-self.lookback :]])
        p2 = np.array([x["price"] for x in list(buf2)[-self.lookback :]])

        # Hedge ratio
        hedge = np.cov(p1, p2)[0, 1] / np.var(p2)

        # Spread
        spread = p1 - hedge * p2

        # Z-score
        mean = np.mean(spread)
        std = np.std(spread)

        if std == 0:
            return 0, hedge

        z = (spread[-1] - mean) / std

        return z, hedge

    def get_account_info(self):
        """Get Alpaca account info"""
        try:
            account = self.alpaca.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            }
        except Exception as e:
            print(f"   Error: {e}")
            return {"equity": 0, "cash": 0, "buying_power": 0}

    def get_position_qty(self, symbol):
        """Get Alpaca position"""
        try:
            # Alpaca uses X-USD format
            alpaca_symbol = f"{symbol}USD"
            position = self.alpaca.get_position(alpaca_symbol)
            return float(position.qty)
        except Exception:
            return 0

    def place_order(self, symbol, side, amount_usd):
        """Place order on Alpaca"""
        try:
            alpaca_symbol = f"{symbol}USD"

            # Alpaca crypto orders use notional (dollar amount)
            self.alpaca.submit_order(
                symbol=alpaca_symbol,
                notional=amount_usd,
                side=side,
                type="market",
                time_in_force="gtc",
            )

            print(f"   📤 {side} ${amount_usd} of {symbol}")
            return True

        except Exception as e:
            print(f"   ❌ Order error: {e}")
            return False

    def trade_pair(self, s1, s2, signal):
        """Execute pair trade"""
        account = self.get_account_info()
        capital_per_side = account["cash"] * self.position_size

        if capital_per_side < 100:
            print("   ⚠️  Insufficient capital")
            return

        if signal == "long_spread":
            # Buy S1, sell S2 not supported on Alpaca crypto
            # So just buy undervalued one
            self.place_order(s1, "buy", capital_per_side)
            print(f"   📈 Long {s1}")
        elif signal == "short_spread":
            # Buy S2 (relatively undervalued)
            self.place_order(s2, "buy", capital_per_side)
            print(f"   📈 Long {s2}")

        self.active_positions[f"{s1}_{s2}"] = signal

    def close_pair(self, s1, s2):
        """Close positions"""
        q1 = self.get_position_qty(s1)
        q2 = self.get_position_qty(s2)

        try:
            if q1 > 0:
                alpaca_symbol = f"{s1}USD"
                self.alpaca.submit_order(
                    symbol=alpaca_symbol,
                    qty=q1,
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                print(f"   ✅ Sold {q1} {s1}")

            if q2 > 0:
                alpaca_symbol = f"{s2}USD"
                self.alpaca.submit_order(
                    symbol=alpaca_symbol,
                    qty=q2,
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                print(f"   ✅ Sold {q2} {s2}")

            pair_key = f"{s1}_{s2}"
            if pair_key in self.active_positions:
                del self.active_positions[pair_key]

        except Exception as e:
            print(f"   ❌ Error: {e}")

    def run(self):
        """Main loop - runs 24/7!"""
        print("\n" + "=" * 60)
        print("☁️  CLOUD STAT ARB BOT - 24/7")
        print("=" * 60)
        print(f"Pairs: {self.pairs}")
        print("Data source: Polygon.io (free real-time crypto)")
        print("Execution: Alpaca")
        print("Check interval: 5 seconds")
        print("=" * 60)

        iteration = 0

        while True:
            try:
                iteration += 1

                if iteration % 12 == 0:  # Print every minute
                    print(f"\n{'=' * 60}")
                    print(
                        f"⏰ Iteration {iteration} - "
                        f"{datetime.now().strftime('%H:%M:%S')}"
                    )
                    account = self.get_account_info()
                    print(f"💰 Equity: ${account['equity']:.2f}")

                # Update prices
                for symbol in ["BTC", "ETH", "LTC"]:
                    price = self.get_crypto_price(symbol)
                    if price == 0:
                        print(f"   ⚠️  Failed to get {symbol} price")

                # Analyze pairs
                for s1, s2 in self.pairs:
                    z, hedge = self.calc_zscore(s1, s2)

                    if iteration % 12 == 0:
                        print(f"   {s1}/{s2}: Z = {z:.2f}")

                    pair_key = f"{s1}_{s2}"
                    has_position = pair_key in self.active_positions

                    if not has_position:
                        if z > self.entry_z:
                            print(f"\n🎯 SHORT SPREAD signal: {s1}/{s2}")
                            self.trade_pair(s1, s2, "short_spread")
                        elif z < -self.entry_z:
                            print(f"\n🎯 LONG SPREAD signal: {s1}/{s2}")
                            self.trade_pair(s1, s2, "long_spread")
                    else:
                        if abs(z) < self.exit_z:
                            print(f"\n🔄 EXIT signal: {s1}/{s2}")
                            self.close_pair(s1, s2)

                time.sleep(5)

            except KeyboardInterrupt:
                print("\n\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(30)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("☁️  CLOUD-READY STAT ARB BOT")
    print("=" * 60)
    print("\n📋 Loading configuration from .env file...")

    try:
        bot = CloudStatArbBot()

        print("\n✅ Configuration loaded:")
        print(f"   Polygon API: {bot.polygon_key[:10]}...")
        print(f"   Entry Z-score: ±{bot.entry_z}")
        print(f"   Exit Z-score: ±{bot.exit_z}")
        print(f"   Position size: {bot.position_size*100}%")
        print("\n🚀 Starting bot...")
        print("=" * 60)

        bot.run()

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n📝 Please create a .env file with:")
        print("   POLYGON_API_KEY=your_key")
        print("   ALPACA_API_KEY=your_key")
        print("   ALPACA_SECRET_KEY=your_secret")
        print("   PAPER_TRADING=true")
    except Exception as e:
        print(f"\n❌ Error: {e}")
