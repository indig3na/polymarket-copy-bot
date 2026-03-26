"""
Polymarket Copy Trading Bot
Monitors a target wallet and instantly mirrors its trades on your own wallet.
"""

import json
import logging
import os
import time
import threading
import signal
import sys
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import yaml
import requests
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from web3 import Web3

from py_clob_client.client import ClobClient
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.clob_types import OrderArgs, OrderType


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("copybot")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = 137  # Polygon mainnet

USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CLOB_CONTRACT = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
USDC_DECIMALS = 6


# ---------------------------------------------------------------------------
# ERC-20 ABI (for allowance checking)
# ---------------------------------------------------------------------------

ERC20_ABI = [
    {
        "inputs": [{"name": "owner"}, {"name": "spender"}],
        "name": "allowance",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "spender"}, {"name": "amount"}],
        "name": "approve",
        "outputs": [{"type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "account"}],
        "name": "balanceOf",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    id: str
    wallet: str
    token_id: str
    side: str
    price: float
    size: float
    market_question: str
    timestamp: datetime
    mirrored: bool = False
    mirror_error: Optional[str] = None


@dataclass
class Position:
    token_id: str
    market_question: str
    size: float
    avg_price: float
    side: str


@dataclass
class BotState:
    watched_wallet_positions: dict[str, Position] = field(default_factory=dict)
    my_positions: dict[str, Position] = field(default_factory=dict)
    trade_log: list[Trade] = field(default_factory=list)
    watched_trades: list[Trade] = field(default_factory=list)
    running: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_token_amount(amount: int, decimals: int = 18) -> float:
    return amount / (10 ** decimals)


def get_rpc_w3() -> Web3:
    rpc = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to Polygon RPC: {rpc}")
    return w3


# ---------------------------------------------------------------------------
# Copy Trading Engine
# ---------------------------------------------------------------------------

class CopyEngine:
    def __init__(
        self,
        clob_client: ClobClient,
        target_wallet: str,
        your_address: str,
        cfg: dict,
    ):
        self.clob = clob_client
        self.target = target_wallet.lower()
        self.your_address = your_address.lower()
        self.cfg = cfg
        self.state = BotState()
        self.lock = threading.Lock()
        self.poll_interval = cfg.get("poll_interval", 2.0)
        self.size_mult = cfg.get("position_size_multiplier", 1.0)
        self.max_spend = cfg.get("max_spend_per_trade", 50.0)
        self.max_slippage = cfg.get("max_slippage", 0.02)
        self._poll_thread: Optional[threading.Thread] = None
        self._seen_fill_ids: set[str] = set()

    # ---- Lifecycle -----------------------------------------------------------

    def start(self):
        log.info(f"Starting copy engine — mirroring wallet: {self.target}")
        self.state.running = True
        self._fetch_initial_state()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop(self):
        log.info("Stopping copy engine...")
        self.state.running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    # ---- Initial snapshot ---------------------------------------------------

    def _fetch_initial_state(self):
        try:
            fills = self.clob.get_fills(address=self.target)
            if fills:
                self._seen_fill_ids = {f["orderID"] for f in fills[-100:]}
                log.info(f"Tracking {len(self._seen_fill_ids)} recent fills")
        except Exception as e:
            log.warning(f"Could not fetch initial fills: {e}")

        try:
            positions = self.clob.get_asset_positions(address=self.target)
            for p in positions:
                bal = float(p.get("balance", 0))
                if bal > 0:
                    self.state.watched_wallet_positions[p["tokenId"]] = Position(
                        token_id=p["tokenId"],
                        market_question=p.get("marketQuestion", "Unknown"),
                        size=bal,
                        avg_price=float(p.get("avgPrice", 0)),
                        side=p.get("side", "BUY"),
                    )
            log.info(f"Watched wallet has {len(self.state.watched_wallet_positions)} positions")
        except Exception as e:
            log.warning(f"Could not fetch watched wallet positions: {e}")

    # ---- Polling loop -------------------------------------------------------

    def _poll_loop(self):
        consecutive_errors = 0
        while self.state.running:
            try:
                self._check_for_new_trades()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                log.error(f"Poll error ({consecutive_errors}): {e}")
                if consecutive_errors >= 5:
                    log.error("Too many errors, pausing 30s...")
                    time.sleep(30)
            time.sleep(self.poll_interval)

    def _check_for_new_trades(self):
        try:
            fills = self.clob.get_fills(address=self.target)
        except Exception as e:
            log.debug(f"Fill fetch error: {e}")
            return

        new_fills = [f for f in fills if f["orderID"] not in self._seen_fill_ids]

        for fill in new_fills:
            self._seen_fill_ids.add(fill["orderID"])
            trade = self._parse_fill(fill)
            if not trade:
                continue

            with self.lock:
                self.state.watched_trades.append(trade)
                if len(self.state.watched_trades) > 500:
                    self.state.watched_trades = self.state.watched_trades[-500:]

            log.info(
                f"🔔 TRADE — {trade.wallet[:8]}… {trade.side} "
                f"{trade.size:.4f} @ ${trade.price:.4f} | {trade.market_question[:60]}"
            )
            self._mirror_trade(trade)

    def _parse_fill(self, fill: dict) -> Optional[Trade]:
        try:
            side = fill.get("side", "").upper()
            if side not in ("BUY", "SELL"):
                return None
            ts = fill["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return Trade(
                id=fill["orderID"],
                wallet=fill.get("address", self.target),
                token_id=fill["tokenId"],
                side=side,
                price=float(fill.get("price", 0)),
                size=float(fill.get("size", 0)),
                market_question=fill.get("marketQuestion", fill.get("conditionId", "Unknown")),
                timestamp=ts,
            )
        except Exception as e:
            log.debug(f"Could not parse fill: {e}")
            return None

    # ---- Mirror logic -------------------------------------------------------

    def _mirror_trade(self, trade: Trade):
        try:
            size = trade.size * self.size_mult
            price = trade.price

            # Cap by max spend
            cost = size * price
            if cost > self.max_spend:
                size = self.max_spend / price
                log.info(f"  → Sized down to {size:.4f} (max spend ${self.max_spend})")

            if size < 0.001:
                log.info(f"  → Size too small, skipping")
                return

            # Check slippage
            try:
                book = self.clob.get_order_book(trade.token_id)
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                best_bid = float(bids[0]["price"]) if bids else price
                best_ask = float(asks[0]["price"]) if asks else price

                if trade.side == "BUY":
                    worst = best_ask * (1 + self.max_slippage)
                    if price > worst:
                        log.warning(f"  → Slippage too high! wanted {price}, best ask {best_ask}")
                        return
                else:
                    worst = best_bid * (1 - self.max_slippage)
                    if price < worst:
                        log.warning(f"  → Slippage too high! wanted {price}, best bid {best_bid}")
                        return
            except Exception as e:
                log.debug(f"  → Could not check order book: {e}")

            # Get market tick size
            tick_size = 0.01
            neg_risk = False
            try:
                book = self.clob.get_order_book(trade.token_id)
                tick_size = float(book.get("tickSize", 0.01))
            except Exception:
                pass

            # Place the order
            side = BUY if trade.side == "BUY" else SELL
            log.info(f"  → Mirroring {trade.side} {size:.4f} @ ${price:.4f}")

            result = self.clob.create_and_post_order(
                OrderArgs(
                    token_id=trade.token_id,
                    price=price,
                    size=size,
                    side=side,
                    order_type=OrderType.GTC,
                ),
                options={
                    "tick_size": str(tick_size),
                    "neg_risk": neg_risk,
                },
            )
            order_id = result.get("orderID") or result.get("orderID") or str(result)[:80]
            log.info(f"  ✅ Mirrored: {order_id}")
            trade.mirrored = True

        except Exception as e:
            trade.mirror_error = str(e)
            log.error(f"  ❌ Mirror failed: {e}")

    def sync_positions(self):
        try:
            positions = self.clob.get_asset_positions(address=self.your_address)
            with self.lock:
                self.state.my_positions.clear()
                for p in positions:
                    bal = float(p.get("balance", 0))
                    if bal > 0:
                        self.state.my_positions[p["tokenId"]] = Position(
                            token_id=p["tokenId"],
                            market_question=p.get("marketQuestion", "Unknown"),
                            size=bal,
                            avg_price=float(p.get("avgPrice", 0)),
                            side=p.get("side", "BUY"),
                        )
        except Exception as e:
            log.error(f"Position sync failed: {e}")


# ---------------------------------------------------------------------------
# Web Dashboard
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

_copy_engine: Optional[CopyEngine] = None


@app.route("/")
def dashboard():
    with _copy_engine.lock:
        watched = _copy_engine.state.watched_wallet_positions
        mine    = _copy_engine.state.my_positions
        trades  = list(reversed(_copy_engine.state.watched_trades[-50:]))
        total   = len(_copy_engine.state.watched_trades)
        mirrored = sum(1 for t in _copy_engine.state.watched_trades if t.mirrored)

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Polymarket Copy Bot</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0d1117; color: #e6edf3; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; font-size: 1.4em; }
        h2 { color: #8b949e; font-size: 0.8em; text-transform: uppercase;
             letter-spacing: 0.1em; margin: 24px 0 10px;
             border-bottom: 1px solid #21262d; padding-bottom: 5px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; text-align: center; }
        .stat { font-size: 2em; font-weight: 700; color: #58a6ff; }
        .stat-label { font-size: 0.7em; color: #8b949e; text-transform: uppercase; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
        th { color: #8b949e; text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262d; }
        td { padding: 6px 8px; border-bottom: 1px solid #21262d; }
        tr:last-child td { border: none; }
        .mono { font-family: 'Courier New', monospace; }
        .buy { color: #3fb950; } .sell { color: #f85149; }
        .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }
        .b-buy  { background: #0f2d14; color: #3fb950; }
        .b-sell { background: #2d1014; color: #f85149; }
        .b-ok   { background: #0f2d14; color: #3fb950; }
        .b-err  { background: #2d1014; color: #f85149; }
        .b-pend { background: #2d2206; color: #d29922; }
        .wrap { max-width: 1100px; margin: 0 auto; }
        .sub  { font-size: 0.75em; color: #8b949e; margin-top: 2px; }
      </style>
    </head>
    <body>
    <div class="wrap">
      <h1>🦞 Polymarket Copy Trading Bot</h1>
      <div class="grid">
        <div class="card"><div class="stat">{{ wc }}</div><div class="stat-label">Watched Positions</div></div>
        <div class="card"><div class="stat">{{ mc }}</div><div class="stat-label">My Positions</div></div>
        <div class="card"><div class="stat">{{ tt }}</div><div class="stat-label">Trades Detected</div></div>
        <div class="card"><div class="stat">{{ mr }}</div><div class="stat-label">Mirrored ✓</div></div>
      </div>

      <h2>Watched Wallet Positions</h2>
      {% if wp %}
      <table><tr><th>Token</th><th>Side</th><th>Size</th><th>Avg Price</th><th>Market</th></tr>
        {% for p in wp %}
        <tr>
          <td class="mono">{{ p.token_id[:16] }}…</td>
          <td><span class="badge b-{{ p.side.lower() }}">{{ p.side }}</span></td>
          <td>{{ "%.4f"|format(p.size) }}</td>
          <td>${{ "%.4f"|format(p.avg_price) }}</td>
          <td>{{ p.market_question[:50] }}</td>
        </tr>
        {% endfor %}
      </table>{% else %}<p style="color:#8b949e;font-size:0.85em">No open positions</p>{% endif %}

      <h2>My Positions</h2>
      {% if mp %}
      <table><tr><th>Token</th><th>Side</th><th>Size</th><th>Avg Price</th><th>Market</th></tr>
        {% for p in mp %}
        <tr>
          <td class="mono">{{ p.token_id[:16] }}…</td>
          <td><span class="badge b-{{ p.side.lower() }}">{{ p.side }}</span></td>
          <td>{{ "%.4f"|format(p.size) }}</td>
          <td>${{ "%.4f"|format(p.avg_price) }}</td>
          <td>{{ p.market_question[:50] }}</td>
        </tr>
        {% endfor %}
      </table>{% else %}<p style="color:#8b949e;font-size:0.85em">No positions yet — mirror a trade to get started</p>{% endif %}

      <h2>Recent Trades ({{ tt }} total)</h2>
      {% if trades %}
      <table><tr><th>Time</th><th>Token</th><th>Side</th><th>Size</th><th>Price</th><th>Status</th></tr>
        {% for t in trades %}
        <tr>
          <td class="mono">{{ t.timestamp.strftime("%H:%M:%S") if t.timestamp else "—" }}</td>
          <td class="mono">{{ t.token_id[:16] }}…</td>
          <td><span class="badge b-{{ t.side.lower() }}">{{ t.side }}</span></td>
          <td>{{ "%.4f"|format(t.size) }}</td>
          <td>${{ "%.4f"|format(t.price) }}</td>
          <td>
            {% if t.mirrored %}<span class="badge b-ok">✓ Mirrored</span>
            {% elif t.mirror_error %}<span class="badge b-err">✗ {{ t.mirror_error[:40] }}</span>
            {% else %}<span class="badge b-pend">Pending</span>{% endif %}
          </td>
        </tr>
        {% endfor %}
      </table>{% else %}<p style="color:#8b949e;font-size:0.85em">No trades detected yet</p>{% endif %}
    </div>
    </body>
    </html>
    """
    from jinja2 import Template
    t = Template(html)
    return t.render(
        wc=len(watched),
        mc=len(mine),
        tt=total,
        mr=mirrored,
        wp=list(watched.values()),
        mp=list(mine.values()),
        trades=trades,
    )


@app.route("/api/state")
def api_state():
    with _copy_engine.lock:
        state = _copy_engine.state
        return jsonify({
            "running": state.running,
            "watched_wallet": _copy_engine.target,
            "my_address": _copy_engine.your_address,
            "watched_positions": {k: asdict(v) for k, v in state.watched_wallet_positions.items()},
            "my_positions": {k: asdict(v) for k, v in state.my_positions.items()},
            "watched_trades": [
                {**asdict(t), "timestamp": t.timestamp.isoformat() if t.timestamp else None}
                for t in state.watched_trades
            ],
            "stats": {
                "total": len(state.watched_trades),
                "mirrored": sum(1 for t in state.watched_trades if t.mirrored),
                "failed": sum(1 for t in state.watched_trades if t.mirror_error),
            },
        })


@app.route("/api/sync", methods=["POST"])
def api_sync():
    _copy_engine.sync_positions()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# USDC Allowance helper
# ---------------------------------------------------------------------------

def ensure_usdc_allowance(private_key: str, spender: str, min_amount: float):
    w3 = get_rpc_w3()
    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(
        Web3.to_checksum_address(USDC_CONTRACT),
        abi=ERC20_ABI,
    )
    spender_addr = Web3.to_checksum_address(spender)
    allowance = contract.functions.allowance(account.address, spender_addr).call()
    min_wei = int(min_amount * 10**USDC_DECIMALS)
    if allowance < min_wei:
        log.info(f"Approving CLOB to spend USDC...")
        tx = contract.functions.approve(spender_addr, 2**256 - 1).build_transaction({
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 80000,
            "chainId": CHAIN_ID,
        })
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        log.info(f"Approval tx: {receipt['transactionHash'].hex()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket Copy Trading Bot")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("=" * 60)
    log.info("Polymarket Copy Trading Bot")
    log.info(f"Target wallet : {cfg['watch_wallet']}")
    log.info(f"Max spend     : $%s" % cfg.get("max_spend_per_trade", 50))
    log.info(f"Size mult     : {cfg.get('position_size_multiplier', 1.0)}x")
    log.info(f"Poll interval : {cfg.get('poll_interval', 2.0)}s")
    log.info("=" * 60)

    private_key = cfg["your_private_key"]
    funder      = cfg.get("funder_address", "")

    # Build CLOB client
    client = ClobClient(
        host=CLOB_HOST,
        key=private_key,
        chain_id=CHAIN_ID,
        creds=None,
        signature_type=0,
        funder=funder,
    )

    # Derive / fetch API credentials (L1 → L2 auth)
    try:
        api_creds = client.create_or_derive_api_creds()
        log.info(f"API credentials ready")
    except Exception as e:
        log.warning(f"Could not derive API credentials: {e}")

    # Show USDC balance
    try:
        w3 = get_rpc_w3()
        account = w3.eth.account.from_key(private_key)
        contract = w3.eth.contract(Web3.to_checksum_address(USDC_CONTRACT), abi=ERC20_ABI)
        bal = contract.functions.balanceOf(account.address).call()
        bal_fmt = bal / 10**USDC_DECIMALS
        log.info(f"USDC.e balance: {bal_fmt:.4f}")
    except Exception as e:
        log.warning(f"Could not fetch USDC balance: {e}")

    # Ensure CLOB has USDC allowance
    try:
        ensure_usdc_allowance(private_key, CLOB_CONTRACT, 1000)
    except Exception as e:
        log.warning(f"Could not set USDC allowance: {e}")

    # Start engine
    global _copy_engine
    engine = CopyEngine(
        client,
        cfg["watch_wallet"],
        account.address,
        cfg,
    )
    _copy_engine = engine
    engine.start()

    def shutdown(*_):
        log.info("Shutting down...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    port = cfg.get("dashboard_port", 5050)
    log.info(f"Dashboard → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
