"""One-shot audit script per strategy: runs oracle table + impl mocks, writes to SCRATCH."""
import json
import os
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

# ensure src + tests for oracles
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from schwab_positions_mcp.twrr_calc import build_subperiods, compute_linked_twrr
from twrr_oracles import ORACLES, build_from_oracle
from schwab_positions_mcp.tools.analytics import get_twrr_analysis_impl

SCRATCH = os.environ.get("SCRATCH", "/tmp/grok-goal-c5c02295a091/implementer")
os.makedirs(SCRATCH, exist_ok=True)

results = {"oracles": [], "impl": []}

for o in ORACLES:
    subs = build_from_oracle(o)
    lnk = compute_linked_twrr(subs, "2026-01-01", "2026-12-31")
    hprs = [round(s.hpr, 6) for s in subs]
    results["oracles"].append({
        "name": o.name,
        "hprs": hprs,
        "linked": lnk,
        "sub_count": len(subs),
    })

# impl mocks for a few
def _mk_resp(payload):
    m = MagicMock()
    m.status_code = 200
    m.headers = {}
    m.json.return_value = payload
    m.text = ""
    return m

# multi buy case - use valid hash to avoid model validation error
mock = MagicMock()
td = date.today()
pos = {"securitiesAccount": {"positions": [{"instrument": {"symbol": "AAPL"}, "longQuantity": 20, "averagePrice": 150, "marketValue": 3500}]}}
txs = [
    {"type": "TRADE", "instrument": {"symbol": "AAPL"}, "netAmount": -1500, "quantity": 10, "price": 150, "tradeDate": (td - timedelta(25)).isoformat()},
    {"type": "TRADE", "instrument": {"symbol": "AAPL"}, "netAmount": -1500, "quantity": 10, "price": 150, "tradeDate": (td - timedelta(10)).isoformat()},
]
mock.get_account.return_value = _mk_resp(pos)
mock.get_transactions.return_value = _mk_resp(txs)
with patch("schwab_positions_mcp.tools.analytics.get_client", return_value=mock):
    out = get_twrr_analysis_impl({"account_hash": "ACCT_HASH_AAAAAAAAAAAA", "symbol": "AAPL", "lookback_days": 60})
results["impl"].append({"case": "multi_buy", "out": out})

with open(os.path.join(SCRATCH, "twrr_audit.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

# also review_edges like
with open(os.path.join(SCRATCH, "review_edges.txt"), "w") as f:
    f.write("audit script output:\n")
    for r in results["oracles"]:
        f.write(f"{r['name']}: hprs={r['hprs']} linked={r['linked']}\n")
    f.write(f"impl multi: {results['impl'][0]['out'].get('twrr_30d')}\n")

print("wrote to", SCRATCH)
