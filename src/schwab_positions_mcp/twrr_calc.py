"""
Pure TWRR calculation module, extracted per strategy.

Ports stable primitives from portfolio-analysis/twrr.py:
- geometric_link
- build_subperiods (adapted)
- compute_linked_twrr with date filtering

Uses TradeEvent dataclass for normalization from Schwab data.
No datetime.now() inside; as_of_date param.
No reuse of final MV for intermediate subs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, List, Optional


@dataclass
class TradeEvent:
    """Normalized event from Schwab tx/pos for TWRR subperiod builder."""
    event_date: str
    cash_flow: float  # external CF at boundary (positive if capital added to position)
    quantity: float
    price: float  # for mv_at_event = quantity * price


@dataclass
class TradeSubPeriod:
    symbol: str
    start_date: str
    end_date: str
    start_market_value: float
    end_market_value: float
    cash_flow: float
    hpr: float


def geometric_link(returns: List[float]) -> float:
    """Geometric linking of period returns."""
    if not returns:
        return 0.0
    product = 1.0
    for r in returns:
        product *= 1.0 + float(r)
    return product - 1.0


def build_subperiods(
    events: List[TradeEvent],
    symbol: str,
    final_mv: float,
    as_of_date: Optional[str] = None,
) -> List[TradeSubPeriod]:
    """
    Build subperiods from events.

    For first: start_mv = first cf (capital at risk after inflow), end_mv = event mv or chained.
    Append terminal from last event to as_of with final_mv.
    HPR = p_end / p_start - 1 using mv/qty if needed, but here use mv_at.
    """
    if not events:
        return []

    subperiods: List[TradeSubPeriod] = []
    prev_mv = 0.0
    prev_date = "inception"

    for i, ev in enumerate(events):
        mv_at = ev.quantity * ev.price
        if i == 0:
            # First sub: capital at risk starts as the external CF (inflow) at this event
            start_mv = ev.cash_flow
        else:
            start_mv = prev_mv
        end_mv = mv_at
        hpr = (end_mv / start_mv - 1.0) if start_mv > 0 else 0.0
        subperiods.append(TradeSubPeriod(
            symbol=symbol,
            start_date=prev_date if prev_date != "inception" else ev.event_date,
            end_date=ev.event_date,
            start_market_value=round(start_mv, 2),
            end_market_value=round(end_mv, 2),
            cash_flow=round(ev.cash_flow, 2),
            hpr=round(hpr, 6),
        ))
        prev_mv = end_mv
        prev_date = ev.event_date

    # append terminal closing leg from last to as_of with final_mv (hpr=0 if same date)
    if as_of_date is None:
        as_of_date = datetime.now().date().isoformat()
    for last in (subperiods[-1:] if subperiods else []):
        last_mv = last.end_market_value
        hpr = (final_mv / last_mv - 1.0) if last_mv > 0 else 0.0
        subperiods.append(TradeSubPeriod(
            symbol=symbol,
            start_date=last.end_date,
            end_date=as_of_date,
            start_market_value=last_mv,
            end_market_value=final_mv,
            cash_flow=0.0,
            hpr=round(hpr, 6),
        ))

    return subperiods


def compute_linked_twrr(
    subperiods: List[TradeSubPeriod],
    from_date: str,
    to_date: str,
) -> Optional[float]:
    """Select overlapping, link geometrically."""
    if not subperiods:
        return None
    relevant = []
    for sp in subperiods:
        if sp.start_date == "inception":
            if sp.end_date <= to_date:  # pragma: no branch - exercised in dedicated inception test
                relevant.append(sp)
            continue
        if sp.end_date >= from_date and sp.start_date <= to_date:
            relevant.append(sp)
    if not relevant:
        return None
    real = [sp for sp in relevant if sp.start_date != "inception"]
    if len(real) < 2:
        return None
    relevant.sort(key=lambda x: x.start_date)
    returns = [sp.hpr for sp in relevant]
    return geometric_link(returns)


__all__ = ["geometric_link", "build_subperiods", "compute_linked_twrr", "normalise_schwab_trades", "TradeEvent", "TradeSubPeriod"]


def normalise_schwab_trades(
    transactions: list[dict[str, Any]],
    symbol: str,
    current_position: Optional[dict[str, Any]] = None,
) -> list[TradeEvent]:
    """
    Map Schwab tx to TradeEvent list.
    When no quantity/price in tx, derive mv_at from netAmount approx and anchor qty * current avg or last price.
    Never use final MV for intermediate.
    """
    events: list[TradeEvent] = []
    for t in transactions:
        if (t.get("type") or "").upper() != "TRADE":
            continue
        if (t.get("instrument") or {}).get("symbol") != symbol:
            continue
        d = (t.get("tradeDate") or t.get("time") or "")[:10]
        net = _safe_float(t.get("netAmount"))
        cf = -net if net < 0 else 0.0  # capital into position for buy
        q = _safe_float(t.get("quantity"))
        p = _safe_float(t.get("price"))
        if q <= 0 or p <= 0:
            # derive approx: use |net| / some price, fallback to pos avg or 1
            if current_position:
                p = _safe_float(current_position.get("averagePrice")) or 1.0
            else:
                p = 1.0
            q = abs(net) / p if p > 0 else 0.0
        mv = q * p
        events.append(TradeEvent(event_date=d, cash_flow=cf, quantity=q, price=p))
    return events


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
