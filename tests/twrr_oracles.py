"""Frozen oracle table for TWRR subperiods and linking. Used by unit tests and audit script."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schwab_positions_mcp.twrr_calc import TradeEvent, build_subperiods, compute_linked_twrr

@dataclass(frozen=True)
class TwrrOracle:
    name: str
    events: list[TradeEvent]
    initial_quantity: float
    final_mv: float
    as_of: str
    expected_hprs: list[float]  # for non-inception subs (incl terminal if appended)
    expected_linked: float | None

ORACLES: list[TwrrOracle] = [
    # buy-only, consistent, initial=0
    TwrrOracle(
        name="buy_only",
        events=[
            TradeEvent("2026-06-01", 1050.0, 10, 105.0),
            TradeEvent("2026-06-10", 1100.0, 10, 110.0),
        ],
        initial_quantity=0.0,
        final_mv=2420.0,
        as_of="2026-06-20",
        expected_hprs=[0.0, 0.047619, 0.1],  # event1, event2, terminal (post 2200->2420)
        expected_linked=0.15238,
    ),
    # buy + sell , linked ~0.2
    TwrrOracle(
        name="buy_sell",
        events=[
            TradeEvent("2026-06-01", 1000.0, 10, 100.0),
            TradeEvent("2026-06-10", -550.0, 5, 110.0),  # cf negative for sell
        ],
        initial_quantity=0.0,
        final_mv=600.0,
        as_of="2026-06-20",
        expected_hprs=[0.0, 0.1, 0.0909],
        expected_linked=0.2,
    ),
    # sell-only with initial (pre-window holding)
    TwrrOracle(
        name="sell_only_initial",
        events=[
            TradeEvent("2026-06-10", -550.0, 5, 110.0),
        ],
        initial_quantity=10.0,  # had 10, sold 5, final 5
        final_mv=600.0,
        as_of="2026-06-20",
        expected_hprs=[0.0, 0.0909],
        expected_linked=0.0909,
    ),
    # 0-MV terminal
    TwrrOracle(
        name="zero_mv_terminal",
        events=[TradeEvent("2026-06-01", 1000.0, 10, 100.0)],
        initial_quantity=0.0,
        final_mv=0.0,
        as_of="2026-06-01",
        expected_hprs=[0.0, -1.0],  # event + terminal
        expected_linked=-1.0,
    ),
]

def get_oracle(name: str) -> TwrrOracle:
    for o in ORACLES:
        if o.name == name:
            return o
    raise KeyError(name)

def build_from_oracle(o: TwrrOracle) -> list:
    subs = build_subperiods(o.events, "TEST", o.final_mv, o.as_of, initial_quantity=o.initial_quantity)
    return subs

def linked_from_oracle(o: TwrrOracle, from_d: str = "2026-01-01", to_d: str = "2026-12-31") -> float | None:
    subs = build_from_oracle(o)
    return compute_linked_twrr(subs, from_d, to_d)
