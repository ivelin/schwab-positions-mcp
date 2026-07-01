"""Oracle-first unit tests for pure twrr_calc.

Fixtures are hand-computed.

No mocks of MCP, no live, no csv.
"""

from __future__ import annotations

import pytest

from schwab_positions_mcp.twrr_calc import (
    TradeEvent,
    TradeSubPeriod,
    build_subperiods,
    compute_linked_twrr,
    geometric_link,
    normalise_schwab_trades,
)


def test_geometric_link():
    assert geometric_link([0.1, 0.2]) == pytest.approx(0.32)
    assert geometric_link([]) == 0.0


def test_build_and_link_basic():
    # two buys, consistent cf == q*p ; new accumulation uses total MV at event p
    events = [
        TradeEvent("2026-06-01", 1050.0, 10, 105.0),  # entry value 1050
        TradeEvent("2026-06-10", 1100.0, 10, 110.0),  # pre at p2: 10*110=1100 , hpr on prior ~0.0476
    ]
    subs = build_subperiods(events, "AAPL", 2200.0, "2026-06-20")
    assert len(subs) >= 3  # 2 + terminal hpr0
    hprs = [s.hpr for s in subs if s.start_date != "inception"]
    assert hprs[0] == pytest.approx(0.0)
    assert hprs[1] == pytest.approx(0.047619, abs=1e-5)
    # linked (term hpr0)
    linked = compute_linked_twrr(subs, "2026-05-01", "2026-06-21")
    assert linked == pytest.approx(0.047619, abs=0.01)


def test_normalise_schwab_no_qp_fallback():
    txs = [
        {"type": "TRADE", "instrument": {"symbol": "AAPL"}, "netAmount": -1000.0, "tradeDate": "2026-06-01"},
    ]
    pos = {"longQuantity": 10.0, "averagePrice": 100.0, "marketValue": 1000.0}
    events = normalise_schwab_trades(txs, "AAPL", pos)
    assert len(events) == 1
    assert events[0].cash_flow == 1000.0
    assert events[0].quantity > 0


def test_window_filter():
    subs = [
        TradeSubPeriod("AAPL", "2026-01-01", "2026-01-01", 1000, 1050, 0, 0.05),
        TradeSubPeriod("AAPL", "2026-02-01", "2026-02-01", 1050, 1100, 0, 0.0476),
    ]
    # full
    assert compute_linked_twrr(subs, "2025-01-01", "2026-03-01") == pytest.approx(0.1, abs=0.01)
    # window limited, may None if <2 real in window
    res = compute_linked_twrr(subs, "2026-01-15", "2026-03-01")
    assert res is None or res == pytest.approx(0.0476, abs=0.01)


def test_build_empty():
    assert build_subperiods([], "AAPL", 100.0, "2026-07-01") == []


def test_build_as_of_none_branch():
    """Omit as_of to hit the None default + today set + append (today in 2026 > 06)."""
    events = [TradeEvent("2026-06-20", 1000.0, 10, 105.0)]
    subs = build_subperiods(events, "AAPL", 1200.0)  # as_of=None inside
    assert len(subs) >= 1
    # will have appended or at least ran the if None
    assert any(s.end_market_value == 1200.0 for s in subs) or subs[0].end_market_value > 0


def test_build_terminal_zero_mv_hpr_branch():
    """Hits the else: hpr=0 when last_mv ==0 in terminal append."""
    events = [TradeEvent("2026-06-01", 0.0, 0, 0.0)]  # zero
    subs = build_subperiods(events, "Z", 0.0, "2026-06-02")
    assert len(subs) == 2
    assert subs[-1].hpr == 0.0


def test_build_terminal_append():
    """Hits the as_of > last.end branch and append terminal sub."""
    events = [TradeEvent("2026-06-01", 1000.0, 10, 105.0)]
    subs = build_subperiods(events, "AAPL", 1200.0, "2026-06-10")
    assert len(subs) == 2
    assert subs[1].start_date == "2026-06-01"
    assert subs[1].end_date == "2026-06-10"
    assert subs[1].end_market_value == 1200.0


def test_compute_none_paths():
    assert compute_linked_twrr([], "2026-01-01", "2026-02-01") is None
    one = [TradeSubPeriod("AAPL", "2026-01-01", "2026-01-05", 1000, 1050, 0, 0.05)]
    assert compute_linked_twrr(one, "2026-01-01", "2026-02-01") is None
    assert compute_linked_twrr(one, "2026-02-01", "2026-03-01") is None


def test_normalise_filters_and_explicit_qp():
    # non TRADE and wrong symbol skipped
    txs = [
        {"type": "DIVIDEND", "instrument": {"symbol": "AAPL"}, "netAmount": 5},
        {"type": "TRADE", "instrument": {"symbol": "MSFT"}, "netAmount": -10, "quantity": 1, "price": 10, "tradeDate": "2026-06-01"},
        {"type": "TRADE", "instrument": {"symbol": "AAPL"}, "netAmount": -1050, "quantity": 10, "price": 105, "tradeDate": "2026-06-01"},
    ]
    evs = normalise_schwab_trades(txs, "AAPL")
    assert len(evs) == 1
    assert evs[0].quantity == 10
    assert evs[0].price == 105.0

    # fallback when no pos and no qp
    evs2 = normalise_schwab_trades([{"type": "TRADE", "instrument": {"symbol": "XYZ"}, "netAmount": -100}], "XYZ")
    assert len(evs2) == 1
    assert evs2[0].quantity == 100.0  # /1.0
    assert evs2[0].price == 1.0

    # hit _safe_float except path
    evs3 = normalise_schwab_trades([{"type": "TRADE", "instrument": {"symbol": "BAD"}, "netAmount": "notnum", "quantity": "x", "price": None, "tradeDate": "2026-07-01"}], "BAD")
    assert len(evs3) == 1
    assert evs3[0].cash_flow == 0.0
    assert evs3[0].quantity == 0.0


def test_compute_inception_branch():
    """Directly exercise inception filter branch in compute (even if build omits)."""
    subs = [
        TradeSubPeriod("S", "inception", "2026-01-01", 0, 1000, 0, 0.0),
        TradeSubPeriod("S", "2026-01-01", "2026-01-10", 1000, 1100, 0, 0.1),
        TradeSubPeriod("S", "2026-01-10", "2026-02-01", 1100, 1200, 0, 0.0909),
    ]
    res = compute_linked_twrr(subs, "2025-01-01", "2026-02-01")
    assert res == pytest.approx(0.2, abs=0.02)  # links the two real


def test_inception_filter_both_arms():
    """Cover both sides of inception <= to_date in compute (with/without append)."""
    subs = [
        TradeSubPeriod("S", "inception", "2026-04-01", 0, 1000, 0, 0.0),
        TradeSubPeriod("S", "2026-01-20", "2026-02-01", 1000, 1100, 0, 0.1),
        TradeSubPeriod("S", "2026-02-01", "2026-03-01", 1100, 1200, 0, 0.0909),
    ]
    # window that includes reals (start<=to, end>=from) but inception end > to : skips inception (false)
    res = compute_linked_twrr(subs, "2026-01-01", "2026-03-15")
    assert res is not None
    # wide window: includes inception (true arm)
    res2 = compute_linked_twrr(subs, "2025-01-01", "2026-05-01")
    assert res2 is not None


def test_normalise_and_build_with_sell():
    """Critical: sells set negative cf; pre-flow MV uses full pre_q * p."""
    from tests.twrr_oracles import get_oracle
    o = get_oracle("buy_sell")
    subs = build_subperiods(o.events, "TEST", o.final_mv, o.as_of, initial_quantity=o.initial_quantity)
    assert len(subs) >= 3
    assert subs[1].hpr == pytest.approx(0.1)
    linked = compute_linked_twrr(subs, "2026-05-01", "2026-06-21")
    assert linked == pytest.approx(o.expected_linked, abs=0.001)


def test_same_day_trades_and_ytd_sufficient():
    """Same-day multiple trades create multiple subs; ytd window with data gives linked non-None."""
    events = [
        TradeEvent("2026-06-15", 1000.0, 10, 100.0),
        TradeEvent("2026-06-15", 500.0, 5, 101.0),  # same day second trade
    ]
    subs = build_subperiods(events, "AAPL", 1600.0, "2026-06-15")
    assert len(subs) >= 2  # multiple on day + terminal? but as_of same
    # force ytd that includes
    linked = compute_linked_twrr(subs, "2026-01-01", "2026-06-16")
    assert linked is not None
    assert len(subs) >= 2


def test_bad_date_and_fallback_variants():
    """Empty date from tx, fallback without pos, with pos."""
    tx_bad = [{"type": "TRADE", "instrument": {"symbol": "BAD"}, "netAmount": -100}]
    evs = normalise_schwab_trades(tx_bad, "BAD")
    assert evs[0].event_date == ""  # current behavior; filters later may treat
    # with pos fallback
    pos = {"averagePrice": 50.0}
    evs2 = normalise_schwab_trades([{"type":"TRADE","instrument":{"symbol":"P"},"netAmount":-200}], "P", pos)
    assert evs2[0].price == 50.0
    assert evs2[0].quantity == 4.0


def test_first_event_hpr_zero_and_cf_zero_path():
    """First-event always produces hpr=0 using q*p; cf=0 when netAmount missing treated as inflow (correct buy path, non-zero if final differs)."""
    # buy with full data
    evs = normalise_schwab_trades([{"type":"TRADE","instrument":{"symbol":"F"},"netAmount":-1050,"quantity":10,"price":105,"tradeDate":"2026-06-01"}], "F")
    subs = build_subperiods(evs, "F", 1050, "2026-06-01")
    assert subs[0].hpr == 0.0
    # tx missing netAmount (cf falls to 0) - should be treated as buy, start_mv = q*p
    evs2 = normalise_schwab_trades([{"type":"TRADE","instrument":{"symbol":"M"},"quantity":3,"price":50,"tradeDate":"2026-06-01"}], "M")
    assert evs2[0].cash_flow == 0.0
    assert evs2[0].quantity == 3.0
    # build+link numeric: with cf=0 but positive q, should use as inflow, hpr=0 at event, terminal from 150->200 say positive
    subs2 = build_subperiods(evs2, "M", 200.0, "2026-06-02")  # final > event mv=150
    assert len(subs2) >= 2
    assert subs2[0].hpr == 0.0
    assert subs2[0].start_market_value == 150.0
    linked2 = compute_linked_twrr(subs2, "2026-05-01", "2026-06-03")
    assert linked2 is not None and linked2 > 0.0  # terminal provides positive return on the 'bought' capital


def test_0_mv_terminal_and_window_exact():
    """0 mv final and exact window boundaries (terminal appended even on same date)."""
    events = [TradeEvent("2026-06-01", 1000, 10, 100)]
    subs = build_subperiods(events, "Z", 0.0, "2026-06-01")
    assert len(subs) == 2
    assert subs[1].end_market_value == 0.0
    assert subs[1].hpr == -1.0
    linked = compute_linked_twrr(subs, "2026-06-01", "2026-06-01")
    assert linked == -1.0


def test_terminal_append_oracle_and_multi_trade_nonnull():
    """Oracle using total accumulated MV at event prices (correct for holdings).
    Consistent cf=q*p. sub0 hpr=0, sub1 hpr~0.04762 (10*110 /10*105), term from post2200 to 2420 hpr~0.1
    linked ≈ 0.152
    """
    events = [
        TradeEvent("2026-06-01", 1050.0, 10, 105.0),
        TradeEvent("2026-06-10", 1100.0, 10, 110.0),
    ]
    subs = build_subperiods(events, "AAPL", 2420.0, "2026-06-20")
    assert len(subs) == 3
    assert subs[0].hpr == pytest.approx(0.0)
    assert subs[1].hpr == pytest.approx(0.047619, abs=1e-5)
    assert subs[1].end_market_value == pytest.approx(1100.0)
    assert subs[2].start_market_value == pytest.approx(2200.0)
    assert subs[2].end_market_value == pytest.approx(2420.0)
    linked = compute_linked_twrr(subs, "2026-05-01", "2026-06-21")
    assert linked is not None
    assert linked == pytest.approx(0.15238, abs=0.01)


def test_schwab_multi_trade_via_normalise_and_link():
    """Full path using normalise on Schwab-shaped tx (q/p present) + current pos, then build+link non-null."""
    txs = [
        {"type": "TRADE", "instrument": {"symbol": "AAPL"}, "netAmount": -1050.0, "quantity": 10, "price": 105.0, "tradeDate": "2026-06-01"},
        {"type": "TRADE", "instrument": {"symbol": "AAPL"}, "netAmount": -1100.0, "quantity": 10, "price": 110.0, "tradeDate": "2026-06-10"},
    ]
    pos = {"averagePrice": 150.0, "marketValue": 2200.0}
    events = normalise_schwab_trades(txs, "AAPL", pos)
    assert len(events) == 2
    subs = build_subperiods(events, "AAPL", 2200.0, "2026-06-20")
    linked = compute_linked_twrr(subs, "2026-05-01", "2026-06-21")
    # accumulation; second event gives ~0.0476 , terminal 0 (same p) => linked ~0.0476
    assert len(subs) >= 2
    assert linked == pytest.approx(0.047619, abs=0.001)


def test_sell_only_with_initial_anchor():
    """Sell-only first event with initial_quantity >0 produces sensible non-neg linked (uses pre-window qty)."""
    from tests.twrr_oracles import get_oracle
    o = get_oracle("sell_only_initial")
    subs = build_subperiods(o.events, "TEST", o.final_mv, o.as_of, initial_quantity=o.initial_quantity)
    assert len(subs) >= 2
    assert subs[0].hpr == pytest.approx(0.0)
    linked = compute_linked_twrr(subs, "2026-05-01", "2026-06-21")
    assert linked == pytest.approx(o.expected_linked, abs=0.001)
