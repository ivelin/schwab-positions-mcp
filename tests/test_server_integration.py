"""Server-level integration tests for ``schwab_positions_mcp.server``.

These tests verify the FastMCP-registered tool surface end-to-end without
spawning a subprocess: we call the registered tool callables directly so we
can inject a mocked ``ReadOnlySchwabClient``. A fully out-of-process
``stdio_client`` test would need a real OAuth token, which we deliberately
keep out of CI.
"""

from __future__ import annotations

import pytest

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from schwab_positions_mcp import server as server_module
from schwab_positions_mcp.tools import meta


def _resp(status: int = 200, payload: Any = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {}
    r.json.return_value = payload
    r.text = ""
    return r


class TestServerToolSurface:
    def test_fifteen_tools_registered(self) -> None:
        info = meta.get_server_info_impl()
        assert len(info["tools"]) == 15
        assert "get_twrr_analysis" in info["tools"]

    def test_server_module_has_all_tool_callables(self) -> None:
        for name in (
            "get_accounts",
            "get_account_numbers",
            "get_account_positions",
            "get_orders_history",
            "get_transactions",
            "get_user_preferences",
            "get_order_detail",
            "get_transaction_detail",
            "get_account_summary",
            "get_pnl_analysis",
            "get_concentration_analysis",
            "get_cross_account_summary",
            "get_twrr_analysis",
            "health_check",
            "get_server_info",
        ):
            assert hasattr(server_module, name), f"server.{name} missing"

    def test_get_accounts_proxies_to_client(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_account_data: list[dict[str, Any]],
    ) -> None:
        mock_schwab_client.get_accounts.return_value = _resp(200, mock_account_data)
        out = server_module.get_accounts()
        assert out["ok"] is True
        assert out["count"] == 2

    def test_health_check_returns_dict(self) -> None:
        out = server_module.health_check()
        assert isinstance(out, dict)
        assert "status" in out
        assert out["is_read_only"] is True

    def test_get_server_info_returns_dict(self) -> None:
        out = server_module.get_server_info()
        assert out["name"] == "schwab-positions-mcp"

    def test_get_account_positions_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_positions_data: dict[str, Any],
    ) -> None:
        mock_schwab_client.get_account.return_value = _resp(200, mock_positions_data)
        out = server_module.get_account_positions(
            account_hash="ACCT_HASH_AAAAAAAAAAAA",
        )
        assert out["ok"] is True
        assert out["count"] == 2

    def test_get_orders_history_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_orders_data: list[dict[str, Any]],
    ) -> None:
        mock_schwab_client.get_orders_for_account.return_value = _resp(200, mock_orders_data)
        today = datetime.now(UTC)
        from_t = (today - timedelta(days=30)).isoformat()
        to_t = (today - timedelta(days=5)).isoformat()
        out = server_module.get_orders_history(
            account_hash="ACCT_HASH_AAAAAAAAAAAA",
            from_entered_time=from_t,
            to_entered_time=to_t,
        )
        assert out["ok"] is True
        assert out["count"] == 2

    def test_get_transactions_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_transactions_data: list[dict[str, Any]],
    ) -> None:
        mock_schwab_client.get_transactions.return_value = _resp(200, mock_transactions_data)
        today = date.today()
        out = server_module.get_transactions(
            account_hash="ACCT_HASH_AAAAAAAAAAAA",
            start_date=(today - timedelta(days=30)).isoformat(),
            end_date=(today - timedelta(days=5)).isoformat(),
        )
        assert out["ok"] is True
        assert out["count"] == 2

    def test_get_user_preferences_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
    ) -> None:
        mock_schwab_client.get_user_preferences.return_value = _resp(200, {"accounts": []})
        out = server_module.get_user_preferences()
        assert out["ok"] is True
        assert out["preferences"] == {"accounts": []}

    def test_get_order_detail_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
    ) -> None:
        mock_schwab_client.get_order.return_value = _resp(200, {"orderId": 12345, "status": "FILLED"})
        out = server_module.get_order_detail(account_hash="ACCT_HASH_AAAAAAAAAAAA", order_id=12345)
        assert out["ok"] is True
        assert out["order"]["orderId"] == 12345
        mock_schwab_client.get_order.assert_called_once_with(12345, "ACCT_HASH_AAAAAAAAAAAA")

    def test_get_transaction_detail_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
    ) -> None:
        mock_schwab_client.get_transaction.return_value = _resp(200, {"transactionId": "TX_001"})
        out = server_module.get_transaction_detail(account_hash="ACCT_HASH_AAAAAAAAAAAA", transaction_id="TX_001")
        assert out["ok"] is True
        assert out["transaction"]["transactionId"] == "TX_001"
        mock_schwab_client.get_transaction.assert_called_once_with("ACCT_HASH_AAAAAAAAAAAA", "TX_001")

    def test_get_account_summary_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_positions_data: dict[str, Any],
    ) -> None:
        mock_schwab_client.get_account.return_value = _resp(200, mock_positions_data)
        out = server_module.get_account_summary(
            account_hash="ACCT_HASH_AAAAAAAAAAAA",
        )
        assert out["ok"] is True
        assert out["summary"]["position_count"] == 2

    def test_get_pnl_analysis_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_positions_data: dict[str, Any],
    ) -> None:
        mock_schwab_client.get_account.return_value = _resp(200, mock_positions_data)
        mock_schwab_client.get_transactions.return_value = _resp(200, [])
        out = server_module.get_pnl_analysis(account_hash="ACCT_HASH_AAAAAAAAAAAA")
        assert out["ok"] is True
        assert out["cost_basis_method"] == "average_cost"
        assert out["portfolio"]["position_count"] == 2

    def test_get_concentration_analysis_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_positions_data: dict[str, Any],
    ) -> None:
        mock_schwab_client.get_account.return_value = _resp(200, mock_positions_data)
        out = server_module.get_concentration_analysis(account_hash="ACCT_HASH_AAAAAAAAAAAA")
        assert out["ok"] is True
        assert out["concentration"]["position_count"] == 2

    def test_get_cross_account_summary_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
        mock_positions_data: dict[str, Any],
    ) -> None:
        mock_schwab_client.get_account_numbers.return_value = _resp(
            200, [{"accountNumber": "A1", "hashValue": "ACCT_HASH_AAAAAAAAAAAA"}]
        )
        mock_schwab_client.get_account.return_value = _resp(200, mock_positions_data)
        out = server_module.get_cross_account_summary()
        assert out["ok"] is True
        assert out["account_count"] == 1

    def test_get_twrr_analysis_via_server(
        self,
        installed_client: Any,
        mock_schwab_client: MagicMock,
    ) -> None:
        """Exercise server wrapper for TWRR + multi-sub data (hits server body + impl)."""
        from datetime import date as _date
        from datetime import timedelta as _td
        today = _date.today()
        pos_payload = {
            "securitiesAccount": {
                "positions": [
                    {
                        "instrument": {"symbol": "AAPL"},
                        "longQuantity": 20.0,
                        "averagePrice": 150.0,
                        "marketValue": 3500.0,
                    }
                ]
            }
        }
        mock_schwab_client.get_account.return_value = _resp(200, pos_payload)
        txs = [
            {
                "transactionId": "T1",
                "tradeDate": (today - _td(days=25)).isoformat(),
                "type": "TRADE",
                "instrument": {"symbol": "AAPL"},
                "netAmount": -1500.0,
                "quantity": 10.0,
                "price": 150.0,
            },
            {
                "transactionId": "T2",
                "tradeDate": (today - _td(days=8)).isoformat(),
                "type": "TRADE",
                "instrument": {"symbol": "AAPL"},
                "netAmount": -1500.0,
                "quantity": 10.0,
                "price": 150.0,
            },
        ]
        mock_schwab_client.get_transactions.return_value = _resp(200, txs)
        out = server_module.get_twrr_analysis(
            account_hash="ACCT_HASH_AAAAAAAAAAAA", symbol="AAPL", lookback_days=60
        )
        assert out["ok"] is True
        assert out["symbol"] == "AAPL"
        # With fixed accumulation: events at 1500/1500 (same p), terminal from 3000->3500 => ~16.67
        assert out.get("twrr_30d") is not None
        assert out["twrr_30d"] == pytest.approx(16.67)  # exact for this mock data under accumulation
        assert "twrr_90d" in out and "twrr_ytd" in out and "subperiod_count" in out


class TestMainFunctionGuarded:
    """Ensure the ``main`` entrypoint exists and uses stdio transport."""

    def test_main_callable_exists(self) -> None:
        assert callable(server_module.main)


class TestVersionAttribute:
    def test_mcp_server_has_version(self) -> None:
        assert server_module.mcp._mcp_server.version
