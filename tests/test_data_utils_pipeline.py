from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    _apply_filters_in_memory,
    _chunked,
    _sql_literal_list,
    _to_api_filters,
    asof_join_point_in_time,
    build_static_top10_universe,
    fetch_zacks_table,
    load_prices_csv_required,
    load_universe_tickers,
    normalize_ticker_for_prices,
    prepare_fundamentals_with_availability,
    validate_point_in_time_panel,
)


def test_normalize_ticker_for_prices_and_sql_literal_list() -> None:
    assert normalize_ticker_for_prices("BRK.B") == "BRK_B"
    assert normalize_ticker_for_prices("AAPL") == "AAPL"
    assert _sql_literal_list(["A", "O'BRIEN"]) == "'A', 'O''BRIEN'"


def test_to_api_filters_and_apply_filters_in_memory() -> None:
    filters = {
        "date": {"between": ("2020-01-01", "2020-12-31")},
        "ticker": {"in": ["A", "B"]},
        "flag": "Y",
    }
    api = _to_api_filters(filters)
    assert api == {
        "date": {"gte": "2020-01-01", "lte": "2020-12-31"},
        "ticker": ["A", "B"],
        "flag": "Y",
    }

    df = pd.DataFrame(
        {
            "date": ["2020-01-05", "2020-02-01", "2021-01-01"],
            "ticker": ["A", "C", "B"],
            "flag": ["Y", "Y", "N"],
        }
    )
    out = _apply_filters_in_memory(df, filters)
    assert out["ticker"].tolist() == ["A"]


def test_build_static_top10_universe_filters_sp500_and_sorts() -> None:
    mt = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "sp500_member_flag": ["Y", "N", "Y", "Y"],
        }
    )
    mktv = pd.DataFrame(
        {
            "ticker": ["A", "C", "D", "A"],
            "per_end_date": pd.to_datetime(["2012-12-31", "2012-12-31", "2012-12-31", "2012-09-30"]),
            "per_type": ["Q", "Q", "Q", "Q"],
            "mkt_val": [100.0, 300.0, 200.0, 999.0],
        }
    )

    out = build_static_top10_universe(mt, mktv, rank_date="2012-12-31")
    assert out["ticker"].tolist() == ["C", "D", "A"]
    assert (out["rank_date"] == pd.Timestamp("2012-12-31")).all()


def test_prepare_fundamentals_with_availability_merges_and_converts() -> None:
    fr = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "per_end_date": ["2020-03-31", "2020-06-30"],
            "per_type": ["Q", "Q"],
            "tot_debt_tot_equity": ["1.5", "2.0"],
            "ret_equity": ["0.1", "bad"],
            "profit_margin": ["0.2", "0.3"],
            "book_val_per_share": ["10", "11"],
        }
    )
    fc = pd.DataFrame(
        {
            "ticker": ["A"],
            "per_end_date": ["2020-03-31"],
            "per_type": ["Q"],
            "diluted_net_eps": ["2.5"],
        }
    )

    out = prepare_fundamentals_with_availability(fr, fc, lag_days=45)
    assert "feature_available_date" in out.columns
    assert out.loc[0, "feature_available_date"] == pd.Timestamp("2020-05-15")
    assert np.isclose(out.loc[0, "tot_debt_tot_equity"], 1.5)
    assert np.isnan(out.loc[1, "ret_equity"])


def test_asof_join_point_in_time_and_validate() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "adj_close": [10, 11, 12, 20],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A"],
            "feature_available_date": pd.to_datetime(["2020-01-02"]),
            "book_val_per_share": [5.0],
        }
    )

    out = asof_join_point_in_time(prices, fundamentals, on_date_col="date", by_ticker_col="ticker")
    a = out[out["ticker"] == "A"].reset_index(drop=True)
    b = out[out["ticker"] == "B"].reset_index(drop=True)

    assert np.isnan(a.loc[0, "book_val_per_share"])
    assert np.isclose(a.loc[1, "book_val_per_share"], 5.0)
    assert np.isclose(a.loc[2, "book_val_per_share"], 5.0)
    assert np.isnan(b.loc[0, "book_val_per_share"])

    validate_point_in_time_panel(out)


def test_asof_join_point_in_time_empty_prices_returns_empty_with_fund_columns() -> None:
    prices = pd.DataFrame(columns=["ticker", "date", "adj_close"])
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A"],
            "feature_available_date": pd.to_datetime(["2020-01-01"]),
            "diluted_net_eps": [1.2],
        }
    )

    out = asof_join_point_in_time(prices, fundamentals, on_date_col="date", by_ticker_col="ticker")
    assert out.empty
    assert "feature_available_date" in out.columns
    assert "diluted_net_eps" in out.columns


def test_validate_point_in_time_panel_raises_on_lookahead() -> None:
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "feature_available_date": pd.to_datetime(["2020-01-02"]),
        }
    )
    with pytest.raises(AssertionError, match="lookahead leakage"):
        validate_point_in_time_panel(bad)


def test_load_universe_tickers_uses_fallback_and_normalizes(tmp_path: Path) -> None:
    primary = tmp_path / "missing.csv"
    fallback = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": [" aapl ", "msft", "", "MSFT"]}).to_csv(fallback, index=False)

    tickers, source = load_universe_tickers(primary, fallback)
    assert tickers == ["AAPL", "MSFT"]
    assert source == fallback


def test_chunked_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        _chunked([1, 2, 3], chunk_size=0)


def test_fetch_zacks_table_applies_post_filters_for_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_get_table(table_code: str, qopts: dict[str, object], paginate: bool, **kwargs: object) -> pd.DataFrame:
        captured["table_code"] = table_code
        captured["kwargs"] = kwargs
        assert qopts["columns"] == ["ticker", "sp500_member_flag"]
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sp500_member_flag": ["Y", "N", "Y"],
                "extra_col": [1, 2, 3],
            }
        )

    monkeypatch.setattr("src.data_utils.nasdaqdatalink.get_table", _fake_get_table)

    out = fetch_zacks_table(
        table_code="ZACKS/MT",
        columns=["ticker", "sp500_member_flag"],
        filters={"sp500_member_flag": "Y", "ticker": {"in": ["A", "C"]}},
        paginate=True,
    )

    assert captured["table_code"] == "ZACKS/MT"
    assert captured["kwargs"] == {"ticker": ["A", "C"]}
    assert out["ticker"].tolist() == ["A", "C"]
    assert out.columns.tolist() == ["ticker", "sp500_member_flag"]


def test_load_prices_csv_required_filters_on_ticker_and_date(tmp_path: Path) -> None:
    csv_path = tmp_path / "PRICES.csv"
    pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "date": ["2020-01-01", "2020-02-01", "2020-01-15"],
            "adj_close": [10, 11, 20],
            "volume": [100, 110, 200],
        }
    ).to_csv(csv_path, index=False)

    out = load_prices_csv_required(
        csv_path=csv_path,
        tickers=["A"],
        start="2020-01-10",
        end="2020-12-31",
        usecols=["ticker", "date", "adj_close", "volume"],
    )
    assert out["ticker"].tolist() == ["A"]
    assert out["date"].tolist() == [pd.Timestamp("2020-02-01")]
