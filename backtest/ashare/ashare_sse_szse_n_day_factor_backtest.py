"""SSE/SZSE A-share N-day factor portfolio backtest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Any

import pandas as pd


FACTOR_REQUIRED_COLUMNS = {
    "ts_code",
    "trade_date",
    "signal_score",
    "strategy_id",
}

BACKTEST_REQUIRED_COLUMNS = {
    "ts_code",
    "trade_date",
    "open",
    "close",
    "adj_factor",
    "can_buy_open",
    "can_sell_open",
    "is_suspended",
}

SSE_SZSE_SUFFIXES = (".SH", ".SZ")


@dataclass(frozen=True)
class NDayFactorBacktestConfig:
    """Configuration for an SSE/SZSE N-day factor portfolio backtest."""

    strategy_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    initial_cash: float = 1_000_000.0
    rebalance_interval: int = 5
    max_holdings: int = 50
    candidate_pool_size: int | None = None
    weight_method: str = "equal"
    custom_rank_weights: tuple[float, ...] | None = None
    lot_size: int = 100
    buy_cost_rate: float = 0.0003
    sell_cost_rate: float = 0.0013
    score_ascending: bool = False
    force_liquidate_on_last_day: bool = False
    annual_trading_days: int = 252


@dataclass
class Holding:
    """Mutable holding state."""

    shares: float
    last_buy_date: str
    last_price: float
    last_adj_factor: float


def _normalize_date(date_text: str | None) -> str | None:
    if date_text is None:
        return None
    digits = "".join(ch for ch in str(date_text) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Invalid date: {date_text!r}. Expected YYYYMMDD.")
    return digits


def _normalize_ts_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _validate_columns(frame: pd.DataFrame, required_columns: set[str], frame_name: str) -> None:
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"`{frame_name}` missing required columns: {missing_columns}")


def _validate_config(config: NDayFactorBacktestConfig) -> tuple[int, tuple[float, ...]]:
    if config.initial_cash <= 0:
        raise ValueError("`initial_cash` must be positive.")
    if config.rebalance_interval <= 0:
        raise ValueError("`rebalance_interval` must be positive.")
    if config.max_holdings <= 0:
        raise ValueError("`max_holdings` must be positive.")
    if config.lot_size <= 0:
        raise ValueError("`lot_size` must be positive.")
    if config.buy_cost_rate < 0 or config.sell_cost_rate < 0:
        raise ValueError("cost rates must be non-negative.")

    candidate_pool_size = config.candidate_pool_size
    if candidate_pool_size is None:
        candidate_pool_size = config.max_holdings * 2
    if candidate_pool_size < config.max_holdings:
        raise ValueError("`candidate_pool_size` must be greater than or equal to `max_holdings`.")

    if config.weight_method == "equal":
        if config.custom_rank_weights is not None:
            raise ValueError("`custom_rank_weights` must be None when `weight_method='equal'`.")
        weights = tuple([1.0 / config.max_holdings] * config.max_holdings)
    elif config.weight_method == "custom_rank_weights":
        if config.custom_rank_weights is None:
            raise ValueError("`custom_rank_weights` is required when `weight_method='custom_rank_weights'`.")
        weights = tuple(float(weight) for weight in config.custom_rank_weights)
        if len(weights) != config.max_holdings:
            raise ValueError("`custom_rank_weights` length must equal `max_holdings`.")
    else:
        raise ValueError("`weight_method` must be 'equal' or 'custom_rank_weights'.")

    if any(weight < 0 for weight in weights):
        raise ValueError("rank weights must be non-negative.")
    if sum(weights) > 1.0 + 1e-12:
        raise ValueError("rank weights must sum to 1.0 or less.")

    return candidate_pool_size, weights


def _prepare_factor_score(
    factor_score: pd.DataFrame,
    config: NDayFactorBacktestConfig,
) -> pd.DataFrame:
    _validate_columns(factor_score, FACTOR_REQUIRED_COLUMNS, "factor_score")
    prepared = factor_score.copy()
    prepared["ts_code"] = _normalize_ts_code(prepared["ts_code"])
    prepared["trade_date"] = prepared["trade_date"].astype(str)
    prepared = prepared[prepared["ts_code"].str.endswith(SSE_SZSE_SUFFIXES, na=False)].copy()
    if config.strategy_id is not None:
        prepared = prepared[prepared["strategy_id"].astype(str).eq(str(config.strategy_id))].copy()
    else:
        strategy_count = prepared["strategy_id"].nunique(dropna=True)
        if strategy_count > 1:
            raise ValueError(
                "`factor_score` contains multiple strategy_id values. "
                "Please pass `strategy_id` to backtest one strategy at a time."
            )
    prepared["signal_score"] = pd.to_numeric(prepared["signal_score"], errors="coerce")
    prepared = prepared.dropna(subset=["ts_code", "trade_date", "signal_score"])
    if prepared.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("`factor_score` contains duplicate (`ts_code`, `trade_date`) rows.")
    return prepared.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _prepare_backtest_base(
    backtest_base: pd.DataFrame,
    config: NDayFactorBacktestConfig,
) -> pd.DataFrame:
    _validate_columns(backtest_base, BACKTEST_REQUIRED_COLUMNS, "backtest_base")
    prepared = backtest_base.copy()
    prepared["ts_code"] = _normalize_ts_code(prepared["ts_code"])
    prepared["trade_date"] = prepared["trade_date"].astype(str)
    prepared = prepared[prepared["ts_code"].str.endswith(SSE_SZSE_SUFFIXES, na=False)].copy()
    if config.start_date is not None:
        prepared = prepared[prepared["trade_date"] >= config.start_date].copy()
    if config.end_date is not None:
        prepared = prepared[prepared["trade_date"] <= config.end_date].copy()
    if prepared.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("`backtest_base` contains duplicate (`ts_code`, `trade_date`) rows.")
    for column in ("open", "close", "adj_factor"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in ("can_buy_open", "can_sell_open", "is_suspended"):
        prepared[column] = prepared[column].fillna(False).astype(bool)
    return prepared.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _frame_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        trade_date: group.set_index("ts_code", drop=False)
        for trade_date, group in frame.groupby("trade_date", sort=True)
    }


def _get_row(date_frame: pd.DataFrame | None, ts_code: str) -> pd.Series | None:
    if date_frame is None or ts_code not in date_frame.index:
        return None
    row = date_frame.loc[ts_code]
    if isinstance(row, pd.DataFrame):
        return row.iloc[-1]
    return row


def _valid_price(value: Any) -> bool:
    try:
        return pd.notna(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _valid_adj_factor(value: Any) -> bool:
    try:
        return pd.notna(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _row_adj_factor(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    value = row.get("adj_factor")
    if _valid_adj_factor(value):
        return float(value)
    return None


def _adjust_holding_for_corporate_action(row: pd.Series | None, holding: Holding) -> None:
    current_adj_factor = _row_adj_factor(row)
    if current_adj_factor is None or not _valid_adj_factor(holding.last_adj_factor):
        return
    ratio = current_adj_factor / holding.last_adj_factor
    if ratio <= 0:
        return
    if abs(ratio - 1.0) > 1e-12:
        holding.shares *= ratio
        holding.last_adj_factor = current_adj_factor


def _open_or_last_price(row: pd.Series | None, holding: Holding) -> float:
    if row is not None and _valid_price(row.get("open")):
        return float(row["open"])
    return holding.last_price


def _close_or_last_price(row: pd.Series | None, holding: Holding) -> float:
    if row is not None and _valid_price(row.get("close")):
        return float(row["close"])
    return holding.last_price


def _can_buy(row: pd.Series | None) -> tuple[bool, str]:
    if row is None:
        return False, "missing_trade_row"
    if not _valid_price(row.get("open")):
        return False, "missing_open"
    if bool(row.get("is_suspended", False)):
        return False, "suspended"
    if not bool(row.get("can_buy_open", False)):
        return False, "cannot_buy_open"
    return True, "ok"


def _can_sell(row: pd.Series | None, trade_date: str, holding: Holding) -> tuple[bool, str]:
    if trade_date <= holding.last_buy_date:
        return False, "t_plus_one_blocked"
    if row is None:
        return False, "missing_trade_row"
    if not _valid_price(row.get("open")):
        return False, "missing_open"
    if bool(row.get("is_suspended", False)):
        return False, "suspended"
    if not bool(row.get("can_sell_open", False)):
        return False, "cannot_sell_open"
    return True, "ok"


def _build_rebalance_pairs(trade_dates: list[str], rebalance_interval: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    signal_dates = trade_dates[::rebalance_interval]
    next_date = {date: trade_dates[idx + 1] for idx, date in enumerate(trade_dates[:-1])}
    for signal_date in signal_dates:
        execution_date = next_date.get(signal_date)
        if execution_date is not None:
            pairs.append((signal_date, execution_date))
    return pairs


def _build_target_portfolio(
    *,
    signal_date: str,
    execution_date: str,
    factor_by_date: dict[str, pd.DataFrame],
    base_by_date: dict[str, pd.DataFrame],
    holdings: dict[str, Holding],
    total_equity: float,
    candidate_pool_size: int,
    max_holdings: int,
    rank_weights: tuple[float, ...],
    lot_size: int,
    buy_cost_rate: float,
    score_ascending: bool,
) -> tuple[dict[str, float], list[str], list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    signal_frame = factor_by_date.get(signal_date)
    if signal_frame is None or signal_frame.empty:
        return {}, [], [
            {
                "trade_date": execution_date,
                "signal_date": signal_date,
                "ts_code": None,
                "action": "build_target",
                "status": "skipped",
                "reason": "missing_signal_date",
            }
        ]

    candidates = (
        signal_frame.sort_values("signal_score", ascending=score_ascending, kind="stable")
        .head(candidate_pool_size)
        .reset_index(drop=True)
    )
    execution_frame = base_by_date.get(execution_date)
    target_order: list[str] = []
    target_weights: dict[str, float] = {}

    for _, row in candidates.iterrows():
        ts_code = str(row["ts_code"])
        if ts_code in target_weights:
            continue
        is_held = ts_code in holdings
        trade_row = _get_row(execution_frame, ts_code)
        slot_index = len(target_order)
        target_weight = rank_weights[slot_index]
        if not is_held:
            buy_allowed, reason = _can_buy(trade_row)
            if not buy_allowed:
                logs.append(
                    {
                        "trade_date": execution_date,
                        "signal_date": signal_date,
                        "ts_code": ts_code,
                        "action": "select_target",
                        "status": "skipped",
                        "reason": reason,
                        "signal_score": row["signal_score"],
                    }
                )
                continue
            open_price = float(trade_row["open"])
            minimum_cash_required = lot_size * open_price * (1.0 + buy_cost_rate)
            target_value = total_equity * target_weight
            if target_value < minimum_cash_required:
                logs.append(
                    {
                        "trade_date": execution_date,
                        "signal_date": signal_date,
                        "ts_code": ts_code,
                        "action": "select_target",
                        "status": "skipped",
                        "reason": "target_value_below_one_lot",
                        "signal_score": row["signal_score"],
                        "target_weight": target_weight,
                        "target_value": target_value,
                        "minimum_cash_required": minimum_cash_required,
                    }
                )
                continue

        target_order.append(ts_code)
        target_weights[ts_code] = target_weight
        logs.append(
            {
                "trade_date": execution_date,
                "signal_date": signal_date,
                "ts_code": ts_code,
                "action": "select_target",
                "status": "selected",
                "reason": "held" if is_held else "buyable",
                "signal_score": row["signal_score"],
                "target_weight": target_weight,
            }
        )
        if len(target_order) >= max_holdings:
            break

    return target_weights, target_order, logs


def _portfolio_value_at_open(
    holdings: dict[str, Holding],
    date_frame: pd.DataFrame | None,
    cash: float,
) -> tuple[float, dict[str, float], dict[str, float]]:
    current_values: dict[str, float] = {}
    open_prices: dict[str, float] = {}
    stock_value = 0.0
    for ts_code, holding in holdings.items():
        row = _get_row(date_frame, ts_code)
        _adjust_holding_for_corporate_action(row, holding)
        price = _open_or_last_price(row, holding)
        value = holding.shares * price
        current_values[ts_code] = value
        open_prices[ts_code] = price
        stock_value += value
    return cash + stock_value, current_values, open_prices


def _execute_sell(
    *,
    trade_date: str,
    signal_date: str,
    ts_code: str,
    shares: float,
    price: float,
    cash: float,
    holding: Holding,
    sell_cost_rate: float,
    trades: list[dict[str, Any]],
    reason: str,
) -> float:
    amount = shares * price
    cost = amount * sell_cost_rate
    cash += amount - cost
    holding.shares -= shares
    trades.append(
        {
            "trade_date": trade_date,
            "signal_date": signal_date,
            "ts_code": ts_code,
            "side": "sell",
            "price": price,
            "shares": shares,
            "amount": amount,
            "cost": cost,
            "cash_after": cash,
            "reason": reason,
        }
    )
    return cash


def _execute_buy(
    *,
    trade_date: str,
    signal_date: str,
    ts_code: str,
    shares: int,
    price: float,
    cash: float,
    holdings: dict[str, Holding],
    buy_cost_rate: float,
    trades: list[dict[str, Any]],
    reason: str,
    adj_factor: float,
) -> float:
    amount = shares * price
    cost = amount * buy_cost_rate
    cash -= amount + cost
    holding = holdings.get(ts_code)
    if holding is None:
        holdings[ts_code] = Holding(
            shares=float(shares),
            last_buy_date=trade_date,
            last_price=price,
            last_adj_factor=adj_factor,
        )
    else:
        holding.shares += shares
        holding.last_buy_date = trade_date
        holding.last_price = price
        holding.last_adj_factor = adj_factor
    trades.append(
        {
            "trade_date": trade_date,
            "signal_date": signal_date,
            "ts_code": ts_code,
            "side": "buy",
            "price": price,
            "shares": shares,
            "amount": amount,
            "cost": cost,
            "cash_after": cash,
            "reason": reason,
        }
    )
    return cash


def _rebalance(
    *,
    trade_date: str,
    signal_date: str,
    date_frame: pd.DataFrame | None,
    holdings: dict[str, Holding],
    cash: float,
    target_weights: dict[str, float],
    target_order: list[str],
    config: NDayFactorBacktestConfig,
    trades: list[dict[str, Any]],
    logs: list[dict[str, Any]],
) -> float:
    total_equity, current_values, open_prices = _portfolio_value_at_open(holdings, date_frame, cash)

    for ts_code in list(holdings.keys()):
        holding = holdings[ts_code]
        target_value = total_equity * target_weights.get(ts_code, 0.0)
        current_value = current_values.get(ts_code, 0.0)
        if current_value <= target_value:
            continue

        row = _get_row(date_frame, ts_code)
        can_sell, block_reason = _can_sell(row, trade_date, holding)
        if not can_sell:
            logs.append(
                {
                    "trade_date": trade_date,
                    "signal_date": signal_date,
                    "ts_code": ts_code,
                    "action": "sell",
                    "status": "blocked",
                    "reason": block_reason,
                    "target_weight": target_weights.get(ts_code, 0.0),
                    "current_value": current_value,
                    "target_value": target_value,
                }
            )
            continue

        price = float(row["open"])
        if ts_code not in target_weights:
            sell_shares = holding.shares
            reason = "sell_not_in_target"
        else:
            sell_value = current_value - target_value
            sell_shares = ceil(sell_value / price / config.lot_size) * config.lot_size
            sell_shares = min(sell_shares, holding.shares)
            reason = "sell_overweight"
        if sell_shares <= 0:
            continue
        cash = _execute_sell(
            trade_date=trade_date,
            signal_date=signal_date,
            ts_code=ts_code,
            shares=sell_shares,
            price=price,
            cash=cash,
            holding=holding,
            sell_cost_rate=config.sell_cost_rate,
            trades=trades,
            reason=reason,
        )
        logs.append(
            {
                "trade_date": trade_date,
                "signal_date": signal_date,
                "ts_code": ts_code,
                "action": "sell",
                "status": "filled",
                "reason": reason,
                "shares": sell_shares,
                "target_weight": target_weights.get(ts_code, 0.0),
                "current_value": current_value,
                "target_value": target_value,
            }
        )
        if holding.shares <= 0:
            del holdings[ts_code]

    total_equity, current_values, open_prices = _portfolio_value_at_open(holdings, date_frame, cash)
    for ts_code in target_order:
        row = _get_row(date_frame, ts_code)
        can_buy, block_reason = _can_buy(row)
        target_value = total_equity * target_weights[ts_code]
        current_value = current_values.get(ts_code, 0.0)
        if current_value >= target_value:
            continue
        if not can_buy:
            logs.append(
                {
                    "trade_date": trade_date,
                    "signal_date": signal_date,
                    "ts_code": ts_code,
                    "action": "buy",
                    "status": "blocked",
                    "reason": block_reason,
                    "target_weight": target_weights[ts_code],
                    "current_value": current_value,
                    "target_value": target_value,
                }
            )
            continue

        price = float(row["open"])
        buy_value = target_value - current_value
        target_limited_shares = floor(
            buy_value / (price * (1.0 + config.buy_cost_rate)) / config.lot_size
        ) * config.lot_size
        cash_limited_shares = floor(
            cash / (price * (1.0 + config.buy_cost_rate)) / config.lot_size
        ) * config.lot_size
        buy_shares = int(min(target_limited_shares, cash_limited_shares))
        if buy_shares < config.lot_size:
            logs.append(
                {
                    "trade_date": trade_date,
                    "signal_date": signal_date,
                    "ts_code": ts_code,
                    "action": "buy",
                    "status": "skipped",
                    "reason": "insufficient_cash_or_below_lot_size",
                    "target_weight": target_weights[ts_code],
                    "current_value": current_value,
                    "target_value": target_value,
                    "cash": cash,
                }
            )
            continue

        cash = _execute_buy(
            trade_date=trade_date,
            signal_date=signal_date,
            ts_code=ts_code,
            shares=buy_shares,
            price=price,
            cash=cash,
            holdings=holdings,
            buy_cost_rate=config.buy_cost_rate,
            trades=trades,
            reason="buy_underweight_or_new_target",
            adj_factor=_row_adj_factor(row) or 1.0,
        )
        logs.append(
            {
                "trade_date": trade_date,
                "signal_date": signal_date,
                "ts_code": ts_code,
                "action": "buy",
                "status": "filled",
                "reason": "buy_underweight_or_new_target",
                "shares": buy_shares,
                "target_weight": target_weights[ts_code],
                "current_value": current_value,
                "target_value": target_value,
                "cash": cash,
            }
        )

    if cash < -1e-8:
        raise RuntimeError(f"cash became negative after rebalance: {cash}")
    return max(cash, 0.0)


def _liquidate_positions(
    *,
    trade_date: str,
    date_frame: pd.DataFrame | None,
    holdings: dict[str, Holding],
    cash: float,
    sell_cost_rate: float,
    trades: list[dict[str, Any]],
    logs: list[dict[str, Any]],
) -> tuple[float, float]:
    turnover_amount = 0.0
    for ts_code in list(holdings.keys()):
        holding = holdings[ts_code]
        row = _get_row(date_frame, ts_code)
        _adjust_holding_for_corporate_action(row, holding)
        can_sell, block_reason = _can_sell(row, trade_date, holding)
        current_value = holding.shares * _open_or_last_price(row, holding)
        if not can_sell:
            logs.append(
                {
                    "trade_date": trade_date,
                    "signal_date": trade_date,
                    "ts_code": ts_code,
                    "action": "liquidate",
                    "status": "blocked",
                    "reason": block_reason,
                    "current_value": current_value,
                    "target_value": 0.0,
                }
            )
            continue
        price = float(row["open"])
        sell_shares = holding.shares
        cash = _execute_sell(
            trade_date=trade_date,
            signal_date=trade_date,
            ts_code=ts_code,
            shares=sell_shares,
            price=price,
            cash=cash,
            holding=holding,
            sell_cost_rate=sell_cost_rate,
            trades=trades,
            reason="force_liquidate_on_last_day",
        )
        turnover_amount += sell_shares * price
        logs.append(
            {
                "trade_date": trade_date,
                "signal_date": trade_date,
                "ts_code": ts_code,
                "action": "liquidate",
                "status": "filled",
                "reason": "force_liquidate_on_last_day",
                "shares": sell_shares,
                "current_value": current_value,
                "target_value": 0.0,
            }
        )
        del holdings[ts_code]
    return max(cash, 0.0), turnover_amount


def _value_portfolio_at_close(
    *,
    trade_date: str,
    date_frame: pd.DataFrame | None,
    holdings: dict[str, Holding],
    cash: float,
    target_weights: dict[str, float],
    nav_records: list[dict[str, Any]],
    position_records: list[dict[str, Any]],
    turnover: float,
    initial_cash: float,
) -> None:
    stock_value = 0.0
    holding_values: dict[str, tuple[float, float]] = {}
    for ts_code, holding in holdings.items():
        row = _get_row(date_frame, ts_code)
        _adjust_holding_for_corporate_action(row, holding)
        price = _close_or_last_price(row, holding)
        if row is not None and _valid_price(row.get("close")):
            holding.last_price = price
        current_adj_factor = _row_adj_factor(row)
        if current_adj_factor is not None:
            holding.last_adj_factor = current_adj_factor
        value = holding.shares * price
        stock_value += value
        holding_values[ts_code] = (price, value)

    total_equity = cash + stock_value
    nav = total_equity / initial_cash
    for ts_code, holding in holdings.items():
        price, value = holding_values[ts_code]
        target_weight = target_weights.get(ts_code, 0.0)
        actual_weight = value / total_equity if total_equity > 0 else 0.0
        position_records.append(
            {
                "trade_date": trade_date,
                "ts_code": ts_code,
                "shares": holding.shares,
                "close": price,
                "last_price": holding.last_price,
                "market_value": value,
                "actual_weight": actual_weight,
                "target_weight": target_weight,
                "weight_diff": actual_weight - target_weight,
                "last_buy_date": holding.last_buy_date,
                "is_target": ts_code in target_weights,
                "is_blocked_position": ts_code not in target_weights,
            }
        )

    nav_records.append(
        {
            "trade_date": trade_date,
            "cash": cash,
            "stock_value": stock_value,
            "total_equity": total_equity,
            "nav": nav,
            "num_holdings": len(holdings),
            "target_num_holdings": len(target_weights),
            "cash_ratio": cash / total_equity if total_equity > 0 else 0.0,
            "turnover": turnover / total_equity if total_equity > 0 else 0.0,
        }
    )


def _compute_nav_fields(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty:
        return nav
    enriched = nav.copy()
    enriched["daily_return"] = enriched["nav"].pct_change().fillna(0.0)
    enriched["cummax_nav"] = enriched["nav"].cummax()
    enriched["drawdown"] = enriched["nav"] / enriched["cummax_nav"] - 1.0
    return enriched.drop(columns=["cummax_nav"])


def _compute_metrics(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    config: NDayFactorBacktestConfig,
) -> dict[str, Any]:
    if nav.empty:
        return {}
    total_return = float(nav["nav"].iloc[-1] - 1.0)
    periods = max(len(nav), 1)
    annual_return = float((nav["nav"].iloc[-1]) ** (config.annual_trading_days / periods) - 1.0)
    daily_return = nav["daily_return"]
    annual_volatility = float(daily_return.std(ddof=0) * sqrt(config.annual_trading_days))
    sharpe = (
        float(daily_return.mean() / daily_return.std(ddof=0) * sqrt(config.annual_trading_days))
        if daily_return.std(ddof=0) > 0
        else 0.0
    )
    total_trade_amount = float(trades["amount"].sum()) if not trades.empty else 0.0
    total_cost = float(trades["cost"].sum()) if not trades.empty else 0.0
    return {
        "start_date": nav["trade_date"].iloc[0],
        "end_date": nav["trade_date"].iloc[-1],
        "initial_cash": config.initial_cash,
        "final_equity": float(nav["total_equity"].iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(nav["drawdown"].min()),
        "win_rate": float((daily_return > 0).mean()),
        "average_turnover": float(nav["turnover"].mean()),
        "total_trade_amount": total_trade_amount,
        "total_cost": total_cost,
        "num_trades": int(len(trades)),
        "num_buy_trades": int((trades["side"].eq("buy")).sum()) if not trades.empty else 0,
        "num_sell_trades": int((trades["side"].eq("sell")).sum()) if not trades.empty else 0,
        "average_num_holdings": float(nav["num_holdings"].mean()),
        "average_cash_ratio": float(nav["cash_ratio"].mean()),
        "average_abs_weight_deviation": (
            float(positions["weight_diff"].abs().mean()) if not positions.empty else 0.0
        ),
    }


def _plot_nav_curve(nav: pd.DataFrame, plot_path: str | Path) -> None:
    output_path = Path(plot_path)
    if output_path.suffix.lower() in {".html", ".htm"}:
        _write_nav_curve_html(nav, output_path)
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for non-HTML plot output. "
            "Use a `.html` plot_path or install matplotlib."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(nav["trade_date"]), nav["nav"], label="NAV", linewidth=1.5)
    ax.set_title("Backtest NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_nav_curve_html(nav: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if nav.empty:
        output_path.write_text("<html><body><p>No NAV data.</p></body></html>", encoding="utf-8")
        return

    width = 1000
    height = 480
    pad_left = 64
    pad_right = 24
    pad_top = 32
    pad_bottom = 54
    nav_values = nav["nav"].astype(float).tolist()
    dates = nav["trade_date"].astype(str).tolist()
    min_nav = min(nav_values)
    max_nav = max(nav_values)
    if max_nav == min_nav:
        max_nav += 0.01
        min_nav -= 0.01
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    point_count = len(nav_values)

    points: list[str] = []
    for idx, value in enumerate(nav_values):
        x = pad_left + (plot_width * idx / max(point_count - 1, 1))
        y = pad_top + plot_height * (1 - (value - min_nav) / (max_nav - min_nav))
        points.append(f"{x:.2f},{y:.2f}")

    tick_dates = []
    tick_count = min(6, point_count)
    for idx in sorted({round(i * (point_count - 1) / max(tick_count - 1, 1)) for i in range(tick_count)}):
        x = pad_left + (plot_width * idx / max(point_count - 1, 1))
        tick_dates.append((x, dates[idx]))

    y_ticks = []
    for i in range(5):
        value = min_nav + (max_nav - min_nav) * i / 4
        y = pad_top + plot_height * (1 - (value - min_nav) / (max_nav - min_nav))
        y_ticks.append((y, value))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backtest NAV Curve</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    .summary {{ margin-bottom: 12px; font-size: 14px; }}
    svg {{ max-width: 100%; height: auto; border: 1px solid #d9dee7; background: #fff; }}
    .axis {{ stroke: #6b7280; stroke-width: 1; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .line {{ fill: none; stroke: #2563eb; stroke-width: 2; }}
    .label {{ fill: #374151; font-size: 12px; }}
  </style>
</head>
<body>
  <h2>Backtest NAV Curve</h2>
  <div class="summary">
    Start: {dates[0]} | End: {dates[-1]} | Final NAV: {nav_values[-1]:.6f}
  </div>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Backtest NAV curve">
    <line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" />
    <line class="axis" x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" />
    {''.join(f'<line class="grid" x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" />'
             f'<text class="label" x="8" y="{y + 4:.2f}">{value:.3f}</text>' for y, value in y_ticks)}
    {''.join(f'<line class="grid" x1="{x:.2f}" y1="{pad_top}" x2="{x:.2f}" y2="{height - pad_bottom}" />'
             f'<text class="label" x="{x - 28:.2f}" y="{height - 22}">{date}</text>' for x, date in tick_dates)}
    <polyline class="line" points="{' '.join(points)}" />
  </svg>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def run_ashare_sse_szse_n_day_factor_backtest(
    factor_score: pd.DataFrame,
    backtest_base: pd.DataFrame,
    *,
    strategy_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_cash: float = 1_000_000.0,
    rebalance_interval: int = 5,
    max_holdings: int = 50,
    candidate_pool_size: int | None = None,
    weight_method: str = "equal",
    custom_rank_weights: list[float] | None = None,
    lot_size: int = 100,
    buy_cost_rate: float = 0.0003,
    sell_cost_rate: float = 0.0013,
    score_ascending: bool = False,
    force_liquidate_on_last_day: bool = False,
    annual_trading_days: int = 252,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run an SSE/SZSE N-day factor ranking portfolio backtest.

    The strategy uses signal date T and executes rebalance orders at T+1 open.
    Each rebalance attempts to move the portfolio toward equal or custom rank
    weights while respecting A-share trading constraints and lot sizes.
    """
    config = NDayFactorBacktestConfig(
        strategy_id=strategy_id,
        start_date=_normalize_date(start_date),
        end_date=_normalize_date(end_date),
        initial_cash=initial_cash,
        rebalance_interval=rebalance_interval,
        max_holdings=max_holdings,
        candidate_pool_size=candidate_pool_size,
        weight_method=weight_method,
        custom_rank_weights=tuple(custom_rank_weights) if custom_rank_weights is not None else None,
        lot_size=lot_size,
        buy_cost_rate=buy_cost_rate,
        sell_cost_rate=sell_cost_rate,
        score_ascending=score_ascending,
        force_liquidate_on_last_day=force_liquidate_on_last_day,
        annual_trading_days=annual_trading_days,
    )
    candidate_pool_size, rank_weights = _validate_config(config)
    prepared_base = _prepare_backtest_base(backtest_base, config)
    prepared_factor = _prepare_factor_score(factor_score, config)
    if prepared_base.empty:
        raise ValueError("`backtest_base` has no rows after filtering.")
    if prepared_factor.empty:
        raise ValueError("`factor_score` has no rows after filtering.")

    trade_dates = sorted(prepared_base["trade_date"].unique().tolist())
    factor_dates = set(prepared_factor["trade_date"].unique().tolist())
    rebalance_pairs = [
        pair for pair in _build_rebalance_pairs(trade_dates, config.rebalance_interval) if pair[0] in factor_dates
    ]
    rebalance_by_execution_date = {execution_date: signal_date for signal_date, execution_date in rebalance_pairs}
    base_by_date = _frame_by_date(prepared_base)
    factor_by_date = _frame_by_date(prepared_factor)

    cash = float(config.initial_cash)
    holdings: dict[str, Holding] = {}
    active_target_weights: dict[str, float] = {}
    nav_records: list[dict[str, Any]] = []
    position_records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rebalance_logs: list[dict[str, Any]] = []
    last_trade_date = trade_dates[-1]

    for trade_date in trade_dates:
        date_frame = base_by_date.get(trade_date)
        turnover_amount = 0.0
        signal_date = rebalance_by_execution_date.get(trade_date)
        if signal_date is not None:
            total_equity_at_open, _, _ = _portfolio_value_at_open(holdings, date_frame, cash)
            target_weights, target_order, target_logs = _build_target_portfolio(
                signal_date=signal_date,
                execution_date=trade_date,
                factor_by_date=factor_by_date,
                base_by_date=base_by_date,
                holdings=holdings,
                total_equity=total_equity_at_open,
                candidate_pool_size=candidate_pool_size,
                max_holdings=config.max_holdings,
                rank_weights=rank_weights,
                lot_size=config.lot_size,
                buy_cost_rate=config.buy_cost_rate,
                score_ascending=config.score_ascending,
            )
            rebalance_logs.extend(target_logs)
            before_trade_count = len(trades)
            cash = _rebalance(
                trade_date=trade_date,
                signal_date=signal_date,
                date_frame=date_frame,
                holdings=holdings,
                cash=cash,
                target_weights=target_weights,
                target_order=target_order,
                config=config,
                trades=trades,
                logs=rebalance_logs,
            )
            turnover_amount = sum(trade["amount"] for trade in trades[before_trade_count:])
            active_target_weights = target_weights

        if config.force_liquidate_on_last_day and trade_date == last_trade_date and holdings:
            cash, liquidation_turnover = _liquidate_positions(
                trade_date=trade_date,
                date_frame=date_frame,
                holdings=holdings,
                cash=cash,
                sell_cost_rate=config.sell_cost_rate,
                trades=trades,
                logs=rebalance_logs,
            )
            turnover_amount += liquidation_turnover
            active_target_weights = {}

        _value_portfolio_at_close(
            trade_date=trade_date,
            date_frame=date_frame,
            holdings=holdings,
            cash=cash,
            target_weights=active_target_weights,
            nav_records=nav_records,
            position_records=position_records,
            turnover=turnover_amount,
            initial_cash=config.initial_cash,
        )

    nav = _compute_nav_fields(pd.DataFrame(nav_records))
    trades_df = pd.DataFrame(trades)
    positions = pd.DataFrame(position_records)
    rebalance_logs_df = pd.DataFrame(rebalance_logs)
    metrics = _compute_metrics(nav, trades_df, positions, config)

    if plot_path is not None:
        _plot_nav_curve(nav, plot_path)

    return {
        "nav": nav,
        "positions": positions,
        "trades": trades_df,
        "rebalance_logs": rebalance_logs_df,
        "metrics": metrics,
        "config": asdict(config) | {"candidate_pool_size": candidate_pool_size, "rank_weights": rank_weights},
    }
