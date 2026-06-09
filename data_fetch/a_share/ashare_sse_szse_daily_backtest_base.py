"""SSE/SZSE A-share backtest base bundle built from Tushare market and status data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shutil
from time import sleep
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .ashare_tushare_client import get_tushare_pro


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_LAYER_ROOT = PROJECT_ROOT / "base_layer"
A_SHARE_DATABASE_ROOT = BASE_LAYER_ROOT / "a_share"
DEFAULT_OUTPUT_DIR = A_SHARE_DATABASE_ROOT / "ashare_sse_szse_daily_backtest_bundle"

BACKTEST_BASE_FILENAME = "ashare_sse_szse_daily_backtest_base.parquet"
STOCK_STATUS_FILENAME = "ashare_sse_szse_stock_status.parquet"
ST_DAILY_STATUS_FILENAME = "ashare_sse_szse_st_status_daily.parquet"
SSE_SZSE_TS_CODE_SUFFIXES = (".SH", ".SZ")


@dataclass(frozen=True)
class SseSzseBacktestBaseConfig:
    """Configuration for one SSE/SZSE A-share backtest-base bundle fetch."""

    start_date: str
    end_date: str
    ts_codes: list[str] | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    sleep_seconds: float = 0.2
    save: bool = True
    return_data: bool = True


BACKTEST_BASE_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "adj_factor",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "up_limit",
    "down_limit",
    "is_close_limit_up",
    "is_close_limit_down",
    "is_open_limit_up",
    "is_open_limit_down",
    "is_suspended",
    "can_buy_open",
    "can_sell_open",
    "data_vendor",
    "panel_name",
]

BACKTEST_BASE_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "adj_factor",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "up_limit",
    "down_limit",
]

BACKTEST_BASE_BOOLEAN_COLUMNS = [
    "is_close_limit_up",
    "is_close_limit_down",
    "is_open_limit_up",
    "is_open_limit_down",
    "is_suspended",
    "can_buy_open",
    "can_sell_open",
]

BACKTEST_BASE_STRING_COLUMNS = [
    "ts_code",
    "trade_date",
    "data_vendor",
    "panel_name",
]

STOCK_STATUS_COLUMNS = [
    "ts_code",
    "name",
    "list_date",
    "delist_date",
    "list_status",
    "data_vendor",
    "panel_name",
]

STOCK_STATUS_STRING_COLUMNS = [
    "ts_code",
    "name",
    "list_date",
    "delist_date",
    "list_status",
    "data_vendor",
    "panel_name",
]

ST_DAILY_STATUS_COLUMNS = [
    "ts_code",
    "trade_date",
    "is_st",
    "st_type",
    "st_type_name",
    "data_vendor",
    "panel_name",
]

ST_DAILY_STATUS_BOOLEAN_COLUMNS = [
    "is_st",
]

ST_DAILY_STATUS_STRING_COLUMNS = [
    "ts_code",
    "trade_date",
    "st_type",
    "st_type_name",
    "data_vendor",
    "panel_name",
]

COMPONENT_COLUMNS = {
    "daily": [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ],
    "adj_factor": [
        "ts_code",
        "trade_date",
        "adj_factor",
    ],
    "stk_limit": [
        "ts_code",
        "trade_date",
        "up_limit",
        "down_limit",
    ],
    "suspend_d": [
        "ts_code",
        "trade_date",
    ],
    "stock_status": [
        "ts_code",
        "name",
        "list_date",
        "delist_date",
        "list_status",
    ],
    "stock_st": [
        "ts_code",
        "trade_date",
        "type",
        "type_name",
    ],
}


def _normalize_date(date_text: str) -> str:
    digits = "".join(ch for ch in str(date_text) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Invalid date: {date_text!r}. Expected YYYYMMDD.")
    return digits


def _is_sse_szse_ts_code(ts_code: str) -> bool:
    return str(ts_code).strip().upper().endswith(SSE_SZSE_TS_CODE_SUFFIXES)


def _filter_sse_szse_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return frame
    mask = frame["ts_code"].astype("string").str.upper().str.endswith(SSE_SZSE_TS_CODE_SUFFIXES, na=False)
    return frame.loc[mask].copy()


def _normalize_ts_codes(ts_codes: Iterable[str] | None) -> list[str] | None:
    if ts_codes is None:
        return None
    normalized = sorted(
        {
            str(code).strip().upper()
            for code in ts_codes
            if str(code).strip() and _is_sse_szse_ts_code(str(code))
        }
    )
    return normalized


def _safe_call(func, *, api_name: str, sleep_seconds: float, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            df = func(**kwargs)
            if df is None:
                return pd.DataFrame()
            return df.copy()
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                print(
                    f"[ashare_sse_szse_daily_backtest_base] retry {api_name} attempt={attempt + 1}/3",
                    flush=True,
                )
                sleep(max(sleep_seconds, 0.5) * attempt)
                continue
            raise RuntimeError(f"{api_name} failed: {exc}") from exc

    raise RuntimeError(f"{api_name} failed: {last_error}")


def _print_progress(current_step: int, total_steps: int, label: str) -> None:
    percentage = current_step / total_steps * 100
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    message = f"[ashare_sse_szse_daily_backtest_base] {percentage:5.1f}% ({current_step}/{total_steps}) {label}"
    max_message_width = max(20, terminal_width - 1)
    if len(message) > max_message_width:
        message = message[: max_message_width - 3] + "..."
    print(
        "\r\033[2K" + message.ljust(max_message_width),
        end="",
        flush=True,
    )


def _finish_progress_line() -> None:
    print("", flush=True)


def _resolve_chunk_size(ts_codes: list[str] | None) -> int:
    if not ts_codes:
        return 1

    safe_row_limit = 5800
    chunk_size = max(1, safe_row_limit // len(ts_codes))
    return min(chunk_size, 240)


def _slice_trade_calendar(trade_cal: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    if trade_cal.empty:
        return []
    mask = (
        trade_cal["cal_date"].astype(str).between(start_date, end_date)
        & (trade_cal["is_open"].astype(int) == 1)
    )
    return sorted(trade_cal.loc[mask, "cal_date"].astype(str).tolist())


def _fetch_trade_calendar(
    config: SseSzseBacktestBaseConfig,
) -> pd.DataFrame:
    pro = get_tushare_pro()
    trade_cal = _safe_call(
        pro.trade_cal,
        api_name=f"trade_cal[{config.start_date}-{config.end_date}]",
        sleep_seconds=config.sleep_seconds,
        exchange="SSE",
        start_date=config.start_date,
        end_date=config.end_date,
    )
    if config.sleep_seconds > 0:
        sleep(config.sleep_seconds)
    if trade_cal.empty:
        return pd.DataFrame(columns=["cal_date", "is_open"])
    return trade_cal[["cal_date", "is_open"]].copy()


def _fetch_trade_dates(config: SseSzseBacktestBaseConfig) -> list[str]:
    print(
        "[ashare_sse_szse_daily_backtest_base] fetching trade calendar from Tushare",
        flush=True,
    )
    trade_cal = _fetch_trade_calendar(config)
    return _slice_trade_calendar(trade_cal, config.start_date, config.end_date)


def _build_date_chunks(trade_dates: list[str], chunk_size: int) -> list[tuple[str, str]]:
    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    return [
        (chunk[0], chunk[-1])
        for chunk in (trade_dates[idx : idx + chunk_size] for idx in range(0, len(trade_dates), chunk_size))
        if chunk
    ]


def _normalize_component_frame(frame: pd.DataFrame, component_name: str) -> pd.DataFrame:
    expected_columns = COMPONENT_COLUMNS[component_name]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=expected_columns)

    normalized = frame.copy()
    normalized = _filter_sse_szse_frame(normalized)
    missing_columns = [column for column in expected_columns if column not in normalized.columns]
    if missing_columns:
        print(
            (
                "[ashare_sse_szse_daily_backtest_base] warning "
                f"{component_name} missing columns={missing_columns}; "
                "using empty normalized frame for this chunk"
            ),
            flush=True,
        )
        return pd.DataFrame(columns=expected_columns)

    return normalized[expected_columns].copy()


def _build_component_kwargs(
    config: SseSzseBacktestBaseConfig,
    *,
    chunk_start_date: str,
    chunk_end_date: str,
) -> dict[str, str]:
    if chunk_start_date == chunk_end_date:
        return {"trade_date": chunk_start_date}
    return {
        "start_date": chunk_start_date,
        "end_date": chunk_end_date,
    }


def _build_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "backtest_base": output_dir / BACKTEST_BASE_FILENAME,
        "stock_status": output_dir / STOCK_STATUS_FILENAME,
        "st_status_daily": output_dir / ST_DAILY_STATUS_FILENAME,
    }


def _append_parquet_chunk(
    writer: pq.ParquetWriter | None,
    panel: pd.DataFrame,
    output_path: Path,
) -> pq.ParquetWriter | None:
    if panel.empty:
        return writer

    table = pa.Table.from_pandas(panel, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
    writer.write_table(table)
    return writer


def _write_parquet_frame(panel: pd.DataFrame, output_path: Path) -> None:
    table = pa.Table.from_pandas(panel, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _is_near_price(left: pd.Series, right: pd.Series) -> pd.Series:
    valid = left.notna() & right.notna()
    return valid & ((left - right).abs() <= 0.0001)


def _fetch_backtest_components(
    config: SseSzseBacktestBaseConfig,
    *,
    chunk_start_date: str,
    chunk_end_date: str,
) -> dict[str, pd.DataFrame]:
    pro = get_tushare_pro()
    common_kwargs = _build_component_kwargs(
        config,
        chunk_start_date=chunk_start_date,
        chunk_end_date=chunk_end_date,
    )
    if config.ts_codes:
        common_kwargs["ts_code"] = ",".join(config.ts_codes)

    api_requests = {
        "daily": (
            pro.daily,
            {
                "api_name": "daily",
                "sleep_seconds": config.sleep_seconds,
                **common_kwargs,
            },
        ),
        "adj_factor": (
            pro.adj_factor,
            {
                "api_name": "adj_factor",
                "sleep_seconds": config.sleep_seconds,
                **common_kwargs,
            },
        ),
        "stk_limit": (
            pro.stk_limit,
            {
                "api_name": "stk_limit",
                "sleep_seconds": config.sleep_seconds,
                **common_kwargs,
            },
        ),
        "suspend_d": (
            pro.suspend_d,
            {
                "api_name": "suspend_d",
                "sleep_seconds": config.sleep_seconds,
                "suspend_type": "S",
                **common_kwargs,
            },
        ),
    }

    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=len(api_requests)) as executor:
        future_map = {
            component_name: executor.submit(_safe_call, func, **kwargs)
            for component_name, (func, kwargs) in api_requests.items()
        }
        for component_name, future in future_map.items():
            frames[component_name] = _normalize_component_frame(future.result(), component_name)

    if config.sleep_seconds > 0:
        sleep(config.sleep_seconds)

    return frames


def _prepare_backtest_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    suspend_d = frames["suspend_d"].copy()
    if not suspend_d.empty:
        suspend_d = suspend_d.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        suspend_d["is_suspended"] = True

    return {
        "daily": frames["daily"].copy(),
        "adj_factor": frames["adj_factor"].copy(),
        "stk_limit": frames["stk_limit"].copy(),
        "suspend_d": suspend_d,
    }


def _build_backtest_base_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    daily = frames["daily"]
    suspend_d = frames["suspend_d"]
    if daily.empty and suspend_d.empty:
        return pd.DataFrame()

    panel = daily.merge(suspend_d, on=["ts_code", "trade_date"], how="outer")
    panel = panel.merge(frames["adj_factor"], on=["ts_code", "trade_date"], how="left")
    panel = panel.merge(frames["stk_limit"], on=["ts_code", "trade_date"], how="left")

    if "adj_factor" in panel.columns:
        for raw_col, adj_col in {
            "open": "adj_open",
            "high": "adj_high",
            "low": "adj_low",
            "close": "adj_close",
        }.items():
            if raw_col in panel.columns:
                panel[adj_col] = panel[raw_col] * panel["adj_factor"]

    if "is_suspended" in panel.columns:
        panel["is_suspended"] = panel["is_suspended"].eq(True)
    else:
        panel["is_suspended"] = False

    panel["is_close_limit_up"] = False
    if "close" in panel.columns and "up_limit" in panel.columns:
        panel["is_close_limit_up"] = _is_near_price(panel["close"], panel["up_limit"])

    panel["is_close_limit_down"] = False
    if "close" in panel.columns and "down_limit" in panel.columns:
        panel["is_close_limit_down"] = _is_near_price(panel["close"], panel["down_limit"])

    panel["is_open_limit_up"] = False
    if "open" in panel.columns and "up_limit" in panel.columns:
        panel["is_open_limit_up"] = _is_near_price(panel["open"], panel["up_limit"])

    panel["is_open_limit_down"] = False
    if "open" in panel.columns and "down_limit" in panel.columns:
        panel["is_open_limit_down"] = _is_near_price(panel["open"], panel["down_limit"])

    has_open_price = panel["open"].notna() if "open" in panel.columns else False
    panel["can_buy_open"] = has_open_price & ~panel["is_suspended"] & ~panel["is_open_limit_up"]
    panel["can_sell_open"] = has_open_price & ~panel["is_suspended"] & ~panel["is_open_limit_down"]

    panel["data_vendor"] = "tushare"
    panel["panel_name"] = "ashare_sse_szse_daily_backtest_base"
    return panel.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _enforce_backtest_base_dtypes(panel: pd.DataFrame) -> pd.DataFrame:
    normalized = panel.copy()
    for column in BACKTEST_BASE_STRING_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype="string")
        else:
            normalized[column] = normalized[column].astype("string")

    for column in BACKTEST_BASE_NUMERIC_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype="float64")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")

    for column in BACKTEST_BASE_BOOLEAN_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.Series(False, index=normalized.index, dtype="bool")
        else:
            normalized[column] = normalized[column].fillna(False).astype("bool")

    return normalized


def _select_backtest_base_columns(panel: pd.DataFrame) -> pd.DataFrame:
    for column in BACKTEST_BASE_COLUMNS:
        if column not in panel.columns:
            panel[column] = pd.NA
    panel = _enforce_backtest_base_dtypes(panel)
    return panel[BACKTEST_BASE_COLUMNS].copy()


def _fetch_stock_status(config: SseSzseBacktestBaseConfig) -> pd.DataFrame:
    pro = get_tushare_pro()
    status_frames: list[pd.DataFrame] = []
    for list_status in ("L", "P", "D"):
        frame = _safe_call(
            pro.stock_basic,
            api_name=f"stock_basic[{list_status}]",
            sleep_seconds=config.sleep_seconds,
            exchange="",
            list_status=list_status,
            fields="ts_code,name,list_date,delist_date,list_status",
        )
        status_frames.append(_normalize_component_frame(frame, "stock_status"))
        if config.sleep_seconds > 0:
            sleep(config.sleep_seconds)

    stock_status = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame()
    if stock_status.empty:
        return _empty_frame(STOCK_STATUS_COLUMNS)

    stock_status = stock_status.drop_duplicates(subset=["ts_code"], keep="last")
    if config.ts_codes:
        stock_status = stock_status[stock_status["ts_code"].isin(config.ts_codes)].copy()

    stock_status["data_vendor"] = "tushare"
    stock_status["panel_name"] = "ashare_sse_szse_stock_status"
    for column in STOCK_STATUS_COLUMNS:
        if column not in stock_status.columns:
            stock_status[column] = pd.NA
    for column in STOCK_STATUS_STRING_COLUMNS:
        stock_status[column] = stock_status[column].astype("string")
    return stock_status[STOCK_STATUS_COLUMNS].sort_values(["ts_code"], kind="stable").reset_index(drop=True)


def _fetch_stock_st_chunk(
    config: SseSzseBacktestBaseConfig,
    *,
    chunk_start_date: str,
    chunk_end_date: str,
) -> pd.DataFrame:
    pro = get_tushare_pro()
    common_kwargs = _build_component_kwargs(
        config,
        chunk_start_date=chunk_start_date,
        chunk_end_date=chunk_end_date,
    )
    frame = _safe_call(
        pro.stock_st,
        api_name="stock_st",
        sleep_seconds=config.sleep_seconds,
        **common_kwargs,
    )
    if config.sleep_seconds > 0:
        sleep(config.sleep_seconds)

    frame = _normalize_component_frame(frame, "stock_st")
    if frame.empty:
        return _empty_frame(ST_DAILY_STATUS_COLUMNS)

    if config.ts_codes:
        frame = frame[frame["ts_code"].isin(config.ts_codes)].copy()

    frame["is_st"] = True
    frame["st_type"] = frame["type"]
    frame["st_type_name"] = frame["type_name"]
    frame["data_vendor"] = "tushare"
    frame["panel_name"] = "ashare_sse_szse_st_status_daily"
    for column in ST_DAILY_STATUS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    for column in ST_DAILY_STATUS_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    frame["is_st"] = frame["is_st"].fillna(False).astype("bool")
    return frame[ST_DAILY_STATUS_COLUMNS].sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _clean_backtest_base_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel

    cleaned = panel.copy()
    cleaned["trade_date"] = cleaned["trade_date"].astype("string")
    cleaned = cleaned.dropna(subset=["ts_code", "trade_date"])
    cleaned = cleaned.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    cleaned = cleaned.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    return cleaned


def _clean_st_status_panel(
    panel: pd.DataFrame,
    *,
    valid_trade_dates: set[str],
) -> pd.DataFrame:
    if panel.empty:
        return panel

    cleaned = panel.copy()
    cleaned["trade_date"] = cleaned["trade_date"].astype("string")
    cleaned = cleaned[cleaned["trade_date"].isin(valid_trade_dates)].copy()
    cleaned = cleaned.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    cleaned = cleaned.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    return cleaned


def fetch_ashare_sse_szse_daily_backtest_base(
    start_date: str,
    end_date: str,
    ts_codes: Iterable[str] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sleep_seconds: float = 0.2,
    save: bool = True,
    return_data: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch an SSE/SZSE A-share backtest base bundle with a custom date range.

    The bundle contains:
    - `backtest_base`: daily trading base table
    - `stock_status`: stock-level status table
    - `st_status_daily`: daily ST status table
    """
    config = SseSzseBacktestBaseConfig(
        start_date=_normalize_date(start_date),
        end_date=_normalize_date(end_date),
        ts_codes=_normalize_ts_codes(ts_codes),
        output_dir=Path(output_dir),
        sleep_seconds=sleep_seconds,
        save=save,
        return_data=return_data,
    )
    if config.start_date > config.end_date:
        raise ValueError("`start_date` must be less than or equal to `end_date`.")
    if config.ts_codes == []:
        return {
            "backtest_base": _empty_frame(BACKTEST_BASE_COLUMNS),
            "stock_status": _empty_frame(STOCK_STATUS_COLUMNS),
            "st_status_daily": _empty_frame(ST_DAILY_STATUS_COLUMNS),
        }

    print("[ashare_sse_szse_daily_backtest_base] start fetching SSE/SZSE A-share backtest bundle", flush=True)
    trade_dates = _fetch_trade_dates(config)
    if not trade_dates:
        print("[ashare_sse_szse_daily_backtest_base] no trading days in range", flush=True)
        return {
            "backtest_base": _empty_frame(BACKTEST_BASE_COLUMNS),
            "stock_status": _empty_frame(STOCK_STATUS_COLUMNS),
            "st_status_daily": _empty_frame(ST_DAILY_STATUS_COLUMNS),
        }

    chunk_size = _resolve_chunk_size(config.ts_codes)
    date_chunks = _build_date_chunks(trade_dates, chunk_size)
    total_steps = len(date_chunks) * 2 + 1
    print(
        (
            "[ashare_sse_szse_daily_backtest_base] "
            f"trade_days={len(trade_dates)} chunks={len(date_chunks)} chunk_size={chunk_size}"
        ),
        flush=True,
    )

    output_paths = _build_output_paths(config.output_dir)
    temp_base_path = output_paths["backtest_base"].with_name(f".{output_paths['backtest_base'].stem}.tmp.parquet")
    temp_st_path = output_paths["st_status_daily"].with_name(f".{output_paths['st_status_daily'].stem}.tmp.parquet")
    for temp_path in (temp_base_path, temp_st_path):
        if temp_path.exists():
            temp_path.unlink()

    collect_backtest = config.return_data and (not config.save or len(date_chunks) <= 20)
    collect_st = config.return_data and (not config.save or len(date_chunks) <= 20)
    backtest_frames: list[pd.DataFrame] = []
    st_frames: list[pd.DataFrame] = []
    backtest_writer: pq.ParquetWriter | None = None
    st_writer: pq.ParquetWriter | None = None
    total_backtest_rows = 0
    total_st_rows = 0
    progress_step = 0
    valid_trade_dates: set[str] = set()

    if config.save:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start_date, chunk_end_date in date_chunks:
        raw_frames = _fetch_backtest_components(
            config,
            chunk_start_date=chunk_start_date,
            chunk_end_date=chunk_end_date,
        )
        prepared_frames = _prepare_backtest_frames(raw_frames)
        panel = _build_backtest_base_panel(prepared_frames)
        panel = _select_backtest_base_columns(panel)
        panel = _clean_backtest_base_panel(panel)
        if not panel.empty:
            valid_trade_dates.update(panel["trade_date"].astype(str).tolist())
        total_backtest_rows += len(panel)
        if collect_backtest:
            backtest_frames.append(panel)
        if config.save:
            backtest_writer = _append_parquet_chunk(backtest_writer, panel, temp_base_path)

        progress_step += 1
        _print_progress(progress_step, total_steps, f"base_rows={total_backtest_rows:>10}")

    if backtest_writer is not None:
        backtest_writer.close()

    stock_status = _fetch_stock_status(config)
    progress_step += 1
    _print_progress(progress_step, total_steps, f"status_rows={len(stock_status):>8}")

    for chunk_start_date, chunk_end_date in date_chunks:
        st_panel = _fetch_stock_st_chunk(
            config,
            chunk_start_date=chunk_start_date,
            chunk_end_date=chunk_end_date,
        )
        st_panel = _clean_st_status_panel(st_panel, valid_trade_dates=valid_trade_dates)
        total_st_rows += len(st_panel)
        if collect_st:
            st_frames.append(st_panel)
        if config.save:
            st_writer = _append_parquet_chunk(st_writer, st_panel, temp_st_path)

        progress_step += 1
        _print_progress(progress_step, total_steps, f"st_rows={total_st_rows:>10}")

    if st_writer is not None:
        st_writer.close()

    if not valid_trade_dates:
        valid_trade_dates.update(trade_dates)

    backtest_base = (
        pd.concat(backtest_frames, ignore_index=True)
        if collect_backtest and backtest_frames
        else _empty_frame(BACKTEST_BASE_COLUMNS)
    )
    st_status_daily = (
        pd.concat(st_frames, ignore_index=True)
        if collect_st and st_frames
        else _empty_frame(ST_DAILY_STATUS_COLUMNS)
    )

    if config.save:
        if output_paths["backtest_base"].exists():
            output_paths["backtest_base"].unlink()
        if temp_base_path.exists():
            temp_base_path.replace(output_paths["backtest_base"])
        else:
            _write_parquet_frame(_empty_frame(BACKTEST_BASE_COLUMNS), output_paths["backtest_base"])

        stock_status.to_parquet(output_paths["stock_status"], index=False)

        if output_paths["st_status_daily"].exists():
            output_paths["st_status_daily"].unlink()
        if temp_st_path.exists():
            temp_st_path.replace(output_paths["st_status_daily"])
        else:
            _write_parquet_frame(_empty_frame(ST_DAILY_STATUS_COLUMNS), output_paths["st_status_daily"])

        _finish_progress_line()
        print(
            (
                "[ashare_sse_szse_daily_backtest_base] saved bundle -> "
                f"{config.output_dir}"
            ),
            flush=True,
        )

    if config.return_data and config.save and not collect_backtest and output_paths["backtest_base"].exists():
        backtest_base = pd.read_parquet(output_paths["backtest_base"])
    if config.return_data and config.save and not collect_st and output_paths["st_status_daily"].exists():
        st_status_daily = pd.read_parquet(output_paths["st_status_daily"])

    if not config.save:
        _finish_progress_line()

    return {
        "backtest_base": backtest_base,
        "stock_status": stock_status,
        "st_status_daily": st_status_daily,
    }
