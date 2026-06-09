"""SSE/SZSE A-share daily technical supplement table built from Tushare daily_basic."""

from __future__ import annotations

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
DEFAULT_OUTPUT_PATH = A_SHARE_DATABASE_ROOT / "ashare_sse_szse_daily_technical_supplement.parquet"
SSE_SZSE_TS_CODE_SUFFIXES = (".SH", ".SZ")


@dataclass(frozen=True)
class SseSzseTechnicalSupplementConfig:
    """Configuration for one SSE/SZSE A-share daily technical supplement fetch."""

    start_date: str
    end_date: str
    ts_codes: list[str] | None = None
    output_path: Path = DEFAULT_OUTPUT_PATH
    sleep_seconds: float = 0.2
    save: bool = True
    return_data: bool = True


OUTPUT_COLUMNS = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "total_mv",
    "circ_mv",
    "pe",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "free_share",
    "total_share",
    "circ_share",
    "data_vendor",
    "panel_name",
]

NUMERIC_OUTPUT_COLUMNS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "total_mv",
    "circ_mv",
    "pe",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "free_share",
    "total_share",
    "circ_share",
]

STRING_OUTPUT_COLUMNS = [
    "ts_code",
    "trade_date",
    "data_vendor",
    "panel_name",
]

COMPONENT_COLUMNS = {
    "daily_basic": [
        "ts_code",
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
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
                    f"[ashare_sse_szse_daily_technical_supplement] retry {api_name} attempt={attempt + 1}/3",
                    flush=True,
                )
                sleep(max(sleep_seconds, 0.5) * attempt)
                continue
            raise RuntimeError(f"{api_name} failed: {exc}") from exc

    raise RuntimeError(f"{api_name} failed: {last_error}")


def _print_progress(current_step: int, total_steps: int, label: str) -> None:
    percentage = current_step / total_steps * 100
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    message = (
        f"[ashare_sse_szse_daily_technical_supplement] {percentage:5.1f}% "
        f"({current_step}/{total_steps}) {label}"
    )
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

    safe_row_limit = 6000
    chunk_size = max(1, safe_row_limit // len(ts_codes))
    return min(chunk_size, 240)


def _fetch_trade_dates(config: SseSzseTechnicalSupplementConfig) -> list[str]:
    pro = get_tushare_pro()
    trade_cal = _safe_call(
        pro.trade_cal,
        api_name="trade_cal",
        sleep_seconds=config.sleep_seconds,
        exchange="SSE",
        start_date=config.start_date,
        end_date=config.end_date,
    )
    if trade_cal.empty:
        return []
    trade_cal = trade_cal.loc[trade_cal["is_open"].astype(int) == 1, ["cal_date"]].drop_duplicates()
    return sorted(trade_cal["cal_date"].astype(str).tolist())


def _build_date_chunks(trade_dates: list[str], chunk_size: int) -> list[tuple[str, str]]:
    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    return [
        (chunk[0], chunk[-1])
        for chunk in (trade_dates[idx : idx + chunk_size] for idx in range(0, len(trade_dates), chunk_size))
        if chunk
    ]


def _build_component_kwargs(
    config: SseSzseTechnicalSupplementConfig,
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
                "[ashare_sse_szse_daily_technical_supplement] warning "
                f"{component_name} missing columns={missing_columns}; "
                "using empty normalized frame for this chunk"
            ),
            flush=True,
        )
        return pd.DataFrame(columns=expected_columns)

    return normalized[expected_columns].copy()


def _fetch_daily_basic_chunk(
    config: SseSzseTechnicalSupplementConfig,
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
    if config.ts_codes:
        common_kwargs["ts_code"] = ",".join(config.ts_codes)

    frame = _safe_call(
        pro.daily_basic,
        api_name="daily_basic",
        sleep_seconds=config.sleep_seconds,
        fields=(
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pb,ps_ttm,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
        ),
        **common_kwargs,
    )
    if config.sleep_seconds > 0:
        sleep(config.sleep_seconds)
    return _normalize_component_frame(frame, "daily_basic")


def _build_technical_supplement_panel(daily_basic: pd.DataFrame) -> pd.DataFrame:
    if daily_basic.empty:
        return pd.DataFrame()

    panel = daily_basic.copy()
    if "float_share" in panel.columns and "circ_share" not in panel.columns:
        panel["circ_share"] = panel["float_share"]

    panel["data_vendor"] = "tushare"
    panel["panel_name"] = "ashare_sse_szse_daily_technical_supplement"
    return panel.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _enforce_output_dtypes(panel: pd.DataFrame) -> pd.DataFrame:
    normalized = panel.copy()
    for column in STRING_OUTPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype="string")
        else:
            normalized[column] = normalized[column].astype("string")

    for column in NUMERIC_OUTPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype="float64")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")

    return normalized


def _select_output_columns(panel: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in panel.columns:
            panel[column] = pd.NA
    panel = _enforce_output_dtypes(panel)
    return panel[OUTPUT_COLUMNS].copy()


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


def _empty_output_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def fetch_ashare_sse_szse_daily_technical_supplement(
    start_date: str,
    end_date: str,
    ts_codes: Iterable[str] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sleep_seconds: float = 0.2,
    save: bool = True,
    return_data: bool = True,
) -> pd.DataFrame:
    """Fetch an SSE/SZSE A-share daily technical supplement table with a custom date range."""
    config = SseSzseTechnicalSupplementConfig(
        start_date=_normalize_date(start_date),
        end_date=_normalize_date(end_date),
        ts_codes=_normalize_ts_codes(ts_codes),
        output_path=Path(output_path),
        sleep_seconds=sleep_seconds,
        save=save,
        return_data=return_data,
    )
    if config.start_date > config.end_date:
        raise ValueError("`start_date` must be less than or equal to `end_date`.")
    if config.ts_codes == []:
        return _empty_output_frame()

    print("[ashare_sse_szse_daily_technical_supplement] start fetching SSE/SZSE A-share technical supplement", flush=True)
    trade_dates = _fetch_trade_dates(config)
    if not trade_dates:
        print("[ashare_sse_szse_daily_technical_supplement] no trading days in range", flush=True)
        return _empty_output_frame()

    chunk_size = _resolve_chunk_size(config.ts_codes)
    date_chunks = _build_date_chunks(trade_dates, chunk_size)
    print(
        (
            "[ashare_sse_szse_daily_technical_supplement] "
            f"trade_days={len(trade_dates)} chunks={len(date_chunks)} chunk_size={chunk_size}"
        ),
        flush=True,
    )

    temp_output_path = config.output_path.with_name(f".{config.output_path.stem}.tmp.parquet")
    if temp_output_path.exists():
        temp_output_path.unlink()

    collected_panels: list[pd.DataFrame] = []
    should_collect = config.return_data and (not config.save or len(date_chunks) <= 20)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    if config.save:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)

    for chunk_index, (chunk_start_date, chunk_end_date) in enumerate(date_chunks, start=1):
        daily_basic = _fetch_daily_basic_chunk(
            config,
            chunk_start_date=chunk_start_date,
            chunk_end_date=chunk_end_date,
        )
        panel = _build_technical_supplement_panel(daily_basic)
        panel = _select_output_columns(panel)
        panel = panel.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        total_rows += len(panel)

        if should_collect:
            collected_panels.append(panel)
        if config.save:
            writer = _append_parquet_chunk(writer, panel, temp_output_path)

        _print_progress(chunk_index, len(date_chunks), f"rows={total_rows:>10}")

    if writer is not None:
        writer.close()

    if config.save:
        if config.output_path.exists():
            config.output_path.unlink()
        if temp_output_path.exists():
            temp_output_path.replace(config.output_path)
        else:
            _write_parquet_frame(_empty_output_frame(), config.output_path)
        _finish_progress_line()
        print(
            f"[ashare_sse_szse_daily_technical_supplement] saved rows={total_rows} -> {config.output_path}",
            flush=True,
        )

    if should_collect:
        if not collected_panels:
            if not config.save:
                _finish_progress_line()
            return _empty_output_frame()
        if not config.save:
            _finish_progress_line()
        return pd.concat(collected_panels, ignore_index=True)

    if config.return_data and config.save and config.output_path.exists():
        return pd.read_parquet(config.output_path)

    if not config.save:
        _finish_progress_line()
    return _empty_output_frame()
