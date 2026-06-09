"""A-share data-fetch package."""

from .ashare_sse_szse_daily_backtest_base import fetch_ashare_sse_szse_daily_backtest_base
from .ashare_sse_szse_daily_technical_supplement import fetch_ashare_sse_szse_daily_technical_supplement

__all__ = [
    "fetch_ashare_sse_szse_daily_backtest_base",
    "fetch_ashare_sse_szse_daily_technical_supplement",
]
