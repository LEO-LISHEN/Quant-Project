"""Tushare client helpers for A-share data fetching."""

from __future__ import annotations

from contextlib import suppress
from functools import lru_cache
import os
from pathlib import Path
from functools import partial

import pandas as pd
import requests

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _configure_tushare_network() -> None:
    """Disable proxy inheritance for Tushare unless explicitly enabled."""
    if os.getenv("TUSHARE_USE_SYSTEM_PROXY", "").strip().lower() in {"1", "true", "yes"}:
        return

    for env_name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        with suppress(KeyError):
            del os.environ[env_name]


class DirectTushareDataApi:
    """Tushare Pro client that ignores system proxy settings by default."""

    def __init__(self, token: str, timeout: int = 120) -> None:
        self._token = token
        self._timeout = timeout
        self._http_url = "http://api.waditu.com/dataapi"
        self._session = requests.Session()
        self._session.trust_env = False

    def query(self, api_name: str, fields: str = "", **kwargs) -> pd.DataFrame:
        kwargs.setdefault("ts_type_name", self._http_url)
        req_params = {
            "api_name": api_name,
            "token": self._token,
            "params": kwargs,
            "fields": fields,
        }
        response = self._session.post(
            f"{self._http_url}/{api_name}",
            json=req_params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result["code"] != 0:
            raise RuntimeError(result["msg"])
        data = result["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])

    def __getattr__(self, name: str):
        return partial(self.query, name)


@lru_cache(maxsize=1)
def get_tushare_pro():
    """Return a cached Tushare Pro client loaded from project `.env`."""
    load_dotenv(PROJECT_ROOT / ".env")
    _configure_tushare_network()
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_KEY")
    timeout_text = os.getenv("TUSHARE_TIMEOUT_SECONDS", "120").strip()
    if not token:
        raise RuntimeError(
            "Missing Tushare token. Please set `TUSHARE_TOKEN` or "
            "`TUSHARE_API_KEY` in the project root `.env` file."
        )
    try:
        timeout = max(30, int(timeout_text))
    except ValueError as exc:
        raise RuntimeError("`TUSHARE_TIMEOUT_SECONDS` must be an integer.") from exc
    return DirectTushareDataApi(token, timeout=timeout)
