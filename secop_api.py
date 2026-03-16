"""Cliente HTTP para consultar la API SODA de datos.gov.co (SECOP).

Versión ligera del cliente del MCP server, sin dependencia de MCP.
Solo usa httpx para hacer las consultas directamente a Socrata.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

BASE_URL = "https://www.datos.gov.co/resource"
_TIMEOUT = 45.0
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

DATASETS = {
    "secop1_procesos": {
        "id": "f789-7hwg",
        "nombre": "SECOP I - Procesos de Compra Publica",
    },
    "secop2_procesos": {
        "id": "p6dx-8zbt",
        "nombre": "SECOP II - Procesos de Contratacion",
    },
    "secop2_contratos": {
        "id": "jbjy-vk9h",
        "nombre": "SECOP II - Contratos Electronicos",
    },
    "secop2_proveedores": {
        "id": "qmzu-gj57",
        "nombre": "SECOP II - Proveedores Registrados",
    },
}


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    return headers


def _get_endpoint(dataset_key: str) -> str:
    return f"{BASE_URL}/{DATASETS[dataset_key]['id']}.json"


async def query_dataset(
    dataset_key: str,
    where: str | None = None,
    select: str | None = None,
    order: str | None = None,
    limit: int = 50,
    offset: int = 0,
    q: str | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    url = _get_endpoint(dataset_key)
    params: dict[str, str] = {"$limit": str(min(limit, 1000))}
    if offset > 0:
        params["$offset"] = str(offset)
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if order:
        params["$order"] = order
    if q:
        params["$q"] = q
    if group:
        params["$group"] = group

    last_exception: Exception | None = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.get(url, params=params, headers=_headers())
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException as exc:
                last_exception = exc
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))
            except httpx.HTTPStatusError:
                raise

    if last_exception:
        raise last_exception
    resp.raise_for_status()
    return resp.json()


def build_where(filters: dict[str, str | float | None]) -> str | None:
    clauses: list[str] = []
    for field, value in filters.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, (int, float)):
            clauses.append(f"{field} >= {value}")
        else:
            safe = value.replace("'", "''")
            clauses.append(f"upper({field}) like upper('%{safe}%')")
    return " AND ".join(clauses) if clauses else None


def add_date_filter(where: str | None, field: str, desde: str, hasta: str) -> str | None:
    clauses = [where] if where else []
    if desde:
        clauses.append(f"{field} >= '{desde}T00:00:00.000'")
    if hasta:
        clauses.append(f"{field} <= '{hasta}T23:59:59.999'")
    return " AND ".join(clauses) if clauses else None


def extract_url(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        return value.get("url", "")
    return str(value)
