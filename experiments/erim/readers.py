"""Strict fixed-width readers used by the current canonical ERIM builder."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ERIM_PURCHASE_WIDTHS = [2, 2, 6, 1, 1, 8, 13, 4, 1, 5, 7, 7, 3, 3, 5,
                        3, 5, 11, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]

ERIM_PURCHASE_COLUMNS = [
    "cell", "market", "week", "dow", "trip", "household", "upc", "store",
    "store_type", "units_raw", "expenditure_raw", "weight_raw", "special_coupon_code",
    "store_coupon_units", "store_coupon_value_raw", "manufacturer_coupon_units",
    "manufacturer_coupon_value_raw", "equivalence_raw", "line_number", "end_display",
    "front_display", "aisle_display", "other_display", "advertised", "ad_type",
    "in_ad_coupon", "point_of_purchase", "special_price", "coupon_factor", "filler",
]

ERIM_SHOPPING_WIDTHS = [2, 4, 1, 8, 6, 7]

ERIM_SHOPPING_COLUMNS = ["market", "store", "store_type", "household", "week", "spend_raw"]

ERIM_RETAIL_WIDTHS = [2, 6, 4, 13, 5, 9, 7, 7, 5, 11, 4, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 2]

ERIM_RETAIL_COLUMNS = [
    "market", "week", "store", "upc", "units", "revenue_raw", "weight_raw",
    "store_coupon_value_raw", "store_coupon_units", "equivalence_raw", "line_number",
    "end_display", "front_display", "aisle_display", "other_display", "advertised",
    "ad_type", "in_ad_coupon", "point_of_purchase", "special_price", "filler",
]

ERIM_PRODUCT_WIDTHS = [13, 30, 2, 6, 7, 3, 11, 4]

ERIM_PRODUCT_COLUMNS = [
    "upc", "description", "weight_code", "weight_description", "weight_amount",
    "multipack", "equivalence_factor", "line_number",
]

def _read_fixed(path: Path, widths: list[int], columns: list[str],
                numeric_columns: list[str]) -> pd.DataFrame:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = sum(widths)
    with path.open("rb") as stream:
        first = stream.readline().rstrip(b"\r\n")
    if len(first) != expected:
        raise ValueError(f"{path.name} has record length {len(first)}, expected {expected}")
    frame = pd.read_fwf(path, widths=widths, names=columns, dtype=str,
                        keep_default_na=False)
    for column in columns:
        frame[column] = frame[column].str.strip()
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            example = frame.loc[converted.isna(), column].iloc[0]
            raise ValueError(f"invalid numeric field {column}={example!r} in {path.name}")
        frame[column] = converted.astype(np.int64)
    return frame

def read_erim_purchase(path: Path) -> pd.DataFrame:
    numeric = ERIM_PURCHASE_COLUMNS[:19]
    frame = _read_fixed(path, ERIM_PURCHASE_WIDTHS, ERIM_PURCHASE_COLUMNS, numeric)
    frame["units"] = frame.units_raw / 100.0
    frame["expenditure"] = frame.expenditure_raw / 100.0
    frame["unit_price"] = frame.expenditure / frame.units
    return frame

def read_erim_shopping(path: Path) -> pd.DataFrame:
    frame = _read_fixed(path, ERIM_SHOPPING_WIDTHS, ERIM_SHOPPING_COLUMNS,
                        ERIM_SHOPPING_COLUMNS)
    frame["spend"] = frame.spend_raw / 100.0
    return frame

def read_erim_retail(path: Path, *, allowed_weeks: set[int] | None = None) -> pd.DataFrame:
    numeric = ERIM_RETAIL_COLUMNS[:11]
    frame = _read_fixed(path, ERIM_RETAIL_WIDTHS, ERIM_RETAIL_COLUMNS, numeric)
    if allowed_weeks is not None:
        frame = frame[frame.week.isin(allowed_weeks)].copy()
        if frame.empty:
            raise ValueError("ERIM retail data have no rows in the requested weeks")
    if bool((frame.units <= 0).any()):
        raise ValueError("ERIM retail prices require positive aggregate units")
    frame["revenue"] = frame.revenue_raw / 100.0
    frame["average_price"] = frame.revenue / frame.units
    if bool((frame.average_price <= 0).any()):
        raise ValueError("ERIM retail average prices must be positive")
    return frame

def read_erim_products(path: Path) -> dict[int, str]:
    """Read the official ERIM UPC stub and return human-readable labels."""
    frame = _read_fixed(path, ERIM_PRODUCT_WIDTHS, ERIM_PRODUCT_COLUMNS,
                        ["upc", "weight_amount", "multipack", "equivalence_factor",
                         "line_number"])
    if frame.upc.duplicated().any():
        raise ValueError("ERIM product UPCs are not unique")
    labels = {}
    for row in frame.itertuples(index=False):
        description = " ".join(str(row.description).split())
        size = " ".join(str(row.weight_description).split())
        label = description if not size else f"{description} ({size})"
        labels[int(row.upc)] = label
    return labels
