"""
Profit Optimizer backend helpers.

This provides the /predict-profit endpoint response expected by the UI:
total_cost, revenue, profit, and a crop suggestion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = os.path.join("app", "Data", "profit_optimizer_dataset.csv")

# Required crops for the advanced dataset spec.
CROPS = ["wheat", "rice", "maize", "tomato", "potato", "onion", "chickpea", "lentil", "soybean", "cotton"]

CROP_DISPLAY = {
    "wheat": "Wheat",
    "rice": "Rice",
    "maize": "Maize",
    "tomato": "Tomato",
    "potato": "Potato",
    "onion": "Onion",
    "chickpea": "Chickpea",
    "lentil": "Lentil",
    "soybean": "Soybean",
    "cotton": "Cotton",
}


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp_min(x: float, lo: float) -> float:
    return x if x >= lo else lo


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        # UI can still work without the advanced dataset.
        return pd.DataFrame(columns=["crop", "yield", "market_price", "rainfall", "temperature"])
    df = pd.read_csv(DATASET_PATH)
    # Normalize crop codes for consistent model input.
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    return df


@dataclass(frozen=True)
class ProfitModels:
    yield_lr: object
    yield_rf: object
    price_lr: object
    price_rf: object
    avg_yield_by_crop: Dict[str, float]
    avg_price_by_crop: Dict[str, float]


@lru_cache(maxsize=1)
def train_models() -> ProfitModels:
    df = load_dataset()
    # If dataset is missing, fall back to basic averages.
    if df.empty or not {"crop", "yield", "market_price", "rainfall", "temperature"}.issubset(df.columns):
        empty_avg = {c: 0.0 for c in CROPS}
        dummy = ProfitModels(
            yield_lr=LinearRegression(),
            yield_rf=RandomForestRegressor(),
            price_lr=LinearRegression(),
            price_rf=RandomForestRegressor(),
            avg_yield_by_crop=empty_avg,
            avg_price_by_crop=empty_avg,
        )
        return dummy

    df = df[df["crop"].isin(CROPS)].copy()
    avg_yield_by_crop = df.groupby("crop")["yield"].mean().to_dict()
    avg_price_by_crop = df.groupby("crop")["market_price"].mean().to_dict()
    for c in CROPS:
        avg_yield_by_crop.setdefault(c, 0.0)
        avg_price_by_crop.setdefault(c, 0.0)

    X = df[["crop", "rainfall", "temperature"]]

    yield_y = df["yield"].astype(float)
    price_y = df["market_price"].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            # scikit-learn version compatibility:
            # - older versions used `sparse=...`
            # - newer versions use `sparse_output=...`
            (
                "crop",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["crop"],
            ),
        ],
        remainder="passthrough",
    )

    yield_lr = Pipeline(steps=[("prep", preprocessor), ("model", LinearRegression())])
    yield_rf = Pipeline(
        steps=[("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=140, random_state=42))]
    )
    price_lr = Pipeline(steps=[("prep", preprocessor), ("model", LinearRegression())])
    price_rf = Pipeline(
        steps=[("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=140, random_state=42))]
    )

    # Train models.
    yield_lr.fit(X, yield_y)
    yield_rf.fit(X, yield_y)
    price_lr.fit(X, price_y)
    price_rf.fit(X, price_y)

    return ProfitModels(
        yield_lr=yield_lr,
        yield_rf=yield_rf,
        price_lr=price_lr,
        price_rf=price_rf,
        avg_yield_by_crop=avg_yield_by_crop,
        avg_price_by_crop=avg_price_by_crop,
    )


def predict_yield_and_future_price_ml(crop: str, rainfall_mm: float, temperature_c: float) -> Tuple[float, float]:
    """
    Predict expected yield (kg/acre) and future market price (INR/kg).
    """
    crop = (crop or "").strip().lower()
    if crop not in CROPS:
        crop = CROPS[0]

    models = train_models()
    # If dataset was missing, averages are the only available signal.
    if models.avg_yield_by_crop.get(crop, 0.0) == 0.0 and models.avg_price_by_crop.get(crop, 0.0) == 0.0:
        return 0.0, 0.0

    X_pred = pd.DataFrame([{"crop": crop, "rainfall": rainfall_mm, "temperature": temperature_c}])

    y_pred_lr = _safe_float(models.yield_lr.predict(X_pred)[0], default=models.avg_yield_by_crop.get(crop, 0.0))
    y_pred_rf = _safe_float(models.yield_rf.predict(X_pred)[0], default=models.avg_yield_by_crop.get(crop, 0.0))
    yield_pred = 0.5 * (y_pred_lr + y_pred_rf)

    p_pred_lr = _safe_float(
        models.price_lr.predict(X_pred)[0], default=models.avg_price_by_crop.get(crop, 0.0)
    )
    p_pred_rf = _safe_float(
        models.price_rf.predict(X_pred)[0], default=models.avg_price_by_crop.get(crop, 0.0)
    )
    price_pred = 0.5 * (p_pred_lr + p_pred_rf)

    return _clamp_min(yield_pred, 0.0), _clamp_min(price_pred, 0.0)


def calc_profit(
    *,
    land_area_acres: float,
    expected_yield_kg_per_acre: float,
    market_price_inr_per_kg: float,
    cost_seeds: float,
    cost_fertilizer: float,
    cost_labor: float,
    cost_irrigation: float,
) -> Dict[str, float]:
    land_area_acres = _clamp_min(land_area_acres, 0.0)
    expected_yield_kg_per_acre = _clamp_min(expected_yield_kg_per_acre, 0.0)
    market_price_inr_per_kg = _clamp_min(market_price_inr_per_kg, 0.0)

    total_cost = cost_seeds + cost_fertilizer + cost_labor + cost_irrigation
    revenue = expected_yield_kg_per_acre * land_area_acres * market_price_inr_per_kg
    profit = revenue - total_cost
    return {
        "total_cost": float(total_cost),
        "revenue": float(revenue),
        "profit": float(profit),
    }


def _avg_yield_and_price(crop: str, *, models: ProfitModels) -> Tuple[float, float]:
    return float(models.avg_yield_by_crop.get(crop, 0.0)), float(models.avg_price_by_crop.get(crop, 0.0))


def suggest_crop_for_better_profit(
    crop_1: str,
    crop_2: Optional[str],
    *,
    land_area_acres: float,
    cost_seeds: float,
    cost_fertilizer: float,
    cost_labor: float,
    cost_irrigation: float,
    expected_yield_1: float,
    market_price_1: float,
    expected_yield_2: Optional[float],
    market_price_2: Optional[float],
    rainfall_mm: float,
    temperature_c: float,
    use_ai: bool,
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Returns (suggestion_string, optional_crop_2_result).
    """
    crop_1 = (crop_1 or "").strip().lower()
    crop_2 = (crop_2 or "").strip().lower() if crop_2 else None

    models = train_models()

    def profit_for(c: str, yield_val: float, price_val: float) -> Dict[str, float]:
        return calc_profit(
            land_area_acres=land_area_acres,
            expected_yield_kg_per_acre=yield_val,
            market_price_inr_per_kg=price_val,
            cost_seeds=cost_seeds,
            cost_fertilizer=cost_fertilizer,
            cost_labor=cost_labor,
            cost_irrigation=cost_irrigation,
        )

    crop1_profit_used: Dict[str, float]
    if use_ai:
        y1_ml, p1_ml = predict_yield_and_future_price_ml(crop_1, rainfall_mm, temperature_c)
        crop1_profit_used = profit_for(crop_1, y1_ml, p1_ml)
    else:
        crop1_profit_used = profit_for(crop_1, expected_yield_1, market_price_1)

    if crop_2:
        if crop_2 not in CROPS:
            crop_2 = None

    crop2_profit_used: Optional[Dict[str, float]] = None
    if crop_2:
        if use_ai:
            y2_ml, p2_ml = predict_yield_and_future_price_ml(crop_2, rainfall_mm, temperature_c)
            crop2_profit_used = profit_for(crop_2, y2_ml, p2_ml)
        else:
            if expected_yield_2 is None or market_price_2 is None:
                crop2_profit_used = None
            else:
                crop2_profit_used = profit_for(crop_2, expected_yield_2, market_price_2)

    # If crop2 was provided, compute suggestion based on computed crop1 and crop2 profits.
    if crop_2 and crop2_profit_used is not None:
        profit1 = crop1_profit_used["profit"]
        profit2 = crop2_profit_used["profit"]
        if profit2 > profit1:
            suggestion = f"Grow {CROP_DISPLAY.get(crop_2, crop_2)} instead of {CROP_DISPLAY.get(crop_1, crop_1)} for better profit"
        elif profit1 > profit2:
            suggestion = f"Grow {CROP_DISPLAY.get(crop_1, crop_1)} instead of {CROP_DISPLAY.get(crop_2, crop_2)} for better profit"
        else:
            suggestion = f"Both {CROP_DISPLAY.get(crop_1, crop_1)} and {CROP_DISPLAY.get(crop_2, crop_2)} have similar profitability"
        return suggestion, crop2_profit_used

    # No crop2: find best alternative crop using predictions (AI mode) or averages.
    best_crop = None
    best_profit = float("-inf")

    for c in CROPS:
        if c == crop_1:
            continue
        if use_ai:
            y_pred, p_pred = predict_yield_and_future_price_ml(c, rainfall_mm, temperature_c)
        else:
            y_pred, p_pred = _avg_yield_and_price(c, models=models)

        prof = profit_for(c, y_pred, p_pred)
        if prof["profit"] > best_profit:
            best_profit = prof["profit"]
            best_crop = c

    # If we couldn't find anything (e.g., missing dataset), provide a generic suggestion.
    if not best_crop or best_crop not in CROPS:
        return f"Improve your inputs (yield, price, and costs) to increase profit", None

    suggestion = f"Grow {CROP_DISPLAY.get(best_crop, best_crop)} instead of {CROP_DISPLAY.get(crop_1, crop_1)} for better profit"
    return suggestion, crop2_profit_used

