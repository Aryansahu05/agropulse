"""
Profit Optimization utilities.

This module is intentionally lightweight and self-contained so the UI can call
the Flask API and still get meaningful results even when ML models are not
available (e.g., `crop_model.pkl` missing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# Expected profit dataset (values are approximate for demo/research).
# Units:
# - average_yield: kg/hectare
# - market_price: INR/kg
# - cost_of_cultivation: INR/hectare
CROP_PROFIT_DATA: Dict[str, Dict[str, float]] = {
    # Cereals
    "wheat": {"average_yield": 3000, "market_price": 22, "cost_of_cultivation": 40000},
    "rice": {"average_yield": 3500, "market_price": 18, "cost_of_cultivation": 45000},
    "maize": {"average_yield": 2800, "market_price": 25, "cost_of_cultivation": 35000},
    "barley": {"average_yield": 2500, "market_price": 20, "cost_of_cultivation": 30000},
    "millet": {"average_yield": 2000, "market_price": 28, "cost_of_cultivation": 25000},
    # Pulses
    "chickpea": {"average_yield": 1200, "market_price": 60, "cost_of_cultivation": 30000},
    "lentil": {"average_yield": 900, "market_price": 65, "cost_of_cultivation": 28000},
    "pigeon_pea": {"average_yield": 1400, "market_price": 55, "cost_of_cultivation": 31000},
    "green_gram": {"average_yield": 800, "market_price": 90, "cost_of_cultivation": 32000},
    "black_gram": {"average_yield": 700, "market_price": 95, "cost_of_cultivation": 33000},
    # Vegetables
    "tomato": {"average_yield": 60000, "market_price": 9, "cost_of_cultivation": 50000},
    "potato": {"average_yield": 25000, "market_price": 18, "cost_of_cultivation": 45000},
    "onion": {"average_yield": 20000, "market_price": 25, "cost_of_cultivation": 40000},
    "cabbage": {"average_yield": 30000, "market_price": 10, "cost_of_cultivation": 45000},
    "cauliflower": {"average_yield": 28000, "market_price": 12, "cost_of_cultivation": 48000},
    "spinach": {"average_yield": 8000, "market_price": 30, "cost_of_cultivation": 25000},
    "carrot": {"average_yield": 25000, "market_price": 15, "cost_of_cultivation": 38000},
    "brinjal": {"average_yield": 12000, "market_price": 20, "cost_of_cultivation": 60000},
    "capsicum": {"average_yield": 8000, "market_price": 45, "cost_of_cultivation": 70000},
    # Fruits
    "mango": {"average_yield": 4000, "market_price": 60, "cost_of_cultivation": 80000},
    "banana": {"average_yield": 35000, "market_price": 18, "cost_of_cultivation": 70000},
    "apple": {"average_yield": 12000, "market_price": 80, "cost_of_cultivation": 150000},
    "orange": {"average_yield": 10000, "market_price": 55, "cost_of_cultivation": 120000},
    "papaya": {"average_yield": 20000, "market_price": 20, "cost_of_cultivation": 60000},
    "pomegranate": {"average_yield": 6000, "market_price": 100, "cost_of_cultivation": 90000},
    "grapes": {"average_yield": 7000, "market_price": 110, "cost_of_cultivation": 90000},
    # Melons
    "watermelon": {"average_yield": 40000, "market_price": 8, "cost_of_cultivation": 45000},
    "muskmelon": {"average_yield": 20000, "market_price": 12, "cost_of_cultivation": 35000},
    # Oil crops
    "mustard": {"average_yield": 1400, "market_price": 55, "cost_of_cultivation": 30000},
    "groundnut": {"average_yield": 2200, "market_price": 60, "cost_of_cultivation": 35000},
    "soybean": {"average_yield": 2500, "market_price": 45, "cost_of_cultivation": 40000},
    "sunflower": {"average_yield": 2000, "market_price": 50, "cost_of_cultivation": 38000},
}


# Crop growth preferences (approx) used for heuristic "suitability" and
# explanation. This lets the feature work even without crop ML models.
CROP_GROWTH_PROFILE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "wheat": {"temp": (10, 25), "rain": (200, 600), "ph": (6.0, 7.5)},
    "rice": {"temp": (20, 35), "rain": (800, 2000), "ph": (5.5, 7.5)},
    "maize": {"temp": (18, 32), "rain": (500, 1200), "ph": (5.8, 7.0)},
    "barley": {"temp": (8, 24), "rain": (250, 700), "ph": (6.0, 7.5)},
    "millet": {"temp": (20, 35), "rain": (100, 600), "ph": (5.5, 7.5)},
    "chickpea": {"temp": (15, 27), "rain": (200, 600), "ph": (6.0, 7.5)},
    "lentil": {"temp": (13, 25), "rain": (200, 600), "ph": (6.0, 8.0)},
    "pigeon_pea": {"temp": (20, 30), "rain": (500, 1200), "ph": (5.5, 7.5)},
    "green_gram": {"temp": (20, 35), "rain": (400, 1000), "ph": (6.0, 7.5)},
    "black_gram": {"temp": (20, 35), "rain": (400, 1000), "ph": (6.0, 7.5)},
    "tomato": {"temp": (18, 30), "rain": (500, 900), "ph": (6.0, 7.0)},
    "potato": {"temp": (15, 22), "rain": (300, 800), "ph": (5.0, 6.5)},
    "onion": {"temp": (13, 28), "rain": (200, 800), "ph": (6.0, 7.5)},
    "cabbage": {"temp": (15, 25), "rain": (600, 1200), "ph": (6.0, 7.5)},
    "cauliflower": {"temp": (14, 22), "rain": (500, 1100), "ph": (6.0, 7.5)},
    "spinach": {"temp": (10, 25), "rain": (400, 1200), "ph": (6.0, 7.5)},
    "carrot": {"temp": (15, 22), "rain": (300, 800), "ph": (6.0, 7.0)},
    "brinjal": {"temp": (20, 30), "rain": (600, 1000), "ph": (5.5, 7.0)},
    "capsicum": {"temp": (20, 30), "rain": (500, 900), "ph": (5.8, 6.8)},
    "mango": {"temp": (24, 35), "rain": (500, 1500), "ph": (5.5, 7.5)},
    "banana": {"temp": (22, 32), "rain": (1000, 2000), "ph": (5.5, 7.0)},
    "apple": {"temp": (0, 25), "rain": (500, 1200), "ph": (5.5, 7.0)},
    "orange": {"temp": (18, 30), "rain": (800, 1500), "ph": (5.5, 7.0)},
    "papaya": {"temp": (24, 32), "rain": (1000, 2000), "ph": (5.5, 7.0)},
    "pomegranate": {"temp": (18, 32), "rain": (200, 700), "ph": (5.5, 7.5)},
    "grapes": {"temp": (15, 30), "rain": (300, 900), "ph": (5.5, 7.5)},
    "watermelon": {"temp": (20, 35), "rain": (200, 600), "ph": (6.0, 7.5)},
    "muskmelon": {"temp": (18, 34), "rain": (200, 600), "ph": (6.0, 7.5)},
    "mustard": {"temp": (15, 30), "rain": (300, 700), "ph": (5.5, 7.5)},
    "groundnut": {"temp": (22, 35), "rain": (400, 900), "ph": (5.5, 7.5)},
    "soybean": {"temp": (20, 35), "rain": (400, 900), "ph": (6.0, 7.5)},
    "sunflower": {"temp": (20, 35), "rain": (200, 800), "ph": (6.0, 7.5)},
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _range_score(value: float, lo: float, hi: float, soft_margin: float) -> float:
    """
    Returns a score in [0, 1].

    - 1 if inside [lo, hi]
    - decreases smoothly outside the range up to ~soft_margin
    """
    if lo > hi:
        lo, hi = hi, lo

    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return _clamp(1.0 - (lo - value) / soft_margin, 0.0, 1.0)
    return _clamp(1.0 - (value - hi) / soft_margin, 0.0, 1.0)


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def nutrient_score(soil_data: Dict[str, object]) -> float:
    """
    Heuristic score for soil fertility based on N/P/K.

    Inputs may be missing; in that case we return a neutral score (0.5).
    """
    n = _safe_float(soil_data.get("nitrogen", None), default=0.0)
    p = _safe_float(soil_data.get("phosphorous", None), default=0.0)
    k = _safe_float(soil_data.get("pottasium", None), default=0.0)

    # Neutral if values are not provided.
    if n <= 0 and p <= 0 and k <= 0:
        return 0.5

    # Approx "good ranges" for many cultivated crops.
    # (The actual model is more complex; this is a stable heuristic.)
    n_score = _range_score(n, 50, 150, soft_margin=80)
    p_score = _range_score(p, 20, 60, soft_margin=40)
    k_score = _range_score(k, 20, 150, soft_margin=80)
    return (0.4 * n_score + 0.3 * p_score + 0.3 * k_score)


def suitability_for_crop(
    crop: str,
    soil_data: Dict[str, object],
    weather_data: Dict[str, object],
) -> Dict[str, float]:
    """
    Returns component scores and overall suitability in [0, 1].
    """
    profile = CROP_GROWTH_PROFILE.get(crop)
    if not profile:
        # Unknown crop: neutral suitability.
        return {"temp_score": 0.5, "rain_score": 0.5, "ph_score": 0.5, "soil_score": 0.5, "overall": 0.5}

    temp = _safe_float(weather_data.get("temperature", None), default=0.0)
    rain = _safe_float(weather_data.get("rainfall", None), default=0.0)
    ph = _safe_float(soil_data.get("ph", None), default=0.0)

    temp_lo, temp_hi = profile["temp"]
    rain_lo, rain_hi = profile["rain"]
    ph_lo, ph_hi = profile["ph"]

    temp_score = _range_score(temp, temp_lo, temp_hi, soft_margin=max(5.0, (temp_hi - temp_lo) / 2))
    rain_score = _range_score(rain, rain_lo, rain_hi, soft_margin=max(200.0, (rain_hi - rain_lo) / 2))
    ph_score = _range_score(ph, ph_lo, ph_hi, soft_margin=1.5)
    soil_score = nutrient_score(soil_data)

    overall = 0.4 * temp_score + 0.3 * rain_score + 0.2 * ph_score + 0.1 * soil_score
    return {
        "temp_score": temp_score,
        "rain_score": rain_score,
        "ph_score": ph_score,
        "soil_score": soil_score,
        "overall": overall,
    }


def expected_yield_kg_per_ha(
    crop: str, soil_data: Dict[str, object], weather_data: Dict[str, object]
) -> Tuple[float, Dict[str, float]]:
    """
    Estimates expected yield (kg/ha) using suitability as a scaling factor.
    """
    base = CROP_PROFIT_DATA.get(crop, {}).get("average_yield", 2000.0)
    sat = suitability_for_crop(crop, soil_data, weather_data)
    overall = sat["overall"]

    # Map suitability [0,1] -> yield scale [0.65, 1.35]
    yield_scale = 0.65 + 0.7 * overall
    return base * yield_scale, sat


def compute_profit_inr_per_ha(crop: str, expected_yield_kg_ha: float) -> Dict[str, float]:
    data = CROP_PROFIT_DATA.get(crop)
    if not data:
        # Neutral defaults for unknown crops.
        price = 30.0
        cost = 30000.0
    else:
        price = float(data["market_price"])
        cost = float(data["cost_of_cultivation"])

    revenue = expected_yield_kg_ha * price
    profit = revenue - cost
    return {"market_price": price, "cost": cost, "revenue": revenue, "profit": profit}


def heuristic_top3_crops(
    soil_data: Dict[str, object],
    weather_data: Dict[str, object],
    crops: Optional[Iterable[str]] = None,
) -> List[Dict[str, float]]:
    """
    Produces top-3 "predicted" crops using suitability score only.

    Output items:
    - crop (str)
    - probability (float in [0,1] normalized among returned candidates)
    - suitability (float in [0,1])  (extra field)
    """
    candidates = list(crops) if crops is not None else list(CROP_PROFIT_DATA.keys())
    scored: List[Tuple[str, float]] = []
    for crop in candidates:
        sat = suitability_for_crop(crop, soil_data, weather_data)["overall"]
        scored.append((crop, sat))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:3]
    if not top:
        return []

    # Normalize to [0,1] for UI consistency.
    max_sat = max(s for _, s in top) if top else 1.0
    min_sat = min(s for _, s in top) if top else 0.0
    denom = max(1e-6, max_sat - min_sat)
    normalized = []
    for crop, sat in top:
        prob = (sat - min_sat) / denom if denom > 0 else 1.0
        normalized.append({"crop": crop, "probability": prob, "suitability": sat})
    return normalized


def rank_profit_table(
    predicted_crops: List[Dict[str, float]],
    soil_data: Dict[str, object],
    weather_data: Dict[str, object],
) -> Tuple[List[Dict[str, float]], str, float, Dict[str, float]]:
    """
    For each predicted crop, compute expected yield + profit,
    then rank by profit.

    Returns:
    - profit_table (top candidates sorted by profit)
    - best_crop
    - best_profit
    - best_crop_sat_summaries (for explanation)
    """
    rows = []
    for item in predicted_crops:
        crop = item["crop"]
        expected_yield, sat = expected_yield_kg_per_ha(crop, soil_data, weather_data)
        profit_info = compute_profit_inr_per_ha(crop, expected_yield)
        rows.append(
            {
                "crop": crop,
                "expected_yield_kg_ha": expected_yield,
                "profit": profit_info["profit"],
                "market_price": profit_info["market_price"],
                "cost": profit_info["cost"],
                "suitability": sat["overall"],
                "sat_breakdown": sat,
            }
        )

    rows.sort(key=lambda r: r["profit"], reverse=True)
    best = rows[0] if rows else None
    if not best:
        return [], "", 0.0, {}
    return rows, best["crop"], float(best["profit"]), best["sat_breakdown"]


def why_this_crop_is_profitable(
    recommended_crop: str,
    profit_rows: List[Dict[str, float]],
    soil_data: Dict[str, object],
    weather_data: Dict[str, object],
) -> List[str]:
    """
    Bonus: returns research-friendly reasons for the selected crop.
    """
    rec_row = None
    for r in profit_rows:
        if r.get("crop") == recommended_crop:
            rec_row = r
            break
    if rec_row is None:
        return ["High expected profitability based on current soil and weather conditions."]

    prices = [r.get("market_price", 0.0) for r in profit_rows]
    costs = [r.get("cost", 0.0) for r in profit_rows]
    max_price = max(prices) if prices else 1.0
    min_cost = min(costs) if costs else 0.0

    reasons: List[str] = []
    price = float(rec_row["market_price"])
    cost = float(rec_row["cost"])
    expected_yield = float(rec_row["expected_yield_kg_ha"])

    # Market price reason.
    if max_price > 0 and abs(price - max_price) / max_price < 0.05:
        reasons.append(f"High market price: ₹{price:.0f}/kg (one of the best in this shortlist).")
    else:
        reasons.append(f"Market price is strong: ₹{price:.0f}/kg.")

    # Cost reason.
    if cost <= (min_cost + 0.05 * max(1.0, min_cost)):
        reasons.append(f"Low cultivation cost: ₹{cost:,.0f}/ha (helps protect margins).")
    else:
        reasons.append(f"Competitive cultivation cost: ₹{cost:,.0f}/ha.")

    # Suitability reason.
    sat = rec_row.get("sat_breakdown", {}) or {}
    temp_score = float(sat.get("temp_score", 0.5))
    rain_score = float(sat.get("rain_score", 0.5))
    ph_score = float(sat.get("ph_score", 0.5))

    best_factor = max(
        ("Temperature", temp_score),
        ("Rainfall", rain_score),
        ("Soil pH", ph_score),
        key=lambda x: x[1],
    )[0]
    reasons.append(
        "Suitable in current conditions: "
        f"expected yield ~{expected_yield:,.0f} kg/ha "
        f"(best match via {best_factor})."
    )

    return reasons

