"""
BioMirror — Indian Food Glycemic Database
==========================================
60 common Indian foods with clinically validated:
  - Glycemic Index (GI)
  - Net carbs per 100g
  - Protein per 100g
  - Fiber per 100g
  - Typical portion size
  - Glucose impact class

Sources:
  - International Tables of Glycemic Index (Foster-Powell et al.)
  - Indian Food Composition Tables (NIN, Hyderabad)
  - Journal of Nutrition and Metabolism — Indian Diet Studies 2022

Usage:
    from food_database import get_food, search_food, predict_meal_glucose
    food = get_food('idli')
    glucose_rise = predict_meal_glucose('dosa', 150, current_glucose=160, p3_index=65)
"""

import math
import re
from typing import Optional

# ─────────────────────────────────────────────
# FOOD DATABASE
# Schema: name → (GI, carbs/100g, protein/100g, fiber/100g,
#                 typical_portion_g, category)
# GI: 0.0–1.0 (0=no effect, 1=pure glucose)
# ─────────────────────────────────────────────
FOOD_DB = {
    # ── RICE & GRAINS ────────────────────────────────────────────────
    'white_rice':        (0.72, 28.2, 2.7, 0.4,  150, 'grain'),
    'brown_rice':        (0.55, 23.5, 2.6, 1.8,  150, 'grain'),
    'basmati_rice':      (0.57, 26.0, 3.5, 0.5,  150, 'grain'),
    'red_rice':          (0.50, 24.0, 2.8, 2.5,  150, 'grain'),
    'millet_bajra':      (0.54, 42.0, 11.6, 1.2, 80,  'grain'),
    'jowar_sorghum':     (0.55, 72.0, 10.4, 6.7, 80,  'grain'),
    'ragi_finger_millet':(0.65, 72.0, 7.3,  3.6, 80,  'grain'),
    'quinoa':            (0.53, 22.0, 4.1,  2.8, 100, 'grain'),
    'oats':              (0.55, 27.0, 17.0, 10.0, 80,  'grain'),
    'poha':              (0.55, 21.0, 1.8,  0.6, 150, 'grain'),

    # ── BREADS & ROTIS ────────────────────────────────────────────────
    'roti_wheat':        (0.62, 47.0, 8.5,  1.9, 30,  'bread'),
    'roti_multigrain':   (0.48, 38.0, 9.5,  4.5, 30,  'bread'),
    'naan':              (0.71, 49.0, 9.0,  2.0, 90,  'bread'),
    'paratha_plain':     (0.66, 42.0, 7.2,  1.5, 80,  'bread'),
    'paratha_stuffed':   (0.58, 36.0, 6.8,  2.2, 100, 'bread'),
    'puri':              (0.74, 44.0, 7.0,  1.5, 50,  'bread'),
    'white_bread':       (0.70, 49.0, 9.0,  2.7, 60,  'bread'),
    'brown_bread':       (0.50, 41.0, 8.5,  4.0, 60,  'bread'),

    # ── BREAKFAST ITEMS ───────────────────────────────────────────────
    'idli':              (0.65, 22.0, 2.1, 0.5,  150, 'breakfast'),
    'dosa':              (0.68, 21.0, 3.0, 0.5,  100, 'breakfast'),
    'uttapam':           (0.62, 20.0, 4.2, 1.2,  150, 'breakfast'),
    'upma':              (0.65, 20.0, 3.5, 1.5,  150, 'breakfast'),
    'pongal':            (0.58, 18.0, 3.0, 1.0,  150, 'breakfast'),
    'aloo_paratha':      (0.65, 38.0, 6.5, 2.8,  120, 'breakfast'),
    'masala_dosa':       (0.65, 25.0, 4.5, 1.8,  200, 'breakfast'),
    'poori_bhaji':       (0.72, 35.0, 5.8, 2.5,  200, 'breakfast'),

    # ── LENTILS & LEGUMES (best for T2D — low GI, high fiber) ────────
    'dal_toor':          (0.22, 20.0, 7.8, 4.5,  200, 'legume'),
    'dal_moong':         (0.25, 18.0, 7.5, 6.8,  200, 'legume'),
    'dal_masoor':        (0.21, 20.0, 9.0, 5.5,  200, 'legume'),
    'rajma':             (0.24, 20.0, 8.7, 6.4,  150, 'legume'),
    'chana_chickpea':    (0.28, 27.0, 8.9, 7.6,  150, 'legume'),
    'black_chana':       (0.30, 25.0, 8.5, 8.0,  150, 'legume'),
    'green_moong':       (0.25, 19.0, 7.0, 7.6,  150, 'legume'),
    'sambhar':           (0.28, 12.0, 4.2, 3.5,  250, 'legume'),

    # ── VEGETABLES ────────────────────────────────────────────────────
    'potato':            (0.78, 20.0, 2.0, 1.8,  150, 'vegetable'),
    'sweet_potato':      (0.54, 20.0, 1.6, 3.0,  150, 'vegetable'),
    'beetroot':          (0.64, 10.0, 1.7, 2.8,  100, 'vegetable'),
    'carrot':            (0.35, 10.0, 0.9, 2.8,  100, 'vegetable'),
    'spinach':           (0.15, 3.6,  2.9, 2.2,  100, 'vegetable'),
    'bitter_gourd':      (0.12, 3.7,  1.5, 2.8,  100, 'vegetable'),
    'bottle_gourd':      (0.15, 2.5,  0.6, 1.8,  150, 'vegetable'),
    'lady_finger_okra':  (0.20, 7.0,  1.9, 3.2,  100, 'vegetable'),
    'brinjal':           (0.15, 5.7,  1.0, 3.0,  100, 'vegetable'),
    'tomato':            (0.15, 3.9,  0.9, 1.2,  100, 'vegetable'),
    'onion':             (0.30, 9.0,  1.1, 1.7,  50,  'vegetable'),
    'salad_mix':         (0.15, 3.5,  1.5, 2.0,  200, 'vegetable'),

    # ── FRUITS ───────────────────────────────────────────────────────
    'banana':            (0.51, 23.0, 1.1, 2.6,  100, 'fruit'),
    'mango':             (0.56, 15.0, 0.6, 1.6,  120, 'fruit'),
    'apple':             (0.36, 14.0, 0.3, 2.4,  150, 'fruit'),
    'guava':             (0.12, 14.0, 2.6, 5.4,  150, 'fruit'),
    'papaya':            (0.38, 10.0, 0.5, 1.8,  150, 'fruit'),
    'watermelon':        (0.72, 7.6,  0.6, 0.4,  300, 'fruit'),
    'orange':            (0.43, 12.0, 0.9, 2.4,  150, 'fruit'),
    'pomegranate':       (0.35, 19.0, 1.7, 4.0,  100, 'fruit'),

    # ── DAIRY ─────────────────────────────────────────────────────────
    'milk_full_fat':     (0.27, 5.0,  3.4, 0.0,  250, 'dairy'),
    'curd_yogurt':       (0.35, 4.0,  10.0,0.0,  150, 'dairy'),
    'paneer':            (0.00, 1.7,  18.3,0.0,  100, 'dairy'),
    'lassi_sweet':       (0.55, 14.0, 4.5, 0.0,  250, 'dairy'),
    'lassi_plain':       (0.30, 5.0,  5.0, 0.0,  250, 'dairy'),
    'chaas_buttermilk':  (0.20, 3.0,  3.5, 0.0,  250, 'dairy'),

    # ── SNACKS & STREET FOOD ──────────────────────────────────────────
    'samosa':            (0.68, 28.0, 4.5, 2.5,  100, 'snack'),
    'pakoda':            (0.62, 24.0, 5.8, 2.0,  100, 'snack'),
    'vada':              (0.55, 22.0, 8.5, 3.5,  100, 'snack'),
    'murukku':           (0.58, 62.0, 8.0, 3.0,  50,  'snack'),
    'biscuit_plain':     (0.70, 65.0, 6.0, 1.0,  30,  'snack'),
    'chakli':            (0.58, 60.0, 8.5, 4.0,  50,  'snack'),

    # ── NON-VEG ──────────────────────────────────────────────────────
    'egg_boiled':        (0.00, 1.1,  13.0,0.0,  60,  'protein'),
    'chicken_curry':     (0.15, 4.0,  22.0,0.0,  150, 'protein'),
    'fish_curry':        (0.12, 3.0,  20.0,0.0,  150, 'protein'),
    'mutton_curry':      (0.18, 5.0,  19.0,0.0,  150, 'protein'),

    # ── BEVERAGES ─────────────────────────────────────────────────────
    'chai_with_sugar':   (0.65, 8.0,  1.5, 0.0,  150, 'beverage'),
    'chai_no_sugar':     (0.20, 2.0,  1.5, 0.0,  150, 'beverage'),
    'coconut_water':     (0.42, 9.0,  0.7, 1.0,  250, 'beverage'),
    'sugarcane_juice':   (0.85, 27.0, 0.3, 0.0,  200, 'beverage'),
    'lemon_water':       (0.05, 2.0,  0.1, 0.0,  300, 'beverage'),
    'coffee_black':      (0.03, 0.3,  0.3, 0.0,  200, 'beverage'),

    # ── SWEETS (high risk) ────────────────────────────────────────────
    'jalebi':            (0.95, 65.0, 2.0, 0.5,  50,  'sweet'),
    'gulab_jamun':       (0.76, 48.0, 5.5, 0.4,  80,  'sweet'),
    'halwa':             (0.72, 45.0, 4.0, 1.5,  100, 'sweet'),
    'kheer':             (0.65, 22.0, 5.0, 0.3,  150, 'sweet'),
    'laddoo_besan':      (0.56, 52.0, 10.0,3.0,  50,  'sweet'),
    'payasam':           (0.68, 25.0, 4.5, 0.5,  150, 'sweet'),

    # ── INTERNATIONAL COMMON ──────────────────────────────────────────
    'pizza':             (0.60, 33.0, 11.0,2.3,  200, 'international'),
    'pasta':             (0.49, 25.0, 5.0, 1.8,  200, 'international'),
    'burger':            (0.65, 32.0, 14.0,1.5,  180, 'international'),
    'sandwich':          (0.58, 28.0, 9.5, 2.8,  150, 'international'),
}

FOOD_ALIASES = {
    "idly": "idli", "idlies": "idli", "dosai": "dosa",
    "chapati": "roti_wheat", "chapathi": "roti_wheat", "roti": "roti_wheat",
    "rice": "white_rice", "cooked_rice": "white_rice", "bowl_rice": "white_rice",
    "dal": "dal_toor", "lentils": "dal_toor", "sambar": "sambhar",
    "curd": "curd_yogurt", "yogurt": "curd_yogurt", "plain_yogurt": "curd_yogurt",
    "egg": "egg_boiled", "boiled_egg": "egg_boiled", "eggs": "egg_boiled",
    "chicken": "chicken_curry", "fish": "fish_curry", "apple": "apple",
    "banana": "banana", "paneer": "paneer", "oats": "oats",
}

FOOD_SERVINGS = {
    "idli": (50, "piece"), "dosa": (100, "piece"), "roti_wheat": (30, "piece"),
    "white_rice": (150, "bowl"), "brown_rice": (150, "bowl"),
    "basmati_rice": (150, "bowl"), "dal_toor": (200, "bowl"),
    "dal_moong": (200, "bowl"), "dal_masoor": (200, "bowl"),
    "sambhar": (250, "bowl"), "curd_yogurt": (150, "bowl"),
    "banana": (100, "piece"), "apple": (150, "piece"),
    "egg_boiled": (60, "piece"), "paneer": (100, "portion"),
    "chicken_curry": (150, "portion"), "fish_curry": (150, "portion"),
    "vegetable_soup": (250, "bowl"),
}

# Energy and fat values are approximate reference values per 100 g, available
# only for these common foods. Other foods retain their known GI/carbohydrate
# fields and leave unavailable nutrients unset instead of inventing them.
FOOD_ENERGY_FAT_PER_100G = {
    "white_rice": (130, 0.3), "brown_rice": (123, 1.0), "basmati_rice": (121, 0.4),
    "idli": (116, 0.4), "dosa": (168, 4.0), "roti_wheat": (297, 7.0),
    "dal_toor": (116, 2.0), "dal_moong": (105, 1.2), "dal_masoor": (116, 0.4),
    "sambhar": (80, 2.0), "curd_yogurt": (61, 3.3), "banana": (89, 0.3),
    "apple": (52, 0.2), "egg_boiled": (155, 11.0), "paneer": (265, 20.8),
    "chicken_curry": (160, 7.0), "fish_curry": (140, 5.0),
}

FOOD_CATEGORIES = {
    "breakfast": "Indian", "grain": "Indian", "bread": "Indian", "legume": "Indian",
    "vegetable": "Common Foods", "fruit": "Fruits", "dairy": "Common Foods",
    "snack": "Snacks", "protein": "Protein", "beverage": "Beverages",
    "sweet": "Snacks", "international": "Common Foods",
}


def _normalize_food_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def resolve_food_name(value: str) -> Optional[str]:
    """Return a canonical food key only for an exact key or configured alias."""
    key = _normalize_food_name(value)
    key = FOOD_ALIASES.get(key, key)
    return key if key in FOOD_DB else None


def list_foods(query: str = "", category: str = "") -> list:
    """Return structured choices from the shared food database."""
    needle = _normalize_food_name(query)
    category_key = category.strip().lower()
    results = []
    for key, data in FOOD_DB.items():
        gi, carbs, protein, fiber, portion, original_category = data
        aliases = [alias.replace("_", " ") for alias, canonical in FOOD_ALIASES.items()
                   if canonical == key]
        names = [key.replace("_", " "), *aliases]
        if needle and not any(needle in _normalize_food_name(name) for name in names):
            continue
        ui_category = FOOD_CATEGORIES.get(original_category, "Common Foods")
        if category_key and ui_category.lower() != category_key:
            continue
        serving_size, serving_unit = FOOD_SERVINGS.get(key, (portion, "serving"))
        energy_fat = FOOD_ENERGY_FAT_PER_100G.get(key)
        results.append({
            "id": key,
            "name": key.replace("_", " ").title(),
            "aliases": aliases,
            "category": ui_category,
            "servingSize": serving_size,
            "servingUnit": serving_unit,
            "caloriesAvailable": energy_fat is not None,
            "glycemicIndex": round(gi * 100),
            "source": "local_food_database",
        })
    return sorted(results, key=lambda item: item["name"].lower())


def calculate_nutrition(food_name: str, quantity: float = 1,
                        serving_unit: Optional[str] = None) -> dict:
    """Scale available local reference nutrients for a simple serving count.

    Unknown food names return null nutrient values and can still be logged.
    """
    parsed_quantity = float(quantity)
    parsed_name = food_name.strip()
    parsed_unit = (serving_unit or "").strip().lower()
    match = re.match(
        r"^\s*(\d+(?:\.\d+)?)\s+(?:(pieces?|bowls?|plates?|cups?|servings?)\s+)?(.+?)\s*$",
        parsed_name, re.IGNORECASE,
    )
    if match and parsed_quantity == 1:
        natural_food = match.group(3)
        natural_key = resolve_food_name(natural_food)
        if natural_key:
            parsed_quantity = float(match.group(1))
            parsed_name = natural_food
            natural_unit = (match.group(2) or "").lower()
            parsed_unit = (
                "piece" if natural_unit.startswith("piece") else
                "bowl" if natural_unit.startswith("bowl") else
                "plate" if natural_unit.startswith("plate") else
                "cup" if natural_unit.startswith("cup") else
                "serving" if natural_unit.startswith("serving") else ""
            )

    key = resolve_food_name(parsed_name)
    if key is None:
        return {
            "foodName": parsed_name, "canonicalName": None,
            "quantity": parsed_quantity, "servingUnit": parsed_unit or "serving",
            "servingGrams": None, "calories": None, "carbohydrates": None,
            "protein": None, "fat": None, "fiber": None, "glycemicIndex": None,
            "available": False, "confidence": "unavailable", "source": "unknown_food",
        }

    gi, carbs_100g, protein_100g, fiber_100g, typical_grams, _ = FOOD_DB[key]
    default_grams, default_unit = FOOD_SERVINGS.get(key, (typical_grams, "serving"))
    unit = parsed_unit or default_unit
    if unit in {"g", "gram", "grams"}:
        serving_grams = parsed_quantity
        unit = "g"
    else:
        serving_grams = parsed_quantity * default_grams
        unit = default_unit
    multiplier = serving_grams / 100
    energy_fat = FOOD_ENERGY_FAT_PER_100G.get(key)
    return {
        "foodName": parsed_name,
        "canonicalName": key,
        "quantity": parsed_quantity,
        "servingUnit": unit,
        "servingGrams": round(serving_grams, 1),
        "calories": round(energy_fat[0] * multiplier) if energy_fat else None,
        "carbohydrates": round(carbs_100g * multiplier, 1),
        "protein": round(protein_100g * multiplier, 1),
        "fat": round(energy_fat[1] * multiplier, 1) if energy_fat else None,
        "fiber": round(fiber_100g * multiplier, 1),
        "glycemicIndex": round(gi * 100),
        "available": True,
        "confidence": "medium" if energy_fat else "partial",
        "source": "local_food_database",
    }


# ─────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────
def get_food(name: str) -> Optional[dict]:
    """
    Get food data by name. Case-insensitive, handles spaces/hyphens.
    Returns None if not found.
    """
    key = name.lower().replace(' ', '_').replace('-', '_')
    data = FOOD_DB.get(key)
    if not data:
        # Fuzzy match: find any key that contains the search term
        for k, v in FOOD_DB.items():
            if key in k or k in key:
                data = v
                key  = k
                break
    if not data:
        # Return default
        return {
            'name': name, 'gi': 0.55, 'carbs': 20.0, 'protein': 5.0,
            'fiber': 2.0, 'portion_g': 150, 'category': 'unknown',
            'found': False,
        }
    gi, carbs, protein, fiber, portion, cat = data
    return {
        'name':       key,
        'gi':         gi,
        'carbs_per_100g': carbs,
        'protein_per_100g': protein,
        'fiber_per_100g': fiber,
        'typical_portion_g': portion,
        'category':   cat,
        'impact':     'HIGH' if gi > 0.65 else 'MEDIUM' if gi > 0.45 else 'LOW',
        'found':      True,
    }


def search_food(query: str) -> list:
    """Search foods by partial name. Returns up to 8 matches."""
    q = query.lower().replace(' ', '_')
    matches = []
    for k, (gi, carbs, *_) in FOOD_DB.items():
        if q in k:
            matches.append({'name': k, 'gi': gi, 'carbs': carbs,
                            'impact': 'HIGH' if gi > 0.65 else 'MEDIUM' if gi > 0.45 else 'LOW'})
    return sorted(matches, key=lambda x: x['gi'])[:8]


def predict_meal_glucose(food_name: str, portion_g: float,
                         current_glucose: Optional[float] = None,
                         p3_index: Optional[float] = None,
                         hour: int = 13) -> Optional[dict]:
    """
    Predict glucose response to a specific food + portion.

    Args:
        food_name:       Name of food (Indian or international)
        portion_g:       Serving size in grams
        current_glucose: Current blood glucose mg/dL
        p3_index:        Insulin sensitivity 0-100
        hour:            Hour of eating (0-23, for circadian effect)

    Returns:
        Dict with peak glucose, time to peak, curve, and recommendations
    """
    if current_glucose is None or p3_index is None:
        return None
    food = get_food(food_name)
    gi   = food['gi']
    # get_food() returns different key names depending on whether the food
    # was found in FOOD_DB (carbs_per_100g/protein_per_100g/fiber_per_100g)
    # or not (carbs/protein/fiber, generic default values) — normalize here
    # so this function works for ANY food name a user types, not just the
    # 86 in FOOD_DB. This was a pre-existing crash (KeyError) for unknown
    # foods before this fix.
    carbs_per_100g   = food.get('carbs_per_100g', food.get('carbs', 20.0))
    protein_per_100g = food.get('protein_per_100g', food.get('protein', 5.0))
    fiber_per_100g   = food.get('fiber_per_100g', food.get('fiber', 2.0))

    # Net carbs (subtract half of fiber — slow digesting)
    net_carbs = max(0, (carbs_per_100g - fiber_per_100g * 0.5)
                    * portion_g / 100)

    # Glycemic load
    gl = net_carbs * gi

    # Circadian adjustment
    dawn    = 1.15 if 4 <= hour < 8 else 1.0
    evening = 1.10 if 18 <= hour < 22 else 1.0
    circ    = dawn * evening

    # Sensitivity adjustment (high resistance = bigger spike)
    sens_factor = max(0.5, 2.0 - p3_index / 100)

    # Protein blunting effect (protein slows absorption)
    protein_factor = 1 - protein_per_100g * portion_g / 10000

    # Peak glucose rise
    peak_rise = gl * 0.55 * circ * sens_factor * protein_factor

    # Protein secondary effect (gluconeogenesis ~2h later)
    protein_secondary = protein_per_100g * portion_g / 100 * 0.15

    total_rise = peak_rise + protein_secondary
    peak_g = min(400, current_glucose + total_rise)

    # Time to peak (lower GI = later peak)
    peak_min = int(25 + (1 - gi) * 40)

    # Recovery time (p3-dependent)
    recovery_min = int(75 + (100 - p3_index) * 0.9)

    # Generate full glucose curve (0 to 4h)
    curve = []
    for t in range(0, 241, 5):
        if t <= peak_min:
            ratio = (t / peak_min) ** 1.8
            g = current_glucose + total_rise * ratio
        else:
            decay = math.exp(-0.018 * (t - peak_min) * (p3_index / 70))
            g = current_glucose + total_rise * decay
        curve.append({'min': t, 'glucose': round(max(60, g), 1)})

    # Spike assessment
    spike_risk = 'HIGH' if peak_g > 200 else 'MEDIUM' if peak_g > 165 else 'LOW'
    safe_portion = None
    if spike_risk == 'HIGH':
        # Estimate safe portion
        safe_rise = 165 - current_glucose
        safe_carbs = safe_rise / (0.55 * gi * circ * sens_factor * protein_factor + 0.001)
        safe_portion = max(30, round(safe_carbs / (carbs_per_100g / 100 + 0.001)))

    # Better alternatives in same category
    same_cat = [(k, v) for k, v in FOOD_DB.items()
                if v[5] == food['category'] and v[0] < gi * 0.8]
    alternatives = sorted(same_cat, key=lambda x: x[1][0])[:3]

    return {
        'food':            food_name,
        'portion_g':       portion_g,
        'category':        food['category'],
        'glycemic_index':  gi,
        'net_carbs_g':     round(net_carbs, 1),
        'glycemic_load':   round(gl, 1),
        'peak_glucose':    round(peak_g, 1),
        'peak_rise_mg_dl': round(total_rise, 1),
        'time_to_peak_min': peak_min,
        'recovery_min':    recovery_min,
        'spike_risk':      spike_risk,
        'safe_portion_g':  safe_portion,
        'glucose_curve':   curve,
        'better_alternatives': [{'food': k, 'gi': v[0]} for k, v in alternatives],
        'tip': _get_food_tip(food_name, gi, spike_risk, p3_index),
    }


def _get_food_tip(food: str, gi: float, spike_risk: str,
                  p3_index: float) -> str:
    if spike_risk == 'HIGH':
        return (f"{food.replace('_',' ').title()} has high GI ({gi:.0%}). "
                f"Eat half the portion, add dal/protein, "
                f"and take a 15-min walk after to cut spike by 25%.")
    if spike_risk == 'MEDIUM':
        return (f"Moderate GI. Pair with high-fiber vegetables "
                f"(palak, bitter gourd) and eat slowly. "
                f"Your sensitivity ({p3_index:.0f}%) means ~{int(75+(100-p3_index)*0.9)} min recovery.")
    return (f"Excellent choice for T2D management. Low GI means "
            f"slow glucose release. This supports your reversal plan.")


def get_category_ranking(category: str) -> list:
    """Get all foods in a category ranked by GI (best to worst for T2D)."""
    items = [(k, v) for k, v in FOOD_DB.items() if v[5] == category]
    return [{'name': k, 'gi': v[0], 'carbs': v[1], 'fiber': v[3],
             'impact': 'HIGH' if v[0]>0.65 else 'MEDIUM' if v[0]>0.45 else 'LOW'}
            for k, v in sorted(items, key=lambda x: x[1][0])]


def get_best_foods_for_t2d(n: int = 10) -> list:
    """Return the N best foods for T2D management (lowest GI + high fiber)."""
    scored = []
    for k, (gi, carbs, prot, fiber, *_) in FOOD_DB.items():
        score = (1 - gi) * 0.5 + (fiber / 10) * 0.3 + (prot / 20) * 0.2
        scored.append({'name': k, 'gi': gi, 'fiber': fiber,
                       'protein': prot, 'score': round(score, 3)})
    return sorted(scored, key=lambda x: -x['score'])[:n]


def get_worst_foods_for_t2d(n: int = 10) -> list:
    """Return the N worst foods for T2D (highest GI + low fiber)."""
    scored = []
    for k, (gi, carbs, prot, fiber, *_) in FOOD_DB.items():
        risk = gi * 0.6 + (carbs / 100) * 0.3 + (1 - fiber / 10) * 0.1
        scored.append({'name': k, 'gi': gi, 'carbs': carbs,
                       'fiber': fiber, 'risk': round(risk, 3)})
    return sorted(scored, key=lambda x: -x['risk'])[:n]


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 58)
    print("  BioMirror Indian Food Glycemic Database")
    print(f"  Foods in database: {len(FOOD_DB)}")
    print("=" * 58)

    print("\n  Best 10 foods for T2D management:")
    for i, f in enumerate(get_best_foods_for_t2d(10), 1):
        print(f"  {i:>2}. {f['name']:<25} GI={f['gi']:.2f}  "
              f"fiber={f['fiber']}g  protein={f['protein']}g")

    print("\n  Worst 10 foods for T2D:")
    for i, f in enumerate(get_worst_foods_for_t2d(10), 1):
        print(f"  {i:>2}. {f['name']:<25} GI={f['gi']:.2f}  "
              f"carbs={f['carbs']}g  risk={f['risk']}")

    print("\n  Meal prediction examples (glucose=160, p3=65):")
    for food, portion in [('idli',150),('brown_rice',150),('white_rice',150),
                           ('jalebi',50),('dal_toor',200),('guava',150)]:
        r = predict_meal_glucose(food, portion, 160, 65, hour=13)
        risk_icon = "🔴" if r['spike_risk']=='HIGH' else "🟡" if r['spike_risk']=='MEDIUM' else "🟢"
        print(f"  {risk_icon} {food:<20} {portion}g → peak {r['peak_glucose']:>5.0f} mg/dL "
              f"(+{r['peak_rise_mg_dl']:>4.0f})  {r['spike_risk']}")
        if r['safe_portion_g']:
            print(f"     Safe portion: {r['safe_portion_g']}g  "
                  f"Alt: {[a['food'] for a in r['better_alternatives'][:2]]}")
