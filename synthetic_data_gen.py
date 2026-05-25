"""
synthetic_data_gen.py
=====================
Synthetic inventory analytics data for a fictional outdoor specialty
retailer (~$150M revenue, "Summit & Stone Outfitters").

Outputs
-------
  data/department_summary.csv    — one row per department, aggregated metrics
                                   + on-order rollup columns
  data/sku_detail.csv            — individual SKUs: turns, markdown, velocity tier
  data/sale_cadence.csv          — per-SKU sale velocity stats (avg gap, max gap,
                                   days since last sale, transaction count)
  data/open_purchase_orders.csv  — open POs (sparse — not every SKU has one)

All files join on Department and/or SKUID.
No data from any real retailer is used or reflected.
"""

import argparse, os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Department definitions
# turns_mu/sd : lognormal params for InventoryTurns (median = exp(turns_mu))
# markup_mu/sd: markup ratio (sell / cost), normal
# seas_level  : "low" / "medium" / "high" — drives SeasonalityCV
# inv_range   : avg daily inventory at cost, USD
# n_skus      : (min, max) active SKU count
#
# Turns benchmarks (general specialty retail, not retailer-specific):
#   Consumables / small accessories : 5–9x
#   Footwear / apparel              : 2.5–4.5x
#   Technical soft goods            : 1.8–3.5x
#   Recreational hard goods         : 1.0–2.2x
#   Big-ticket / specialty equipment: 0.6–1.4x
# ---------------------------------------------------------------------------

DEPARTMENTS = {
    "Ski Hardware": {
        "turns_mu": np.log(1.1),  "turns_sd": 0.40,
        "markup_mu": 1.21,        "markup_sd": 0.07,
        "seas_level": "high",
        "inv_range": (260_000, 400_000),
        "n_skus": (90, 160),
    },
    "Snowboard Hardware": {
        "turns_mu": np.log(1.2),  "turns_sd": 0.40,
        "markup_mu": 1.23,        "markup_sd": 0.07,
        "seas_level": "high",
        "inv_range": (200_000, 340_000),
        "n_skus": (80, 140),
    },
    "Bikes": {
        "turns_mu": np.log(0.9),  "turns_sd": 0.45,
        "markup_mu": 1.17,        "markup_sd": 0.06,
        "seas_level": "medium",
        "inv_range": (350_000, 560_000),
        "n_skus": (120, 220),
    },
    "Bike Components": {
        "turns_mu": np.log(1.5),  "turns_sd": 0.48,
        "markup_mu": 1.28,        "markup_sd": 0.09,
        "seas_level": "medium",
        "inv_range": (100_000, 200_000),
        "n_skus": (200, 380),
    },
    "Kayaks & Canoes": {
        "turns_mu": np.log(0.8),  "turns_sd": 0.42,
        "markup_mu": 1.19,        "markup_sd": 0.07,
        "seas_level": "medium",
        "inv_range": (140_000, 260_000),
        "n_skus": (50, 100),
    },
    "Paddle & SUP Accessories": {
        "turns_mu": np.log(2.8),  "turns_sd": 0.45,
        "markup_mu": 1.40,        "markup_sd": 0.11,
        "seas_level": "medium",
        "inv_range": (50_000, 110_000),
        "n_skus": (100, 180),
    },
    "Camping & Shelter": {
        "turns_mu": np.log(1.6),  "turns_sd": 0.42,
        "markup_mu": 1.30,        "markup_sd": 0.09,
        "seas_level": "low",
        "inv_range": (120_000, 240_000),
        "n_skus": (120, 220),
    },
    "Sleep Systems": {
        "turns_mu": np.log(1.4),  "turns_sd": 0.40,
        "markup_mu": 1.32,        "markup_sd": 0.09,
        "seas_level": "low",
        "inv_range": (80_000, 160_000),
        "n_skus": (80, 150),
    },
    "Climbing Gear": {
        "turns_mu": np.log(2.2),  "turns_sd": 0.42,
        "markup_mu": 1.34,        "markup_sd": 0.09,
        "seas_level": "low",
        "inv_range": (70_000, 140_000),
        "n_skus": (130, 240),
    },
    "Packs & Luggage": {
        "turns_mu": np.log(1.8),  "turns_sd": 0.42,
        "markup_mu": 1.38,        "markup_sd": 0.10,
        "seas_level": "low",
        "inv_range": (90_000, 180_000),
        "n_skus": (100, 190),
    },
    "Snow Outerwear": {
        "turns_mu": np.log(1.7),  "turns_sd": 0.43,
        "markup_mu": 1.72,        "markup_sd": 0.16,
        "seas_level": "high",
        "inv_range": (150_000, 290_000),
        "n_skus": (160, 300),
    },
    "Technical Outerwear": {
        "turns_mu": np.log(2.0),  "turns_sd": 0.43,
        "markup_mu": 1.78,        "markup_sd": 0.17,
        "seas_level": "medium",
        "inv_range": (130_000, 250_000),
        "n_skus": (140, 260),
    },
    "Midlayer & Fleece": {
        "turns_mu": np.log(2.5),  "turns_sd": 0.42,
        "markup_mu": 1.82,        "markup_sd": 0.18,
        "seas_level": "medium",
        "inv_range": (100_000, 200_000),
        "n_skus": (160, 280),
    },
    "Baselayer": {
        "turns_mu": np.log(3.2),  "turns_sd": 0.40,
        "markup_mu": 1.88,        "markup_sd": 0.19,
        "seas_level": "low",
        "inv_range": (60_000, 130_000),
        "n_skus": (120, 220),
    },
    "Casual Apparel": {
        "turns_mu": np.log(3.5),  "turns_sd": 0.42,
        "markup_mu": 2.10,        "markup_sd": 0.22,
        "seas_level": "low",
        "inv_range": (80_000, 160_000),
        "n_skus": (200, 360),
    },
    "Hiking Footwear": {
        "turns_mu": np.log(2.4),  "turns_sd": 0.40,
        "markup_mu": 1.62,        "markup_sd": 0.13,
        "seas_level": "low",
        "inv_range": (100_000, 200_000),
        "n_skus": (100, 190),
    },
    "Snow Boots & Footwear": {
        "turns_mu": np.log(1.6),  "turns_sd": 0.40,
        "markup_mu": 1.58,        "markup_sd": 0.13,
        "seas_level": "high",
        "inv_range": (80_000, 160_000),
        "n_skus": (80, 150),
    },
    "Casual Footwear": {
        "turns_mu": np.log(3.0),  "turns_sd": 0.40,
        "markup_mu": 1.70,        "markup_sd": 0.14,
        "seas_level": "low",
        "inv_range": (70_000, 140_000),
        "n_skus": (90, 170),
    },
    "Helmets & Protection": {
        "turns_mu": np.log(2.0),  "turns_sd": 0.42,
        "markup_mu": 1.36,        "markup_sd": 0.10,
        "seas_level": "medium",
        "inv_range": (60_000, 120_000),
        "n_skus": (80, 150),
    },
    "Goggles & Eyewear": {
        "turns_mu": np.log(2.4),  "turns_sd": 0.42,
        "markup_mu": 1.50,        "markup_sd": 0.12,
        "seas_level": "high",
        "inv_range": (50_000, 110_000),
        "n_skus": (90, 160),
    },
    "Gloves & Handwear": {
        "turns_mu": np.log(4.0),  "turns_sd": 0.42,
        "markup_mu": 1.68,        "markup_sd": 0.14,
        "seas_level": "high",
        "inv_range": (40_000, 90_000),
        "n_skus": (80, 150),
    },
    "Hydration & Nutrition": {
        "turns_mu": np.log(6.0),  "turns_sd": 0.38,
        "markup_mu": 1.55,        "markup_sd": 0.12,
        "seas_level": "low",
        "inv_range": (30_000, 70_000),
        "n_skus": (120, 220),
    },
    "Camp Kitchen & Tools": {
        "turns_mu": np.log(3.8),  "turns_sd": 0.40,
        "markup_mu": 1.42,        "markup_sd": 0.11,
        "seas_level": "low",
        "inv_range": (40_000, 90_000),
        "n_skus": (130, 240),
    },
    "Navigation & Safety": {
        "turns_mu": np.log(2.6),  "turns_sd": 0.42,
        "markup_mu": 1.38,        "markup_sd": 0.10,
        "seas_level": "low",
        "inv_range": (35_000, 80_000),
        "n_skus": (80, 150),
    },
}

SEASONALITY_CV = {
    "low":    {"mu": 0.22, "sd": 0.07},
    "medium": {"mu": 0.48, "sd": 0.11},
    "high":   {"mu": 0.98, "sd": 0.16},
}

BRANDS = [
    "Ridgecrest", "Alpenglow", "Stonepath", "Clearwater", "Ironpass",
    "Frostline", "Trailhead", "Highmark", "Coldrun", "Summitform",
    "Duskridge", "Terracross", "Opentrail", "Canyonset", "Mesaflow",
    "Tidefall", "Powderline", "Gravelpath", "Northeave", "Firncrest",
    "Hardpack", "Edgewood", "Deepfreeze", "Coastline", "Basecamp",
    "Pinnacle", "Dustfall", "Shorebreak", "Rockline", "Iceform",
]

MODELS = [
    "Pro", "Elite", "Sport", "Base", "LTD", "Carbon", "Comp", "Tour",
    "Lite", "MAX", "Series II", "X1", "Signature", "Team", "Foundation",
    "Apex", "Ridge", "Summit", "Crest", "Trail", "Venture", "Enduro",
    "Xpedition", "Traverse", "Origin", "Ascent",
]

VELOCITY_THRESHOLDS = {
    "Dead":     (0.00, 0.50),
    "Slow":     (0.50, 1.50),
    "Moderate": (1.50, 3.00),
    "Fast":     (3.00, float("inf")),
}


def velocity_tier(turns: float) -> str:
    for tier, (lo, hi) in VELOCITY_THRESHOLDS.items():
        if lo <= turns < hi:
            return tier
    return "Fast"


# ---------------------------------------------------------------------------
# SKU generation
# ---------------------------------------------------------------------------

def make_skus(rng, dept_name: str, cfg: dict, n_skus: int,
              dept_brands: list) -> pd.DataFrame:
    """
    Generate individual SKU rows for one department.
    Turns → inventory → COGS → revenue (never the reverse).
    Markdown probability is inversely correlated with turns.
    """
    rows = []
    total_daily_inv = rng.uniform(*cfg["inv_range"])
    # Dirichlet gives realistic Pareto-ish inventory concentration
    inv_shares = rng.dirichlet(np.ones(n_skus) * 0.6)

    for i in range(n_skus):
        brand  = rng.choice(dept_brands)
        model  = rng.choice(MODELS)
        sku_id = f"{dept_name.replace(' ','_').replace('&','AND').upper()[:14]}_{i+1:04d}"

        turns = float(np.clip(
            rng.lognormal(cfg["turns_mu"], cfg["turns_sd"] * 1.4),
            0.01, 25.0
        ))

        avg_daily_inv = max(20.0, total_daily_inv * inv_shares[i])
        annual_cogs   = turns * avg_daily_inv
        markup        = float(np.clip(
            rng.normal(cfg["markup_mu"], cfg["markup_sd"] * 1.5), 1.02, 4.0
        ))
        net_revenue   = annual_cogs * markup
        avg_unit_cost = rng.uniform(18, 650)
        units_sold    = max(0, int(annual_cogs / avg_unit_cost))

        # Markdown: slow SKUs are more likely to be marked down, deeper cuts
        # P(markdown) rises as turns falls — logistic-ish relationship
        md_prob  = float(np.clip(1 / (1 + np.exp(2.5 * (turns - 1.2))), 0.02, 0.92))
        is_md    = bool(rng.random() < md_prob)
        # Markdown depth: 10–50% off, deeper for slower SKUs
        md_depth = float(np.clip(
            rng.normal(0.12 + 0.22 * (1 - min(turns / 3.0, 1.0)), 0.06),
            0.05, 0.60
        )) if is_md else 0.0
        # Fraction of units sold at markdown price
        md_rate  = float(np.clip(
            rng.normal(0.30 + 0.35 * (1 - min(turns / 3.0, 1.0)), 0.10),
            0.0, 1.0
        )) if is_md else 0.0

        rows.append({
            "Department":      dept_name,
            "SKUID":           sku_id,
            "Brand":           brand,
            "Model":           f"{brand} {model}",
            "UnitsSold":       units_sold,
            "AvgUnitCost":     round(avg_unit_cost, 2),
            "NetRevenue":      round(net_revenue, 2),
            "COGS":            round(annual_cogs, 2),
            "AvgDailyInvCOGS": round(avg_daily_inv, 2),
            "MarkupRatio":     round(markup, 4),
            "ItemTurns":       round(turns, 4),
            "VelocityTier":    velocity_tier(turns),
            "NoSalesHistory":  units_sold == 0,
            "IsMarkdown":      is_md,
            "MarkdownDepth":   round(md_depth, 4),
            "MarkdownRate":    round(md_rate, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sale cadence generation
# Produces per-SKU velocity statistics — not full transaction log
# ---------------------------------------------------------------------------

def make_sale_cadence(rng, df_skus: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sale cadence summary stats for each SKU.
    Stats are consistent with ItemTurns:
      - Fast SKUs  → low AvgDaysBetweenSales, low MaxGapDays, recent LastSale
      - Dead SKUs  → high gaps, DaysSinceLastSale in the hundreds
      - No history → nulls throughout
    """
    rows = []
    for _, sku in df_skus.iterrows():
        turns = sku["ItemTurns"]
        units = sku["UnitsSold"]

        if units == 0:
            # No sales history — all nulls
            rows.append({
                "SKUID":               sku["SKUID"],
                "Department":          sku["Department"],
                "TxnCount":            0,
                "AvgDaysBetweenSales": None,
                "MaxGapDays":          None,
                "DaysSinceLastSale":   None,
                "FirstSaleDaysAgo":    None,
            })
            continue

        # Approximate transaction count: units sold spread across visits
        # (multiple units sometimes in one transaction)
        avg_units_per_txn = float(np.clip(rng.normal(1.4, 0.4), 1.0, 5.0))
        txn_count = max(1, int(units / avg_units_per_txn))

        # Avg days between sales inversely related to turns
        # turns=6 → ~14 days avg; turns=1 → ~90 days avg; turns=0.3 → ~200 days avg
        avg_gap_mu = 365.0 / (turns * avg_units_per_txn + 0.5)
        avg_gap    = float(np.clip(rng.lognormal(np.log(max(1, avg_gap_mu)), 0.35),
                                   1.0, 365.0))

        # Max gap is always >= avg gap, typically 1.5–4x for slow movers
        gap_multiplier = float(np.clip(rng.lognormal(np.log(2.2), 0.45), 1.1, 8.0))
        max_gap        = float(np.clip(avg_gap * gap_multiplier, avg_gap, 365.0))

        # Days since last sale: slow movers trail off, fast movers recent
        # Dead tier: likely hasn't sold in months
        if turns < 0.5:
            days_since = float(np.clip(rng.lognormal(np.log(180), 0.50), 60, 400))
        elif turns < 1.5:
            days_since = float(np.clip(rng.lognormal(np.log(60),  0.55), 10, 280))
        else:
            days_since = float(np.clip(rng.lognormal(np.log(18),  0.60),  1, 120))

        # First sale: roughly txn_count * avg_gap days ago (bounded to ~2 years)
        first_sale_days_ago = float(np.clip(
            txn_count * avg_gap + days_since, days_since + avg_gap, 730
        ))

        rows.append({
            "SKUID":               sku["SKUID"],
            "Department":          sku["Department"],
            "TxnCount":            txn_count,
            "AvgDaysBetweenSales": round(avg_gap, 1),
            "MaxGapDays":          round(max_gap, 1),
            "DaysSinceLastSale":   round(days_since, 1),
            "FirstSaleDaysAgo":    round(first_sale_days_ago, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Open purchase orders
# Sparse — only a subset of SKUs have open POs.
# Watchlisted / slow departments have more POs to amplify the problem.
# ---------------------------------------------------------------------------

def make_open_pos(rng, df_skus: pd.DataFrame,
                  watchlist_depts: set) -> pd.DataFrame:
    """
    Generate open PO rows. Each row is one PO line (one SKU, one order).
    A SKU can have at most one open PO line.
    PO probability is higher for slow departments (they're the concerning ones).
    """
    rows = []
    for _, sku in df_skus.iterrows():
        dept    = sku["Department"]
        turns   = sku["ItemTurns"]
        is_slow = dept in watchlist_depts or turns < 1.5

        # Probability of having an open PO
        # Slow SKUs in problem depts: ~35%  |  fast / clean: ~8%
        po_prob = 0.35 if is_slow else 0.08
        if rng.random() > po_prob:
            continue

        avg_unit_cost  = sku["AvgUnitCost"]
        units_on_order = max(1, int(rng.lognormal(np.log(8), 0.70)))
        po_cost        = round(units_on_order * avg_unit_cost, 2)

        # Expected receipt: sooner for fast movers, could be soon or delayed for slow
        if is_slow:
            receipt_days = int(np.clip(rng.normal(45, 30), 7, 120))
        else:
            receipt_days = int(np.clip(rng.normal(21, 14), 3,  90))

        rows.append({
            "Department":       dept,
            "SKUID":            sku["SKUID"],
            "Brand":            sku["Brand"],
            "UnitsOnOrder":     units_on_order,
            "POCostValue":      po_cost,
            "ExpectedReceiptDays": receipt_days,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Department summary (aggregated from SKUs)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prior year STR patterns — calibrated to outdoor specialty retail dynamics
#
# boom_bust     : bikes, kayaks, ski/snowboard — COVID demand surge 2021-2023
#                 inflated historical STR; over-buying into demand normalization
#                 means prior STR is genuinely above current turns
# snow_seasonal : snow categories — snowfall-driven, wide year-to-year swing
#                 good seasons push STR up, bad seasons pull it down
# moderate      : some demand variability but no structural boom/bust cycle
# stable        : replenishment categories with consistent year-round demand
# ---------------------------------------------------------------------------

PRIOR_STR_PATTERNS = {
    "boom_bust": {
        "departments": {
            "Bikes", "Kayaks & Canoes", "Ski Hardware",
            "Snowboard Hardware", "Paddle & SUP Accessories",
        },
        # Prior STR = current_turns * lognormal(mu, sd)
        # mu > 1 means prior years ran hotter — the correction makes current look weak
        "multiplier_mu": 1.45,
        "multiplier_sd": 0.18,
    },
    "snow_seasonal": {
        "departments": {
            "Snow Outerwear", "Snow Boots & Footwear",
            "Goggles & Eyewear", "Gloves & Handwear",
        },
        # Snowfall variance: good years spike up, bad years drop — centered near current
        "multiplier_mu": 1.08,
        "multiplier_sd": 0.32,
    },
    "moderate": {
        "departments": {
            "Bike Components", "Camping & Shelter", "Sleep Systems",
            "Packs & Luggage", "Helmets & Protection",
            "Technical Outerwear", "Midlayer & Fleece",
        },
        # Mild upward drift — slight demand softening post-peak, not structural
        "multiplier_mu": 1.06,
        "multiplier_sd": 0.12,
    },
    # Everything else falls through to stable defaults below
}

PRIOR_STR_STABLE = {"multiplier_mu": 1.00, "multiplier_sd": 0.07}


def make_prior_year_str(rng, dept_name: str, current_turns: float) -> float:
    """
    Simulate a 3-year average prior STR for the department.

    Pattern is determined by department category:
    - boom_bust: prior STR materially above current (post-COVID correction)
    - snow_seasonal: high variance, direction uncertain (snowfall-dependent)
    - moderate: slight upward drift, minor softening
    - stable: tight band around current turns

    Each of the 3 prior years is drawn independently around the pattern
    multiplier, then averaged — mimicking real FY averaging.
    """
    cfg = PRIOR_STR_STABLE
    for pattern_cfg in PRIOR_STR_PATTERNS.values():
        if dept_name in pattern_cfg["departments"]:
            cfg = pattern_cfg
            break

    prior_years = []
    for _ in range(3):
        multiplier = rng.lognormal(
            mean=np.log(cfg["multiplier_mu"]),
            sigma=cfg["multiplier_sd"]
        )
        prior_years.append(max(0.05, current_turns * multiplier))

    return round(float(np.mean(prior_years)), 4)


def aggregate_department(dept_name: str, df_skus: pd.DataFrame,
                         seas_cv: float,
                         df_pos: pd.DataFrame,
                         rng) -> dict:
    turns   = df_skus["ItemTurns"]
    w       = df_skus["AvgDailyInvCOGS"]
    wt_turns = float((turns * w).sum() / w.sum())

    dept_pos       = df_pos[df_pos["Department"] == dept_name]
    on_order_cogs  = float(dept_pos["POCostValue"].sum())
    avg_daily_inv  = float(df_skus["AvgDailyInvCOGS"].sum())
    on_order_ratio = round(on_order_cogs / max(avg_daily_inv, 1), 4)

    return {
        "Department":       dept_name,
        "InventoryTurns":   round(wt_turns, 4),
        "AvgDailyInvCOGS":  round(avg_daily_inv, 2),
        "NetBookedRevenue": round(df_skus["NetRevenue"].sum(), 2),
        "SKUCount":         len(df_skus),
        "BrandCount":       df_skus["Brand"].nunique(),
        "TurnsCV":          round(turns.std() / turns.mean(), 4),
        "TurnsP25":         round(turns.quantile(0.25), 4),
        "TurnsP50":         round(turns.quantile(0.50), 4),
        "TurnsP75":         round(turns.quantile(0.75), 4),
        "MarkupRatio":      round(df_skus["MarkupRatio"].mean(), 4),
        "SeasonalityCV":    round(seas_cv, 4),
        "ZeroTurnSKUs":     int((turns < 0.10).sum()),
        "SlowSKUs":         int((turns < 0.50).sum()),
        "FastSKUs":         int((turns >= 2.0).sum()),
        "MarkdownSKUs":     int(df_skus["IsMarkdown"].sum()),
        "AvgMarkdownDepth": round(df_skus.loc[df_skus["IsMarkdown"], "MarkdownDepth"].mean(), 4)
                            if df_skus["IsMarkdown"].any() else 0.0,
        "OnOrderCOGS":      round(on_order_cogs, 2),
        "OnOrderRatio":     on_order_ratio,   # on-order / avg daily inv
        "PriorYearSTR":     make_prior_year_str(rng, dept_name, wt_turns),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42):
    rng = np.random.default_rng(seed)
    os.makedirs("data", exist_ok=True)

    all_skus  = []
    dept_rows = []

    # Pass 1: generate all SKUs
    dept_brand_map = {}
    for dept_name, cfg in DEPARTMENTS.items():
        sp      = SEASONALITY_CV[cfg["seas_level"]]
        seas_cv = float(np.clip(rng.normal(sp["mu"], sp["sd"]), 0.05, 2.0))
        n_skus  = int(rng.uniform(*cfg["n_skus"]))

        n_brands    = int(np.clip(rng.normal(6, 2), 3, 12))
        n_brands    = min(n_brands, len(BRANDS))
        dept_brands = list(rng.choice(BRANDS, size=n_brands, replace=False))
        dept_brand_map[dept_name] = (cfg, seas_cv, dept_brands)

        df_skus = make_skus(rng, dept_name, cfg, n_skus, dept_brands)
        all_skus.append(df_skus)

    df_skus_all = pd.concat(all_skus, ignore_index=True)

    # Pass 2: sale cadence
    df_cadence = make_sale_cadence(rng, df_skus_all)

    # Pass 3: open POs — need a rough watchlist first (slow depts by turns)
    dept_turns = df_skus_all.groupby("Department")["ItemTurns"].apply(
        lambda x: float((x * df_skus_all.loc[x.index, "AvgDailyInvCOGS"]).sum() /
                        df_skus_all.loc[x.index, "AvgDailyInvCOGS"].sum())
    )
    slow_depts = set(dept_turns[dept_turns < 1.5].index)
    df_pos = make_open_pos(rng, df_skus_all, slow_depts)

    # Pass 4: department summary (needs POs for on-order columns)
    for dept_name, (cfg, seas_cv, _) in dept_brand_map.items():
        dept_skus = df_skus_all[df_skus_all["Department"] == dept_name]
        summary   = aggregate_department(dept_name, dept_skus, seas_cv, df_pos, rng)
        dept_rows.append(summary)

    df_dept = pd.DataFrame(dept_rows)

    # Write outputs
    df_dept.to_csv("data/department_summary.csv",   index=False)
    df_skus_all.to_csv("data/sku_detail.csv",        index=False)
    df_cadence.to_csv("data/sale_cadence.csv",       index=False)
    df_pos.to_csv("data/open_purchase_orders.csv",   index=False)

    # Summary
    print(f"Synthetic data generated  (seed={seed})\n")
    print(f"  Departments  : {len(df_dept)}")
    print(f"  Total SKUs   : {len(df_skus_all):,}")
    print(f"  Cadence rows : {len(df_cadence):,}  ({(df_cadence['TxnCount']==0).sum()} no-history SKUs)")
    print(f"  Open PO lines: {len(df_pos):,}  across {df_pos['Department'].nunique()} departments")

    print(f"\n{'Department':<28} {'Turns':>6} {'TurnsCV':>8} {'SlowSKUs':>9} "
          f"{'MdSKUs':>7} {'OnOrderRatio':>13}")
    print("-" * 80)
    for _, r in df_dept.sort_values("InventoryTurns").iterrows():
        print(f"  {r['Department']:<26} {r['InventoryTurns']:>6.2f} "
              f"{r['TurnsCV']:>8.3f} {int(r['SlowSKUs']):>9} "
              f"{int(r['MarkdownSKUs']):>7} {r['OnOrderRatio']:>13.2f}x")

    return df_dept, df_skus_all, df_cadence, df_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
