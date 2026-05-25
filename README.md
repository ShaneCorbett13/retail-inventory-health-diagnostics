# Inventory Health Diagnostics

A department-level inventory diagnostic framework for a fictional outdoor specialty
retailer (~$150M revenue, "Summit & Stone Outfitters"). Identifies underperforming
inventory using a five-layer methodology and surfaces SKU-level detail for any
flagged department.

All data is synthetic. See [`synthetic_data_gen.py`](synthetic_data_gen.py) to
regenerate it.

---

## The Problem

Retailers carry inventory that isn't earning its keep — departments where capital is
tied up in slow-moving or non-moving stock, quietly dragging on cash flow and margin.
Aggregate turns figures mask this: a department can show acceptable average turns while
a significant portion of its SKU count has effectively stopped moving. A department
already flagged for slow turns with more inventory inbound on open POs is a materially
different problem than the same turns with a clean order book.

The goal is to surface those situations early enough to act — before the markdown
spiral, before the inventory ages out of relevance, and before the next PO receipt
compounds a problem that's already there.

---

## Methodology

### Regression model

`log(InventoryTurns) ~ log(PriorYearSTR) + log(MarkupRatio) + SeasonalityCV + log(NetBookedRevenue)`

`PriorYearSTR` — a 3-year average sell-through rate — is the primary structural
predictor. Controlling for a department's own historical baseline means the residual
captures *current-year underperformance relative to that department's track record*,
not just low turns in absolute terms. A department like Bikes that ran at 1.5x turns
during the 2021–2023 demand cycle and is now running at 1.0x is flagged because it's
underperforming *itself*, not just because bikes turn slowly.

Departments with standardized residual < −1.0 enter the regression watchlist.

### Five diagnostic layers

Each layer is designed to catch a failure mode the others miss.

| Layer | What it catches | Flag condition |
|---|---|---|
| **Regression** | Underperformance relative to the department's own structural baseline | ResidualStd < −1.0 |
| **Dispersion** | Fast aggregate turns masking a stagnant SKU majority | TurnsCV > 0.70 |
| **Tail Drag** | Slow bottom quartile dragging the department median down | TurnsCV > 0.60 and TurnsP25 < 1.05 |
| **Sell-Through** | Low conversion of received inventory into sales | STR < 0.49 and Turns < 1.0 |
| **On-Order Compounding** | More inventory inbound than currently on hand, for departments already flagged | OnOrderRatio > 1.0 (compounding marker only) |

Departments appearing on multiple layers are highest priority — independent diagnostics
pointing at the same department strengthen the signal.

### Influence diagnostics

Cook's distance is computed after regression to identify departments that
disproportionately influence the model fit. High Cook's D on a watchlist department
strengthens the finding — it's pulling the regression line toward itself and still
underperforming. High Cook's D on a clean department warrants manual review before
trusting the model's prediction for it.

### Item-level diagnostics

For every flagged department, SKU-level detail surfaces the specific items driving the
department's position: velocity tier (Dead / Slow / Moderate / Fast), days since last
sale, average and maximum gap between sales, markdown status, and open PO exposure.
Item velocity flags apply cadence thresholds independently of turns — a SKU that sold
in one burst six months ago and hasn't moved since won't show up as a turns problem
until the denominator catches up, but the gap metric flags it immediately.

---

## Repository Structure

```
├── inventory_health_analytics.ipynb   # Main analysis notebook
├── synthetic_data_gen.py              # Generates all four data files from scratch
├── data/
│   ├── department_summary.csv         # One row per department — turns, regression
│   │                                  # outputs, watchlist flags, on-order rollup
│   ├── sku_detail.csv                 # Individual SKUs — turns, velocity tier,
│   │                                  # markdown status
│   ├── sale_cadence.csv               # Per-SKU cadence stats — avg gap, max gap,
│   │                                  # days since last sale, transaction count
│   └── open_purchase_orders.csv       # Open PO lines — sparse, SKU-level
└── README.md
```

---

## Notebook Walkthrough

| Cell | Purpose |
|---|---|
| 1 — Setup | Load all four files, merge cadence and PO data onto SKU detail, derive log features |
| 2 — Turns Distributions | Department bar chart sorted by turns, SKU-level log-turns histogram with percentile lines |
| 3 — Velocity Cadence | Dept-level rollup of avg days between sales, max gap, days since last sale; consistency vs. turns scatter |
| 4 — OLS Regression | Model fit with PriorYearSTR + MarkupRatio + SeasonalityCV + Revenue; full statsmodels summary; standardized residuals |
| 5 — Dispersion Watchlist | TurnsCV > 0.70 with SKU count and inventory floor |
| 6 — Tail Drag Watchlist | TurnsCV > 0.60 and TurnsP25 < 1.05 — slow bottom quartile |
| 7 — Sell-Through & Markdown | STR proxy watchlist; markdown overlay showing departments where discounting isn't clearing inventory |
| 8 — On-Order Compounding | OnOrderRatio > 1.0 applied to watchlisted departments only |
| 9 — Watchlist Overlap Summary | Cross-tab heatmap of all five layers; layer counts |
| 10 — Cook's Distance | Influence diagnostics; high-influence + watchlist overlap check |
| 11 — Diagnostic Plots | 2×3 panel: actual vs. predicted, residuals by department, residuals vs. dispersion, tail drag scatter, STR vs. markdown, inventory exposure |
| 12 — Item-Level Diagnostic | SKU detail per flagged department — turns, velocity tier, cadence, markdown, open PO; velocity tier breakdown table |
| 13 — Item Velocity Flags | Cadence thresholds applied per SKU across all watchlisted departments with plain-English flag strings |

---

## Findings (Synthetic Data)

Five of 24 departments flagged across at least one layer:

| Department | Turns | Layers | Flags | On-Order |
|---|---|---|---|---|
| Kayaks & Canoes | 1.05 | 3 | Regression, Dispersion, TailDrag | — |
| Bike Components | 1.81 | 2 | Dispersion, TailDrag | ⚠ 1.75x on hand |
| Snowboard Hardware | 1.46 | 1 | TailDrag | — |
| Paddle & SUP Accessories | 3.52 | 1 | Regression | — |
| Gloves & Handwear | 4.56 | 1 | Dispersion | — |

Kayaks & Canoes is the highest-priority finding — three independent layers pointing at
the same department. Bike Components is the most operationally urgent: acceptable
aggregate turns masking a slow SKU tail, with 1.75x current inventory on open POs
inbound.

Paddle & SUP Accessories is a notable case: turns of 3.52 looks healthy in isolation,
but the regression flags it because its prior year STR was materially higher — current
performance is below its own baseline, not just below the category median.

> **Note on R²:** The regression R² of 0.93 is higher than what this model would
> produce on real data. In the synthetic dataset `PriorYearSTR` is generated from
> current turns with controlled noise, making the two variables more correlated than
> they would be in practice. On the production dataset this model ran at R² ≈ 0.20,
> reflecting genuine year-to-year variance across departments.

---

## Stack

| Tool | Role |
|---|---|
| Python / pandas | Data preparation and aggregation |
| statsmodels | OLS regression and influence diagnostics |
| matplotlib / seaborn | All visualizations |
| Jupyter | Notebook environment |

The production version of this project was built in **Microsoft Fabric** using PySpark
for data preparation against a Delta Lake warehouse, with results surfaced in a
**Power BI** report. This repository is a portable, dependency-light reproduction of
the same methodology on synthetic data.

---

## Running Locally

```bash
# Install dependencies
pip install pandas numpy statsmodels matplotlib seaborn jupyter

# Regenerate synthetic data (optional — data/ is included)
python synthetic_data_gen.py

# Launch notebook
jupyter notebook inventory_health_analytics.ipynb
```
