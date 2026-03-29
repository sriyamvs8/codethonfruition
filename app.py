"""
Multi-Context Conflict Resolver
Dataset : pricing_dataset.xlsx  (sheet: Dataset)
Columns : record_id, user_goal, urgency_level, price, avg_market_price,
          competition_level, brand_value, condition_score, demand_score,
          inventory_level, days_listed, discount_allowed
Output  : final_decisions_output.xlsx
"""

import pandas as pd
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1. CONTEXT WEIGHTS  (higher = stronger influence on decision)
# ──────────────────────────────────────────────────────────────
CONTEXT_WEIGHTS = {
    "urgency":     1.00,
    "user_goal":   0.90,
    "demand":      0.75,
    "competition": 0.65,
    "brand":       0.55,
    "condition":   0.45,
    "inventory":   0.40,
    "staleness":   0.35,
    "discount":    0.30,
}

# ──────────────────────────────────────────────────────────────
# 2. ENCODERS  (text → numeric)
# ──────────────────────────────────────────────────────────────
def encode_urgency(level: str) -> float:
    return {"low": 0.1, "medium": 0.4, "high": 0.75, "critical": 1.0}.get(str(level).lower(), 0.4)

def encode_brand(val: str) -> float:
    return {"budget": 0.1, "mid_range": 0.35, "premium": 0.65, "luxury": 1.0}.get(str(val).lower(), 0.35)

def goal_vector(goal: str) -> dict:
    """Base preference scores per action, keyed by user goal."""
    table = {
        "maximize_profit":     {"hold_price": 0.9, "slight_discount": 0.4, "relist": 0.5, "heavy_discount": 0.1},
        "quick_sale":          {"hold_price": 0.1, "slight_discount": 0.6, "relist": 0.3, "heavy_discount": 0.9},
        "brand_preservation":  {"hold_price": 0.8, "slight_discount": 0.5, "relist": 0.6, "heavy_discount": 0.0},
        "inventory_clearance": {"hold_price": 0.1, "slight_discount": 0.5, "relist": 0.2, "heavy_discount": 1.0},
    }
    return table.get(str(goal).lower(), table["maximize_profit"]).copy()

# ──────────────────────────────────────────────────────────────
# 3. CONFLICT DETECTION
# ──────────────────────────────────────────────────────────────
def detect_conflicts(row) -> list:
    conflicts = []
    price_ratio = float(row["price"]) / float(row["avg_market_price"]) if float(row["avg_market_price"]) > 0 else 1.0

    # Conflict 1: price above market + high competition
    if price_ratio > 1.05 and str(row["competition_level"]).lower() == "high":
        conflicts.append({
            "type":        "price_vs_competition",
            "description": f"Price is {price_ratio:.2f}x market avg but competition is HIGH",
            "severity":    min(1.0, (price_ratio - 1.0) * 2 + 0.5),
            "pulls_toward":"slight_discount",
            "weight":      CONTEXT_WEIGHTS["competition"],
        })

    # Conflict 2: low demand + price above market
    demand = float(row["demand_score"])
    if demand < 0.3 and price_ratio > 1.0:
        conflicts.append({
            "type":        "demand_vs_price",
            "description": f"Demand score {demand:.3f} is low but price exceeds market",
            "severity":    (1 - demand) * price_ratio,
            "pulls_toward":"heavy_discount",
            "weight":      CONTEXT_WEIGHTS["demand"],
        })

    # Conflict 3: high urgency + price well above market
    urgency_score = encode_urgency(row["urgency_level"])
    if urgency_score >= 0.75 and price_ratio > 1.10:
        conflicts.append({
            "type":        "urgency_vs_price",
            "description": f"Urgency={str(row['urgency_level']).upper()} but price is {price_ratio:.2f}x market",
            "severity":    urgency_score * (price_ratio - 1.0),
            "pulls_toward":"heavy_discount" if urgency_score == 1.0 else "slight_discount",
            "weight":      CONTEXT_WEIGHTS["urgency"],
        })

    # Conflict 4: stale listing + goal not clearance
    days = int(row["days_listed"])
    goal = str(row["user_goal"]).lower()
    if days > 60 and goal not in ["quick_sale", "inventory_clearance"]:
        staleness = min(1.0, (days - 60) / 60)
        conflicts.append({
            "type":        "staleness_vs_goal",
            "description": f"Listed {days} days but goal is '{goal}'",
            "severity":    staleness,
            "pulls_toward":"relist" if staleness < 0.5 else "slight_discount",
            "weight":      CONTEXT_WEIGHTS["staleness"],
        })

    # Conflict 5: premium brand + high discount allowed + brand_preservation goal
    brand_score   = encode_brand(row["brand_value"])
    discount_frac = float(row["discount_allowed"])
    if brand_score >= 0.65 and discount_frac > 0.25 and goal == "brand_preservation":
        conflicts.append({
            "type":        "brand_vs_discount_policy",
            "description": f"{str(row['brand_value']).title()} brand conflicts with discount_allowed={discount_frac:.0%}",
            "severity":    brand_score * discount_frac,
            "pulls_toward":"slight_discount",
            "weight":      CONTEXT_WEIGHTS["brand"],
        })

    return conflicts

# ──────────────────────────────────────────────────────────────
# 4. DECISION ENGINE
# ──────────────────────────────────────────────────────────────
def resolve(row) -> dict:
    goal      = str(row["user_goal"]).lower()
    conflicts = detect_conflicts(row)
    scores    = goal_vector(goal)

    # Boost scores based on conflict pulls
    for c in conflicts:
        scores[c["pulls_toward"]] += c["severity"] * c["weight"]

    # Context nudges (independent of conflicts)
    if float(row["condition_score"]) / 10.0 < 0.4:
        scores["heavy_discount"] += 0.2 * CONTEXT_WEIGHTS["condition"]

    if int(row["inventory_level"]) > 100:
        scores["heavy_discount"] += 0.15 * CONTEXT_WEIGHTS["inventory"]

    if encode_urgency(row["urgency_level"]) >= 0.75:
        scores["hold_price"] *= 0.5          # urgency penalises holding price

    if float(row["discount_allowed"]) < 0.10:
        scores["heavy_discount"] *= 0.2      # discount not allowed

    # Pick the highest-scoring action
    decision   = max(scores, key=lambda k: scores[k])
    confidence = scores[decision] / (sum(scores.values()) or 1)

    # Build human-readable justification
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    dominant  = sorted_scores[0]
    runner_up = sorted_scores[1] if len(sorted_scores) > 1 else None

    reasons = [
        f"User goal '{goal}' gave '{dominant[0]}' a base score of "
        f"{goal_vector(goal).get(dominant[0], 0):.2f}"
    ]
    for c in [c for c in conflicts if c["pulls_toward"] == decision][:2]:
        reasons.append(
            f"Conflict '{c['type']}' (severity={c['severity']:.2f}, weight={c['weight']}) "
            f"added {c['severity']*c['weight']:.2f} to '{decision}'"
        )
    if runner_up and runner_up[1] / (dominant[1] or 1) > 0.75:
        reasons.append(
            f"Close runner-up: '{runner_up[0]}' scored {runner_up[1]:.2f} "
            f"({runner_up[1]/dominant[1]*100:.0f}% of winner)"
        )

    price_ratio = float(row["price"]) / float(row["avg_market_price"]) if float(row["avg_market_price"]) > 0 else 1.0

    return {
        "decision":            decision,
        "confidence":          round(confidence, 3),
        "conflict_count":      len(conflicts),
        "conflicts_detected":  " | ".join(c["type"] for c in conflicts) if conflicts else "none",
        "dominant_constraint": max(conflicts, key=lambda c: c["severity"]*c["weight"])["type"] if conflicts else "user_goal",
        "why_this_decision":   " • ".join(reasons),
        "parameters_used":     json.dumps({
            "user_goal":        row["user_goal"],
            "urgency_level":    row["urgency_level"],
            "price_vs_market":  f"{price_ratio:.2f}x",
            "competition":      row["competition_level"],
            "demand_score":     round(float(row["demand_score"]), 3),
            "brand_value":      row["brand_value"],
            "condition_score":  int(row["condition_score"]),
            "days_listed":      int(row["days_listed"]),
            "discount_allowed": f"{float(row['discount_allowed']):.0%}",
            "inventory_level":  int(row["inventory_level"]),
        }),
        "action_scores":       json.dumps({k: round(v, 3) for k, v in scores.items()}),
    }

# ──────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────
def main():
    src = Path(__file__).parent / "pricing_dataset.xlsx"

    if not src.exists():
        raise FileNotFoundError(
            f"\n❌  Cannot find: {src}"
            f"\n    Make sure 'pricing_dataset.xlsx' is in the same folder as app.py"
        )

    print(f"✅  Loading dataset: {src}")
    df = pd.read_excel(src, sheet_name="Dataset")
    print(f"✅  Loaded {len(df)} records with columns: {list(df.columns)}")

    print("⚙️   Running conflict resolver...")
    results = df.apply(resolve, axis=1, result_type="expand")
    out     = pd.concat([df, results], axis=1)

    # ── Console preview (5 random samples) ───────────────────
    print("\n" + "=" * 80)
    print(f"{'MULTI-CONTEXT CONFLICT RESOLVER  —  SAMPLE OUTPUT':^80}")
    print("=" * 80)

    for _, r in out.sample(min(5, len(out)), random_state=1).iterrows():
        params = json.loads(r["parameters_used"])
        scores = json.loads(r["action_scores"])
        print(f"\n{'─'*80}")
        print(f"  Record        : {r['record_id']}")
        print(f"  DECISION      : {r['decision'].upper().replace('_',' ')}")
        print(f"  Confidence    : {r['confidence']:.1%}")
        print(f"  Conflicts     : {r['conflict_count']}  →  {r['conflicts_detected']}")
        print(f"  Dominant ctr  : {r['dominant_constraint']}")
        print(f"\n  WHY THIS DECISION:")
        for part in r["why_this_decision"].split(" • "):
            print(f"    • {part}")
        print(f"\n  PARAMETERS USED:")
        for k, v in params.items():
            print(f"    {k:<22}: {v}")
        print(f"\n  ACTION SCORES : {scores}")

    # ── Save to Excel ─────────────────────────────────────────
    out_path = Path(__file__).parent / "final_decisions_output.txt"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Decisions")

        out.groupby("decision").agg(
            count        =("decision",   "count"),
            avg_confidence=("confidence","mean"),
            avg_conflicts =("conflict_count","mean"),
        ).reset_index().to_excel(writer, index=False, sheet_name="Decision_Summary")

        cf = out["conflicts_detected"].str.split(" | ", expand=True).stack()
        cf = cf[cf != "none"].value_counts().rename_axis("conflict_type").reset_index(name="count")
        cf.to_excel(writer, index=False, sheet_name="Conflict_Breakdown")

    print(f"\n{'='*80}")
    print(f"✅  Output saved  →  {out_path}")
    print(f"    Sheets: Decisions | Decision_Summary | Conflict_Breakdown")
    print("=" * 80)

if __name__ == "__main__":
    main()
