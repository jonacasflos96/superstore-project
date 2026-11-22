import pandas as pd

CLEAN_PATH = "data/clean/superstore_clean.csv"


def load_clean(path=CLEAN_PATH):
    """Load the clean dataset produced by data_exam.py."""
    return pd.read_csv(path, parse_dates=["Order Date", "Ship Date"])


def overall_margin(df):
    """Profit margin of the given lines, in % (the whole store if df is the full dataset)."""
    return df["Profit"].sum() / df["Sales"].sum() * 100


# =====================================================================
# Q1 - Which sub-categories bring strong sales but weak profit,
#      and how much of their loss is linked to discounts?
# =====================================================================

def sales_share(df):
    """Indicator 1: share of total store sales brought by each sub-category, in %."""
    return df.groupby("Sub-Category")["Sales"].sum() / df["Sales"].sum() * 100


def profit_margin(df):
    """Indicator 2: profit / sales of each sub-category, in %."""
    totals = df.groupby("Sub-Category")[["Profit", "Sales"]].sum()
    return totals["Profit"] / totals["Sales"] * 100


def average_discount(df):
    """Indicator 3: average discount applied in each sub-category, in %."""
    return df.groupby("Sub-Category")["Discount"].mean() * 100


def discount_linked_loss(df):
    """Indicator 4: $ lost on lines that were discounted AND lost money, per sub-category."""
    losing = df[(df["Discount"] > 0) & (df["Profit"] < 0)]
    loss = -losing.groupby("Sub-Category")["Profit"].sum()
    return loss.reindex(df["Sub-Category"].unique(), fill_value=0)


def q1_table(df):
    """All Q1 indicators in one table, plus indicator 5: the flag."""
    table = pd.DataFrame({
        "sales": df.groupby("Sub-Category")["Sales"].sum(),
        "sales_share_pct": sales_share(df),
        "margin_pct": profit_margin(df),
        "avg_discount_pct": average_discount(df),
        "discount_loss": discount_linked_loss(df),
    })
    # Flag: strong sales (at or above the median) AND margin below the store's
    table["flag"] = (table["sales"] >= table["sales"].median()) & (
        table["margin_pct"] < overall_margin(df)
    )
    return table.sort_values("sales", ascending=False)


# =====================================================================
# Run this file directly to check the results
# =====================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load_clean()

    print(f"Store margin: {overall_margin(df):.1f}%")

    q1 = q1_table(df)
    print(q1.round(1))

    flagged = q1[q1["flag"]]
    flagged_lines = df[df["Sub-Category"].isin(flagged.index)]
    print("Flagged sub-categories:", list(flagged.index))
    print(f"Their share of sales: {flagged['sales_share_pct'].sum():.0f}%")
    print(f"Their combined margin: {overall_margin(flagged_lines):.1f}%")
    print(f"Their discount-linked loss: ${flagged['discount_loss'].sum():,.0f}")