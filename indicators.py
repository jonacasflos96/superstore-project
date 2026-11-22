import pandas as pd

CLEAN_PATH = "data/clean/superstore_clean.csv"
BAND_ORDER = ["0%", "1-20%", "21-40%", "above 40%"]


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
# Q2 - From which discount level does profit turn negative, and when
#      and where should Superstore test smaller discounts?
# =====================================================================

def _filter(df, sub_category=None):
    """Keep only one sub-category if one is given, otherwise the whole store."""
    return df if sub_category is None else df[df["Sub-Category"] == sub_category]


def margin_by_band(df, sub_category=None):
    """Indicator 6: profit margin in each discount band, in %."""
    data = _filter(df, sub_category)
    totals = data.groupby("Discount Band")[["Profit", "Sales"]].sum()
    return (totals["Profit"] / totals["Sales"] * 100).reindex(BAND_ORDER)


def break_band(df, sub_category=None):
    """Indicator 7: first discount band where the margin turns negative."""
    margins = margin_by_band(df, sub_category)
    negative = margins[margins < 0]
    return negative.index[0] if len(negative) > 0 else None


def units_per_line_by_band(df, sub_category=None):
    """Indicator 8: average units per order line in each discount band."""
    data = _filter(df, sub_category)
    return data.groupby("Discount Band")["Quantity"].mean().reindex(BAND_ORDER)


def seasonality_index(df):
    """Indicator 9: sales of each month / sales of an average month (1 = average)."""
    monthly = df.groupby("Order Month")["Sales"].sum()
    return monthly / monthly.mean()


def average_discount_by_month(df):
    """Support for indicator 9: average discount applied in each month, in %."""
    return df.groupby("Order Month")["Discount"].mean() * 100


def basket_rescue_rate(df):
    """Indicator 10: share of orders with a loss-making discounted line of a
    sub-category that are still profitable overall, in % per sub-category."""
    order_profit = df.groupby("Order ID")["Profit"].sum()
    losing = df[(df["Discount"] > 0) & (df["Profit"] < 0)]
    losing = losing[["Order ID", "Sub-Category"]].drop_duplicates()
    losing["rescued"] = losing["Order ID"].map(order_profit) > 0
    return losing.groupby("Sub-Category")["rescued"].mean() * 100


def single_line_order_share(df):
    """Support for indicator 10: share of orders with only one line, in %."""
    return (df.groupby("Order ID").size() == 1).mean() * 100


# =====================================================================
# Run this file directly to check the results
# =====================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load_clean()

    # ----- Q1 -----
    print("===== Q1 =====")
    print(f"Store margin: {overall_margin(df):.1f}%")
    q1 = q1_table(df)
    print(q1.round(1))

    flagged = q1[q1["flag"]]
    flagged_lines = df[df["Sub-Category"].isin(flagged.index)]
    print("Flagged sub-categories:", list(flagged.index))
    print(f"Their share of sales: {flagged['sales_share_pct'].sum():.0f}%")
    print(f"Their combined margin: {overall_margin(flagged_lines):.1f}%")
    print(f"Their discount-linked loss: ${flagged['discount_loss'].sum():,.0f}")

    # ----- Q2 -----
    print("\n===== Q2 =====")
    print("Margin by band, whole store (%):")
    print(margin_by_band(df).round(1))
    print("Break band, whole store:", break_band(df))

    for sub in ["Storage", "Tables"]:
        print(f"\nMargin by band, {sub} (%):")
        print(margin_by_band(df, sub).round(1))
        print(f"Break band, {sub}:", break_band(df, sub))

    print("\nUnits per line by band:")
    print(units_per_line_by_band(df).round(1))

    print("\nSeasonality index by month (1 = average month):")
    print(seasonality_index(df).round(1))

    print("\nAverage discount by month (%):")
    print(average_discount_by_month(df).round(1))

    print("\nBasket rescue rate of flagged sub-categories (%):")
    print(basket_rescue_rate(df).reindex(flagged.index).round(0))

    print(f"\nOrders with only one line: {single_line_order_share(df):.0f}%")