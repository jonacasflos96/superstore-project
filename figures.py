"""Charts and story sentences for the dashboard and the report.

Every title and sentence is computed from the data, so the story updates by
itself when a new batch arrives. Used by dashboard.py and export_report.py.
"""
import pandas as pd
import plotly.express as px

import indicators as ind
import ml_model

BLUE = "#1F5FA8"
ORANGE = "#D9822B"
GREY = "#A0A7B4"
COLORS = {"Flagged": BLUE, "Other sub-categories": GREY}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_k(value):
    """Format a dollar amount as +$108.1k / -$78.1k."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value) / 1000:.1f}k"


def load_data():
    """Clean dataset with the line-level indicator columns."""
    return ind.add_line_indicators(ind.load_clean())


# =====================================================================
# Q1 - Where is the problem?
# =====================================================================
def q1_figures(df):
    q1 = ind.q1_table(df).reset_index()
    q1["group"] = q1["flag"].map({True: "Flagged", False: "Other sub-categories"})
    store_margin = ind.overall_margin(df)

    flagged = q1[q1["flag"]]
    flagged_lines = df[df["Flagged Sub-Category"]]

    # Chart 1: sales share vs margin
    median_share = q1["sales_share_pct"].median()
    fig_scatter = px.scatter(
        q1, x="sales_share_pct", y="margin_pct", text="Sub-Category",
        color="group", color_discrete_map=COLORS,
        labels={"sales_share_pct": "Share of store sales (%)",
                "margin_pct": "Profit margin (%)", "group": ""},
        title=(f"{len(flagged)} of {len(q1)} sub-categories bring "
               f"{flagged['sales_share_pct'].sum():.0f}% of sales at a "
               f"{ind.overall_margin(flagged_lines):.1f}% margin"),
    )
    fig_scatter.update_traces(textposition="top center", marker_size=12)
    # Blue zone: strong sales AND margin below the store's
    fig_scatter.add_shape(
        type="rect", x0=median_share, x1=q1["sales_share_pct"].max() + 1,
        y0=q1["margin_pct"].min() - 5, y1=store_margin,
        fillcolor=BLUE, opacity=0.08, line_width=0, layer="below",
    )
    fig_scatter.add_hline(y=store_margin, line_dash="dash",
                          annotation_text=f"Store margin {store_margin:.1f}%",
                          annotation_position="bottom left")
    fig_scatter.add_vline(x=median_share, line_dash="dot",
                          annotation_text="Median sales")

    # Chart 2: discount-linked loss
    fig_loss = px.bar(
        q1, x="discount_loss", y="Sub-Category",
        orientation="h", color="group", color_discrete_map=COLORS,
        labels={"discount_loss": "$ lost on discounted lines that lose money",
                "Sub-Category": "", "group": ""},
        title=(f"Flagged sub-categories lose ${flagged['discount_loss'].sum() / 1000:.1f}k "
               f"on loss-making discounted lines"),
    )
    fig_loss.update_yaxes(categoryorder="total ascending")

    upto_20 = flagged_lines[flagged_lines["Discount"] <= 0.2]["Profit"].sum()
    above_20 = flagged_lines[flagged_lines["Discount"] > 0.2]["Profit"].sum()

    # The sub-category with the largest loss, whichever it is
    top = q1.sort_values("discount_loss").iloc[-1]
    if top["flag"]:
        top_sentence = (f"{top['Sub-Category']} loses the most "
                        f"(${top['discount_loss'] / 1000:.1f}k).")
    else:
        top_sentence = (f"{top['Sub-Category']} loses the most "
                        f"(${top['discount_loss'] / 1000:.1f}k) but is not flagged: "
                        f"its overall margin ({top['margin_pct']:.1f}%) stays above the store's.")

    loss_note = (f"On lines discounted up to 20%, the flagged groups earn {fmt_k(upto_20)}; "
                 f"above 20%, they make {fmt_k(above_20)}. {top_sentence}")

    return {"scatter": fig_scatter, "loss": fig_loss, "loss_note": loss_note}


# =====================================================================
# Q2 - What should Superstore do about it?
# =====================================================================
def bands_figure(df, sub_category=None):
    """Bar chart of the margin in each discount band."""
    margins = ind.margin_by_band(df, sub_category)
    data = margins.rename("margin").reset_index()
    data["label"] = ["no lines" if pd.isna(m) else f"{m:.1f}%" for m in data["margin"]]
    data["margin"] = data["margin"].fillna(0)
    data["result"] = ["Loss" if m < 0 else "Profit" for m in data["margin"]]

    name = sub_category or "the whole store"
    first_loss = ind.break_band(df, sub_category)
    if first_loss:
        title = f"For {name}, profit turns negative in the {first_loss} discount band"
    else:
        title = f"For {name}, profit stays positive in every discount band"

    fig = px.bar(
        data, x="Discount Band", y="margin", text="label", color="result",
        color_discrete_map={"Profit": BLUE, "Loss": ORANGE},
        category_orders={"Discount Band": ind.BAND_ORDER},
        labels={"margin": "Profit margin (%)", "Discount Band": "Discount band", "result": ""},
        title=title,
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    return fig


def bands_note(df, sub_category=None):
    """Sentence on units per line, shown under the bands chart."""
    units = ind.units_per_line_by_band(df, sub_category).dropna()
    return (f"Units per order line range from {units.min():.1f} to {units.max():.1f} "
            f"across discount bands: deeper discounts do not sell more units per line.")


def q2_figures(df):
    flagged_names = list(ind.q1_table(df).query("flag").index)

    # Flagged groups that already lose money at 1-20% (e.g. Storage, Tables)
    early_loss = [s for s in flagged_names if ind.break_band(df, s) == "1-20%"]

    # Chart 4: seasonality heatmap (year x month)
    year_month = df.groupby(["Order Year", "Order Month"])["Sales"].sum().unstack()
    year_month_index = year_month.div(year_month.mean(axis=1), axis=0)
    year_month_index.columns = [MONTH_NAMES[m - 1] for m in year_month_index.columns]
    year_month_index.index = year_month_index.index.astype(str)

    season = ind.seasonality_index(df)
    peaks = season.nlargest(3).sort_index()
    peak_names = ", ".join(MONTH_NAMES[m - 1] for m in peaks.index)
    discount_by_month = ind.average_discount_by_month(df)

    fig_heat = px.imshow(
        year_month_index, text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
        labels={"x": "Month", "y": "Year", "color": "Sales vs average month"},
        title=f"Sales peak in {peak_names}: {peaks.min():.1f} to {peaks.max():.1f} times an average month",
    )
    heat_note = (
        f"Yet the average discount stays between {discount_by_month.min():.1f}% and "
        f"{discount_by_month.max():.1f}% every month: Superstore does not discount less "
        f"when demand is high. Peak months are the best time to test smaller discounts."
    )

    # Chart 5: basket rescue rate (flagged groups, or all if none is flagged)
    groups = flagged_names or sorted(df["Sub-Category"].unique())
    rescue = ind.basket_rescue_rate(df).reindex(groups).dropna()
    rescue = rescue.rename("rate").rename_axis("Sub-Category").reset_index().sort_values("rate")
    rescue["label"] = rescue["rate"].round(0).astype(int).astype(str) + "%"
    lowest, highest = rescue.iloc[0], rescue.iloc[-1]

    fig_rescue = px.bar(
        rescue, x="rate", y="Sub-Category", orientation="h", text="label",
        color_discrete_sequence=[BLUE],
        labels={"rate": "Orders still profitable overall (%)", "Sub-Category": ""},
        title=(f"Few losing discounts are rescued by the basket: "
               f"{lowest['rate']:.0f}% ({lowest['Sub-Category']}) to "
               f"{highest['rate']:.0f}% ({highest['Sub-Category']})"),
    )
    rescue_note = (f"{ind.single_line_order_share(df):.0f}% of orders have only one line, "
                   f"so no basket effect is possible for them.")

    return {"early_loss_groups": early_loss,
            "heat": fig_heat, "heat_note": heat_note,
            "rescue": fig_rescue, "rescue_note": rescue_note}


# =====================================================================
# Machine learning - loss-risk model
# =====================================================================
def ml_figures(df):
    ml = ml_model.run_model(df)
    results = ml["results"]
    tree_recall = results.loc["recall", "Decision tree"]
    rule_recall = results.loc["recall", "Rule: discount > 20%"]

    # Chart 6: model vs rule
    metrics_long = results.rename_axis("metric").reset_index().melt(
        id_vars="metric", var_name="method", value_name="score"
    )
    fig_metrics = px.bar(
        metrics_long, x="metric", y="score", color="method", barmode="group",
        text_auto=".1f", color_discrete_sequence=[BLUE, GREY],
        labels={"metric": "", "score": "Score on losing lines (%)", "method": ""},
        title=(f"The decision tree catches {tree_recall:.1f}% of losing lines, "
               f"vs {rule_recall:.1f}% for the 20% rule"),
    )

    # Chart 7: what drives the model
    importance = ml["importance"].rename_axis("feature").reset_index(name="importance")
    split_feature, split_threshold = ml["first_split"]
    fig_importance = px.bar(
        importance, x="importance", y="feature", orientation="h",
        text_auto=".1f", color_discrete_sequence=[BLUE],
        labels={"importance": "Share of the tree's decisions (%)", "feature": ""},
        title=(f"{importance.iloc[0]['feature']} carries {importance.iloc[0]['importance']:.0f}% "
               f"of the decisions; first split at {split_feature} <= {split_threshold:.0%}"),
    )
    fig_importance.update_yaxes(categoryorder="total ascending")

    # Chart 8: loss-risk heatmap
    fig_risk = px.imshow(
        ml["risk_table"], text_auto=".0f", color_continuous_scale="Oranges",
        zmin=0, zmax=100, aspect="auto",
        labels={"x": "Discount band", "y": "", "color": "Loss risk (%)"},
        title="Predicted loss risk (%) by sub-category and discount band",
    )

    ml_note = (
        f"Tested on {ml['test_lines']} lines the model never saw. Only "
        f"{ml['loss_share']:.0f}% of lines lose money, so we report precision, recall and F1 "
        f"on losing lines instead of accuracy. Empty cells: no sales in that band."
    )

    return {"metrics": fig_metrics, "importance": fig_importance,
            "risk": fig_risk, "ml_note": ml_note}


# =====================================================================
# Everything at once
# =====================================================================
def build_all(df):
    """All charts and sentences, in one dictionary."""
    out = {"bands": bands_figure(df), "bands_note": bands_note(df)}
    out.update(q1_figures(df))
    out.update(q2_figures(df))
    out.update(ml_figures(df))
    return out