import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.express as px

import indicators as ind
import ml_model

BLUE = "#1F5FA8"
ORANGE = "#D9822B"
GREY = "#A0A7B4"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_k(value):
    """Format a dollar amount as +$108.1k / -$78.1k."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value) / 1000:.1f}k"


# =====================================================================
# Data
# =====================================================================
df = ind.add_line_indicators(ind.load_clean())
q1 = ind.q1_table(df).reset_index()
store_margin = ind.overall_margin(df)

flagged = q1[q1["flag"]]
flagged_lines = df[df["Flagged Sub-Category"]]
n_flagged = len(flagged)
flagged_share = flagged["sales_share_pct"].sum()
flagged_margin = ind.overall_margin(flagged_lines)
flagged_loss = flagged["discount_loss"].sum()

q1["group"] = q1["flag"].map({True: "Flagged", False: "Other sub-categories"})
COLORS = {"Flagged": BLUE, "Other sub-categories": GREY}

# =====================================================================
# Q1 - Chart 1: sales share vs margin
# =====================================================================
median_share = q1["sales_share_pct"].median()

fig_scatter = px.scatter(
    q1, x="sales_share_pct", y="margin_pct", text="Sub-Category",
    color="group", color_discrete_map=COLORS,
    labels={"sales_share_pct": "Share of store sales (%)",
            "margin_pct": "Profit margin (%)", "group": ""},
    title=(f"{n_flagged} of {len(q1)} sub-categories bring "
           f"{flagged_share:.0f}% of sales at a {flagged_margin:.1f}% margin"),
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

# =====================================================================
# Q1 - Chart 2: discount-linked loss
# =====================================================================
fig_loss = px.bar(
    q1, x="discount_loss", y="Sub-Category",
    orientation="h", color="group", color_discrete_map=COLORS,
    labels={"discount_loss": "$ lost on discounted lines that lose money",
            "Sub-Category": "", "group": ""},
    title=f"Flagged sub-categories lose ${flagged_loss / 1000:.1f}k on loss-making discounted lines",
)
# Largest loss at the top, whatever the colour
fig_loss.update_yaxes(categoryorder="total ascending")

upto_20 = flagged_lines[flagged_lines["Discount"] <= 0.2]["Profit"].sum()
above_20 = flagged_lines[flagged_lines["Discount"] > 0.2]["Profit"].sum()
binders = q1.set_index("Sub-Category").loc["Binders"]

loss_note = (
    f"On lines discounted up to 20%, the flagged groups earn {fmt_k(upto_20)}; "
    f"above 20%, they make {fmt_k(above_20)}. "
    f"Binders loses the most (${binders['discount_loss'] / 1000:.1f}k) but is not flagged: "
    f"its overall margin ({binders['margin_pct']:.1f}%) stays above the store's."
)

# =====================================================================
# Q2 - Chart 3: margin by discount band (updated by the dropdown)
# =====================================================================
def bands_figure(sub_category=None):
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


def bands_note(sub_category=None):
    """Sentence on units per line, shown under the bands chart."""
    units = ind.units_per_line_by_band(df, sub_category).dropna()
    return (f"Units per order line range from {units.min():.1f} to {units.max():.1f} "
            f"across discount bands: deeper discounts do not sell more units per line.")


dropdown_options = ["Whole store"] + sorted(df["Sub-Category"].unique())

# =====================================================================
# Q2 - Chart 4: seasonality heatmap (year x month)
# =====================================================================
year_month = df.groupby(["Order Year", "Order Month"])["Sales"].sum().unstack()
year_month_index = year_month.div(year_month.mean(axis=1), axis=0)
year_month_index.columns = MONTH_NAMES
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


# =====================================================================
# Q2 - Chart 4b: seasonality of each flagged sub-category
# =====================================================================
sub_season = ind.seasonality_by_subcategory(df, flagged["Sub-Category"])
sub_season.columns = [MONTH_NAMES[m - 1] for m in sub_season.columns]
sub_peaks = sub_season.idxmax(axis=1)

fig_sub_heat = px.imshow(
    sub_season, text_auto=".1f", aspect="auto",
    color_continuous_scale="RdBu", color_continuous_midpoint=1,
    labels={"x": "Month", "y": "", "color": "Sales vs its average month"},
    title="Each flagged sub-category has its own busy months (blue = above its average month)",
)

sub_heat_note = (
    "Peak month of each flagged sub-category: "
    + ", ".join(f"{sub} in {month}" for sub, month in sub_peaks.items())
    + ". A spike can come from a few very large orders, so check it before planning a test."
)


# =====================================================================
# Q2 - Chart 5: basket rescue rate of the flagged sub-categories
# =====================================================================
rescue = ind.basket_rescue_rate(df).reindex(flagged["Sub-Category"])
rescue = rescue.rename("rate").reset_index().sort_values("rate")
rescue["label"] = rescue["rate"].round(0).astype(int).astype(str) + "%"
lowest = rescue.iloc[0]
highest = rescue.iloc[-1]

fig_rescue = px.bar(
    rescue, x="rate", y="Sub-Category", orientation="h", text="label",
    color_discrete_sequence=[BLUE],
    labels={"rate": "Orders still profitable overall (%)", "Sub-Category": ""},
    title=(f"Few losing discounts are rescued by the basket: "
           f"{lowest['rate']:.0f}% ({lowest['Sub-Category']}) to "
           f"{highest['rate']:.0f}% ({highest['Sub-Category']})"),
)

rescue_note = (
    f"{ind.single_line_order_share(df):.0f}% of orders have only one line, "
    f"so no basket effect is possible for them."
)

# =====================================================================
# Layout
# =====================================================================
# =====================================================================
# Machine learning - loss-risk model
# =====================================================================
ml = ml_model.run_model(df)
ml_results = ml["results"]
tree_recall = ml_results.loc["recall", "Decision tree"]
rule_recall = ml_results.loc["recall", "Rule: discount > 20%"]

# Chart 6: model vs rule
metrics_long = ml_results.rename_axis("metric").reset_index().melt(
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

app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Where should Superstore stop discounting?"),

        html.H2("Q1 - Where is the problem?"),
        dcc.Graph(figure=fig_scatter),
        dcc.Graph(figure=fig_loss),
        html.P(loss_note),

        html.H2("Q2 - What should Superstore do about it?"),
        html.Label("Show discount bands for:"),
        dcc.Dropdown(id="band-dropdown", options=dropdown_options,
                     value="Whole store", clearable=False),
        dcc.Graph(id="bands-chart"),
        html.P(id="bands-note"),
        dcc.Graph(figure=fig_heat),
        html.P(heat_note),
        dcc.Graph(figure=fig_sub_heat),
        html.P(sub_heat_note),
        dcc.Graph(figure=fig_rescue),
        html.P(rescue_note),
        html.H2("Machine learning - Which lines will lose money?"),
        dcc.Graph(figure=fig_metrics),
        dcc.Graph(figure=fig_importance),
        dcc.Graph(figure=fig_risk),
        html.P(ml_note),
    ],
)


# =====================================================================
# Callback: update the bands chart when the dropdown changes
# =====================================================================
@app.callback(
    Output("bands-chart", "figure"),
    Output("bands-note", "children"),
    Input("band-dropdown", "value"),
)
def update_bands(choice):
    sub_category = None if choice == "Whole store" else choice
    return bands_figure(sub_category), bands_note(sub_category)


if __name__ == "__main__":
    app.run(debug=True)