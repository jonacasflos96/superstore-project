from dash import Dash, dcc, html
import plotly.express as px

import indicators as ind


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
COLORS = {"Flagged": "#1F5FA8", "Other sub-categories": "#A0A7B4"}

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
    fillcolor="#1F5FA8", opacity=0.08, line_width=0, layer="below",
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
# Layout
# =====================================================================
app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Where should Superstore stop discounting?"),
        html.H2("Q1 - Where is the problem?"),
        dcc.Graph(figure=fig_scatter),
        dcc.Graph(figure=fig_loss),
        html.P(loss_note),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)