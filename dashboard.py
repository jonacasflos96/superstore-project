from dash import Dash, Input, Output, dcc, html

import figures as fg

df = fg.load_data()
figs = fg.build_all(df)
dropdown_options = ["Whole store"] + sorted(df["Sub-Category"].unique())

app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Where should Superstore stop discounting?"),

        html.H2("Q1 - Where is the problem?"),
        dcc.Graph(figure=figs["scatter"]),
        dcc.Graph(figure=figs["loss"]),
        html.P(figs["loss_note"]),

        html.H2("Q2 - What should Superstore do about it?"),
        html.Label("Show discount bands for:"),
        dcc.Dropdown(id="band-dropdown", options=dropdown_options,
                     value="Whole store", clearable=False),
        dcc.Graph(id="bands-chart"),
        html.P(id="bands-note"),
        dcc.Graph(figure=figs["heat"]),
        html.P(figs["heat_note"]),
        dcc.Graph(figure=figs["rescue"]),
        html.P(figs["rescue_note"]),

        html.H2("Machine learning - Which lines will lose money?"),
        dcc.Graph(figure=figs["metrics"]),
        dcc.Graph(figure=figs["importance"]),
        dcc.Graph(figure=figs["risk"]),
        html.P(figs["ml_note"]),
    ],
)


@app.callback(
    Output("bands-chart", "figure"),
    Output("bands-note", "children"),
    Input("band-dropdown", "value"),
)
def update_bands(choice):
    """Redraw the bands chart when another sub-category is picked."""
    sub_category = None if choice == "Whole store" else choice
    return fg.bands_figure(df, sub_category), fg.bands_note(df, sub_category)


if __name__ == "__main__":
    app.run(debug=True)