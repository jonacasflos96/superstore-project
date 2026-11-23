"""Build report.html: a static copy of the dashboard that needs no server.

A Dash app needs a running server, so the pipeline publishes this HTML file
instead. The charts stay interactive (zoom, hover) and every title is
computed from the data received so far.
"""
from datetime import datetime

import figures as fg

OUTPUT = "report.html"


def build_report(df):
    figs = fg.build_all(df)
    blocks = []
    include_js = True  # embed plotly.js once, so the file also works offline

    def add_chart(fig):
        nonlocal include_js
        blocks.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False

    def add_text(tag, text):
        blocks.append(f"<{tag}>{text}</{tag}>")

    start, end = df["Order Date"].min(), df["Order Date"].max()
    add_text("h1", "Where should Superstore stop discounting?")
    add_text("p", f"Data: {len(df):,} order lines from {start:%b %Y} to {end:%b %Y}. "
                  f"Report generated on {datetime.now():%Y-%m-%d %H:%M}.")

    add_text("h2", "Q1 - Where is the problem?")
    add_chart(figs["scatter"])
    add_chart(figs["loss"])
    add_text("p", figs["loss_note"])

    add_text("h2", "Q2 - What should Superstore do about it?")
    add_chart(figs["bands"])
    add_text("p", figs["bands_note"])
    # No dropdown in a static file: show the groups that already lose at 1-20%
    for sub_category in figs["early_loss_groups"]:
        add_chart(fg.bands_figure(df, sub_category))
    add_chart(figs["heat"])
    add_text("p", figs["heat_note"])
    add_chart(figs["sub_heat"])
    add_text("p", figs["sub_heat_note"])
    add_chart(figs["rescue"])
    add_text("p", figs["rescue_note"])

    add_text("h2", "Machine learning - Which lines will lose money?")
    add_chart(figs["metrics"])
    add_chart(figs["importance"])
    add_chart(figs["risk"])
    add_text("p", figs["ml_note"])

    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Superstore discount report</title>
<style>
  body {{ max-width: 1100px; margin: auto; font-family: Arial, sans-serif;
         color: #222; padding: 24px; }}
  h2 {{ margin-top: 48px; border-bottom: 2px solid #1F5FA8; padding-bottom: 6px; }}
  p {{ font-size: 16px; line-height: 1.5; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


if __name__ == "__main__":
    df = fg.load_data()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(build_report(df))
    print(f"Report saved: {OUTPUT} ({len(df)} lines)")