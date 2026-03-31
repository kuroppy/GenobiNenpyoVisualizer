import pandas as pd
import numpy as np

from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import plotly.graph_objects as go
import requests
from io import StringIO

# =========================
# Google Spreadsheet TSV URLs
# =========================
TOPICS_URL = "https://docs.google.com/spreadsheets/d/1WDctFI3ZhAtOs7pu9mM0m_9fYxRHtQX0wozEezuhkQY/export?format=tsv"
BACKGROUND_URL = "https://docs.google.com/spreadsheets/d/1Gh96Advt9QutEyxYMW8dkcLBQ35AhlMDrnt3BgKym30/export?format=tsv"
DATA_URL = "https://docs.google.com/spreadsheets/d/1CaCvS8ladr0VzVxZJsds8U4qGWjYThi_mUmWqHvyfiQ/export?format=tsv"
TOPIC_COMP_URL = "https://docs.google.com/spreadsheets/d/1Fkw1XAEe8cXFog8O7DHDUpL2W266MugjHtY7jqr1bbI/export?format=tsv"


def read_tsv(url: str) -> pd.DataFrame:
    url = url.strip()
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.content.decode("utf-8-sig")), sep="\t")



def normalize_data_df(df: pd.DataFrame) -> pd.DataFrame:
    df["paper_id"] = df["paper_id"].astype(str)

    numeric_cols = [c for c in ["UMAP1", "UMAP2", "main_topic", "Publication_Year", "Cited_by_count"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def normalize_topic_df(df: pd.DataFrame) -> pd.DataFrame:
    df["topic_id"] = df["topic_id"].astype(str)
    for c in ["UMAP1", "UMAP2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_background_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["UMAP1", "UMAP2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_topic_comp_df(df: pd.DataFrame) -> pd.DataFrame:
    df["paper_id"] = df["paper_id"].astype(str)
    topic_cols_local = [c for c in df.columns if c.startswith("topic_")]
    for c in topic_cols_local:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


topics_df = normalize_topic_df(read_tsv(TOPICS_URL))
background_df = normalize_background_df(read_tsv(BACKGROUND_URL))
data_df = normalize_data_df(read_tsv(DATA_URL))
topic_comp_df = normalize_topic_comp_df(read_tsv(TOPIC_COMP_URL))

merged_df = data_df.merge(topic_comp_df, on="paper_id", how="left")

topic_cols = [c for c in topic_comp_df.columns if c.startswith("topic_")]
topic_id_set = set(topics_df["topic_id"].tolist())
topic_cols = [c for c in topic_cols if c in topic_id_set]

for c in topic_cols:
    if c in merged_df.columns:
        merged_df[c] = merged_df[c].fillna(0.0)

topic_info = topics_df.set_index("topic_id").to_dict("index")
table_cols = [c for c in merged_df.columns if c not in topic_cols]

paper_hover_text = merged_df["title"].fillna(merged_df["paper_id"]).astype(str).tolist()
topic_hover_text = topics_df["theme"].fillna(topics_df["topic_id"]).astype(str).tolist()


def topic_marker_sizes(selected_paper_id: str | None, base_size=14, scale=42, min_size=8):
    if not selected_paper_id:
        return [base_size] * len(topics_df)

    row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
    if row.empty:
        return [base_size] * len(topics_df)

    row = row.iloc[0]
    sizes = []
    for topic_id in topics_df["topic_id"]:
        weight = float(row.get(topic_id, 0.0))
        sizes.append(min_size + scale * np.sqrt(max(weight, 0.0)))
    return sizes


def topic_text_sizes(selected_paper_id: str | None, base=11, scale=8):
    if not selected_paper_id:
        return [base] * len(topics_df)

    row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
    if row.empty:
        return [base] * len(topics_df)

    row = row.iloc[0]
    sizes = []
    for topic_id in topics_df["topic_id"]:
        weight = float(row.get(topic_id, 0.0))
        sizes.append(base + scale * np.sqrt(max(weight, 0.0)))
    return sizes


def paper_marker_sizes(selected_topic_id: str | None, base_size=8, scale=38, min_size=5):
    if not selected_topic_id or selected_topic_id not in merged_df.columns:
        return [9.0] * len(merged_df)

    vals = merged_df[selected_topic_id].fillna(0.0).astype(float).values
    return [min_size + scale * np.sqrt(max(v, 0.0)) for v in vals]


def make_topic_bar_figure(selected_paper_id: str | None):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=35, b=20),
        title="Topic composition",
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    if not selected_paper_id:
        fig.add_annotation(
            text="No paper selected",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )
        return fig

    row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
    if row.empty:
        fig.add_annotation(
            text="Selected paper not found",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )
        return fig

    row = row.iloc[0]
    weights = pd.Series({t: float(row.get(t, 0.0)) for t in topic_cols}).sort_values(ascending=False).head(10)

    x_labels = []
    for tid in weights.index:
        theme = topic_info.get(tid, {}).get("theme", "")
        label = f"{tid}"
        if theme:
            label += f"<br>{theme}"
        x_labels.append(label)

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=weights.values,
            hovertemplate="%{x}<br>weight=%{y:.3f}<extra></extra>"
        )
    )
    fig.update_yaxes(title="weight")
    return fig


def build_detail_panel(selected_paper_id: str | None, selected_topic_id: str | None):
    blocks = []

    # -------------------------
    # Paper info
    # -------------------------
    if selected_paper_id:

        row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]

        if not row.empty:

            row = row.iloc[0]

            title = row.get("title", "")
            event = row.get("event", "")
            year = row.get("Publication_Year", "")
            journal = row.get("journal", "")
            doi = row.get("DOI", "")
            authors = row.get("Author", "")

            blocks.append(
                html.Div(
                    [
                        html.H4(title, style={"marginBottom": "8px"}),

                        html.Div(
                            [
                                html.Span(journal),
                                html.Span(f" ({int(year)})" if pd.notna(year) else "")
                            ],
                            className="detail-block"
                        ),

                        html.Div(event, className="detail-block"),

                        html.Div(
                            [
                                html.B("DOI: "),
                                html.A(
                                    doi,
                                    href=doi,
                                    target="_blank",
                                    style={"color": "#2563eb"}
                                )
                            ],
                            className="detail-block"
                        ),

                        html.Div(
                            [
                                html.B("Authors"),
                                html.Div(
                                    authors.replace(";", "; "),
                                    style={
                                        "marginTop": "4px",
                                        "fontSize": "13px",
                                        "lineHeight": "1.5",
                                        "color": "#374151"
                                    }
                                )
                            ],
                            style={"marginTop": "10px"}
                        )
                    ],
                    className="card",
                )
            )

    # -------------------------
    # Topic info
    # -------------------------
    if selected_topic_id:

        info = topic_info.get(selected_topic_id, {})

        theme = info.get("theme", "")
        keywords = str(info.get("keywords", ""))

        keyword_items = [k.strip() for k in keywords.split(",") if k.strip()]

        blocks.append(
            html.Div(
                [
                    html.H4("Topic"),

                    html.Div(
                        theme,
                        style={
                            "fontWeight": "600",
                            "marginBottom": "8px"
                        }
                    ),

                    html.Div(
                        [
                            html.B("Keywords"),
                            html.Ul(
                                [html.Li(k) for k in keyword_items],
                                style={
                                    "marginTop": "6px",
                                    "paddingLeft": "18px"
                                }
                            )
                        ]
                    )
                ],
                className="card",
                style={"marginTop": "12px"}
            )
        )

    if not blocks:
        return html.Div(
            "No paper or topic selected.",
            className="helper-text"
        )

    return html.Div(blocks)


def make_figure(selected_paper_id: str | None, selected_topic_id: str | None):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=background_df["UMAP1"],
            y=background_df["UMAP2"],
            mode="markers",
            name="background",
            marker=dict(size=5, opacity=0.18, color="#cfd4dc"),
            hoverinfo="skip",
            showlegend=False
        )
    )

    base_paper_sizes = np.array(paper_marker_sizes(selected_topic_id), dtype=float)
    paper_sizes = base_paper_sizes.copy()
    paper_opacity = np.full(len(merged_df), 0.82)
    paper_line_width = np.zeros(len(merged_df))

    if selected_paper_id:
        paper_opacity[:] = 0.22
        selected_mask = merged_df["paper_id"] == selected_paper_id
        paper_sizes[selected_mask] = np.maximum(paper_sizes[selected_mask], 18.0)
        paper_opacity[selected_mask] = 1.0
        paper_line_width[selected_mask] = 2.5

    paper_customdata = merged_df["paper_id"].astype(str)

    fig.add_trace(
        go.Scattergl(
            x=merged_df["UMAP1"],
            y=merged_df["UMAP2"],
            mode="markers",
            name="papers",
            marker=dict(
                size=paper_sizes,
                opacity=paper_opacity,
                line=dict(width=paper_line_width, color="#111827"),
                color="#3b82f6"
            ),
            text=paper_hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=paper_customdata,
            showlegend=False
        )
    )

    t_sizes = topic_marker_sizes(selected_paper_id)
    text_sizes = topic_text_sizes(selected_paper_id)

    fig.add_trace(
        go.Scatter(
            x=topics_df["UMAP1"],
            y=topics_df["UMAP2"],
            mode="markers+text",
            name="topics",
            text=topics_df["theme"],
            textposition="top center",
            marker=dict(
                size=t_sizes,
                opacity=0.95,
                color="#ef4444",
                line=dict(width=1.2, color="#991b1b"),
                symbol="circle"
            ),
            textfont=dict(size=12, color="#374151"),
            texttemplate="%{text}",
            textfont_size=text_sizes,
            hovertext=topic_hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            customdata=topics_df["topic_id"].astype(str),
            showlegend=False
        )
    )

    fig.update_layout(
    template="plotly_white",
    height=850,
    margin=dict(l=20, r=20, t=50, b=20),
    title="UMAP Explorer",
    clickmode="event+select",
    paper_bgcolor="white",
    plot_bgcolor="white",

    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        title=""
    ),

    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        title=""
    )
)
    return fig


app = Dash(__name__)
server = app.server
table_cols = ["paper_id", "Publication_Year", "title", "journal"]

app.layout = html.Div(
    [
        html.H2("UMAP Explorer with Topics and Metadata"),

        dcc.Store(id="selected-paper-id", data=None),
        dcc.Store(id="selected-topic-id", data=None),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="umap-plot", figure=make_figure(None, None), config={"displaylogo": False})],
                            className="card"
                        )
                    ],
                    className="left-panel",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Selection"),
                                html.Div(
                                    id="detail-panel",
                                    className="detail-block"
                                ),
                            ],
                            className="card card-muted",
                        ),
                        html.Div(
                            [
                                html.H3("Topic composition"),
                                dcc.Graph(id="topic-bar-plot", figure=make_topic_bar_figure(None), config={"displaylogo": False}),
                                html.Div(
                                    "Paper を選ぶと topic の組成を表示します。Topic を選ぶと関連 keyword を表示します。",
                                    className="helper-text"
                                )
                            ],
                            className="card",
                        ),
                    ],
                    className="right-panel",
                ),
            ],
            className="top-row",
        ),

        html.Div(
            [
                html.H3("Metadata table"),

                dash_table.DataTable(
                    id="meta-table",

                    columns=[
                        {"name": "paper_id", "id": "paper_id", "hidden": True},
                        {"name": "Year", "id": "Publication_Year"},
                        {"name": "Title", "id": "title"},
                        {"name": "Journal", "id": "journal"},
                    ],

                    data=merged_df.sort_values("Publication_Year")[table_cols].to_dict("records"),

                    row_selectable="single",
                    selected_rows=[],

                    filter_action="native",

                    sort_action="native",
                    sort_mode="multi",
                    sort_by=[{"column_id": "Publication_Year", "direction": "asc"}],

                    page_action="none",

                    fixed_rows={"headers": True},

                    style_table={
                        "height": "900px",
                        "overflowY": "auto",
                        "overflowX": "auto",
                    },

                    style_cell={
                        "textAlign": "left",
                        "padding": "8px",
                        "fontSize": 13,
                        "whiteSpace": "normal",
                        "height": "auto",
                        "lineHeight": "1.5",
                        "border": "none",
                    },

                    style_header={
                        "fontWeight": "bold",
                        "border": "none",
                        "backgroundColor": "white",
                    },

                    style_cell_conditional=[
                        {
                            "if": {"column_id": "title"},
                            "minWidth": "420px",
                            "maxWidth": "700px",
                        },
                        {
                            "if": {"column_id": "Publication_Year"},
                            "width": "80px",
                        },
                        {
                            "if": {"column_id": "journal"},
                            "minWidth": "160px",
                            "maxWidth": "260px",
                        },
                    ],
                ),
            ],
            className="table-card section-gap",
        ),
    ],
    className="app-container",
)


@app.callback(
    Output("selected-paper-id", "data"),
    Output("selected-topic-id", "data"),
    Input("umap-plot", "clickData"),
    Input("meta-table", "derived_virtual_selected_rows"),
    State("meta-table", "derived_virtual_data"),
    State("selected-paper-id", "data"),
    State("selected-topic-id", "data"),
    prevent_initial_call=True,
)
def update_selection(clickData, selected_rows, virtual_rows, current_paper, current_topic):
    trigger = ctx.triggered_id

    if trigger == "umap-plot":
        if clickData and "points" in clickData and len(clickData["points"]) > 0:
            point = clickData["points"][0]
            curve_number = point.get("curveNumber", None)
            customdata = point.get("customdata", None)

            if curve_number == 1 and customdata is not None:
                return str(customdata), None

            if curve_number == 2 and customdata is not None:
                return None, str(customdata)

        return current_paper, current_topic

    if trigger == "meta-table":
        if selected_rows and virtual_rows:
            idx = selected_rows[0]
            if 0 <= idx < len(virtual_rows):
                return str(virtual_rows[idx]["paper_id"]), None
        return current_paper, current_topic

    return current_paper, current_topic


@app.callback(
    Output("umap-plot", "figure"),
    Output("detail-panel", "children"),
    Output("topic-bar-plot", "figure"),
    Output("meta-table", "selected_rows"),
    Input("selected-paper-id", "data"),
    Input("selected-topic-id", "data"),
    State("meta-table", "derived_virtual_data"),
)
def refresh_view(selected_paper_id, selected_topic_id, virtual_rows):
    fig = make_figure(selected_paper_id, selected_topic_id)
    detail = build_detail_panel(selected_paper_id, selected_topic_id)
    bar_fig = make_topic_bar_figure(selected_paper_id)

    selected_rows = []
    if selected_paper_id and virtual_rows:
        for i, row in enumerate(virtual_rows):
            if str(row.get("paper_id")) == str(selected_paper_id):
                selected_rows = [i]
                break

    return fig, detail, bar_fig, selected_rows


if __name__ == "__main__":
    app.run(debug=True)
