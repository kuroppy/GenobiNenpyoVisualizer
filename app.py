"""
ゲノム微生物学の地図 — Genome Microbiology Topic Atlas
=====================================================

ゲノビ年表に収録された論文群を、トピックモデル（LDA）と UMAP で
二次元空間に配置し、分野の構造を探索できるインタラクティブ Web アプリ。

このバージョンでは、見た目とインタラクションを「研究アトラス（星図）」の
コンセプトで刷新しています。スタイルはすべて app.py 内に同梱しているため、
このファイル単体で動作します（旧 assets/style.css は不要です）。

主な変更点
----------
- 暗い「観測ウィンドウ」上に光点として論文・トピックを描画
- 凡例・操作パネル・選択状態をアプリ内に常設
- 論文の配色を「均一 / 出版年 / 被引用数」で切り替え可能
- 出版年スライダーで年表論文を絞り込み
- 選択クリアボタン、ホバー表示・表・トピック組成グラフの刷新
"""

import os
from io import StringIO

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update

# =========================================================================
# Design tokens — keep Plotly colours in sync with the embedded CSS below
# White-background map: amber landmarks, teal papers, faint grey nebula.
# =========================================================================
C = {
    "map_bg": "#ffffff",        # white observation window
    "grid": "rgba(15,30,55,0.05)",
    "topic": "#ef9a2e",         # warm amber — landmarks
    "topic_ring": "#b9701a",
    "topic_text": "#7a531a",    # dark amber — readable on white
    "paper": "#1593b8",         # teal — ゲノビ年表 papers
    "bg_point": "#aeb8c8",      # faint grey nebula — related papers
    "ink": "#101a26",
    "ink_soft": "#5d6b7d",
    "halo": "#ffffff",
    "select_ring": "#0c151f",   # dark ring so a selected paper stands out
}

# =========================================================================
# Data source
# -------------------------------------------------------------------------
# データはリポジトリの data/ ディレクトリに格納されています。
# ローカルに data/ があればそれを読み、無ければ GitHub の raw から取得します。
# 旧 Google Sheets からは移行済みです。
# =========================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_BASE = "https://raw.githubusercontent.com/kuroppy/GenobiNenpyoVisualizer/main/data"

FILES = {
    "topics": "topics.csv",
    "background": "background.csv",
    "data": "data.csv",
    "topic_comp": "topic_composition.csv",
}


def read_table(key: str) -> pd.DataFrame:
    """Read a data CSV — prefer the local data/ dir, fall back to GitHub raw."""
    fname = FILES[key]
    local_path = os.path.join(DATA_DIR, fname)
    if os.path.exists(local_path):
        return pd.read_csv(local_path, sep=",", encoding="utf-8-sig")
    r = requests.get(f"{RAW_BASE}/{fname}", timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.content.decode("utf-8-sig")), sep=",")


# =========================================================================
# Normalisation
# =========================================================================
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


# =========================================================================
# Data loading (本番は Google Sheets。GENOBI_MOCK=1 でモックデータを使用)
# =========================================================================
def _mock_dataframes():
    """Schema-compatible synthetic data for local smoke-testing only."""
    rng = np.random.default_rng(7)
    n_topics, n_papers, n_bg = 6, 60, 800

    topics_df = pd.DataFrame({
        "topic_id": [f"topic_{i}" for i in range(n_topics)],
        "theme": [f"テーマ {i}" for i in range(n_topics)],
        "keywords": [", ".join(f"kw{i}_{j}" for j in range(5)) for i in range(n_topics)],
        "UMAP1": rng.normal(0, 5, n_topics),
        "UMAP2": rng.normal(0, 5, n_topics),
    })

    background_df = pd.DataFrame({
        "UMAP1": rng.normal(0, 6, n_bg),
        "UMAP2": rng.normal(0, 6, n_bg),
    })

    comp = rng.dirichlet(np.ones(n_topics) * 0.4, n_papers)
    main_topic = comp.argmax(axis=1)
    data_df = pd.DataFrame({
        "paper_id": [f"P{i:04d}" for i in range(n_papers)],
        "UMAP1": rng.normal(0, 5, n_papers),
        "UMAP2": rng.normal(0, 5, n_papers),
        "main_topic": main_topic,
        "Publication_Year": rng.integers(1995, 2025, n_papers),
        "Cited_by_count": rng.integers(0, 4000, n_papers),
        "title": [f"Sample paper {i}: microbial genomics study" for i in range(n_papers)],
        "journal": rng.choice(["Nature", "Science", "ISME J", "mBio", "NAR"], n_papers),
        "DOI": [f"https://doi.org/10.0000/mock.{i}" for i in range(n_papers)],
        "Author": ["Yamada T.; Suzuki H.; Tanaka K." for _ in range(n_papers)],
        "event": rng.choice(["2024年版", "2023年版", "2022年版"], n_papers),
    })

    comp_df = pd.DataFrame(comp, columns=[f"topic_{i}" for i in range(n_topics)])
    comp_df.insert(0, "paper_id", data_df["paper_id"])

    return (
        normalize_topic_df(topics_df),
        normalize_background_df(background_df),
        normalize_data_df(data_df),
        normalize_topic_comp_df(comp_df),
    )


def load_dataframes():
    if os.environ.get("GENOBI_MOCK") == "1":
        return _mock_dataframes()
    return (
        normalize_topic_df(read_table("topics")),
        normalize_background_df(read_table("background")),
        normalize_data_df(read_table("data")),
        normalize_topic_comp_df(read_table("topic_comp")),
    )


topics_df, background_df, data_df, topic_comp_df = load_dataframes()

merged_df = data_df.merge(topic_comp_df, on="paper_id", how="left")

topic_cols = [c for c in topic_comp_df.columns if c.startswith("topic_")]
topic_id_set = set(topics_df["topic_id"].tolist())
topic_cols = [c for c in topic_cols if c in topic_id_set]
for c in topic_cols:
    if c in merged_df.columns:
        merged_df[c] = merged_df[c].fillna(0.0)

topic_info = topics_df.set_index("topic_id").to_dict("index")

paper_hover_text = merged_df["title"].fillna(merged_df["paper_id"]).astype(str).tolist()
topic_hover_text = topics_df["theme"].fillna(topics_df["topic_id"]).astype(str).tolist()

# Derived constants for header stats & controls
N_PAPERS = len(merged_df)
N_TOPICS = len(topics_df)
N_TOTAL = N_PAPERS + len(background_df)
HAS_CITES = "Cited_by_count" in merged_df.columns

_years = pd.to_numeric(merged_df.get("Publication_Year"), errors="coerce").dropna()
if len(_years):
    YEAR_MIN, YEAR_MAX = int(_years.min()), int(_years.max())
else:
    YEAR_MIN, YEAR_MAX = 1990, 2025


def _fmt(n: int) -> str:
    return f"{n:,}"


# =========================================================================
# Marker-size helpers
# =========================================================================
def topic_marker_sizes(selected_paper_id, base_size=15, scale=46, min_size=9):
    if not selected_paper_id:
        return [base_size] * len(topics_df)
    row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
    if row.empty:
        return [base_size] * len(topics_df)
    row = row.iloc[0]
    return [min_size + scale * np.sqrt(max(float(row.get(t, 0.0)), 0.0)) for t in topics_df["topic_id"]]


def topic_text_sizes(selected_paper_id, base=11, scale=9):
    if not selected_paper_id:
        return [base] * len(topics_df)
    row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
    if row.empty:
        return [base] * len(topics_df)
    row = row.iloc[0]
    return [base + scale * np.sqrt(max(float(row.get(t, 0.0)), 0.0)) for t in topics_df["topic_id"]]


def paper_marker_sizes(selected_topic_id, base=9.0, scale=40, min_size=6):
    if not selected_topic_id or selected_topic_id not in merged_df.columns:
        return np.full(len(merged_df), base)
    vals = merged_df[selected_topic_id].fillna(0.0).astype(float).values
    return min_size + scale * np.sqrt(np.clip(vals, 0.0, None))


# =========================================================================
# Topic-composition bar chart
# =========================================================================
def make_topic_bar_figure(selected_paper_id):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=240,
        margin=dict(l=8, r=14, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Noto Sans JP', system-ui, sans-serif", color=C["ink"]),
    )

    if not selected_paper_id:
        fig.add_annotation(text="論文を選ぶと組成を表示", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=12, color=C["ink_soft"]))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
    if row.empty:
        fig.add_annotation(text="該当論文が見つかりません", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=12, color=C["ink_soft"]))
        return fig

    row = row.iloc[0]
    weights = (pd.Series({t: float(row.get(t, 0.0)) for t in topic_cols})
               .sort_values(ascending=False).head(8))

    labels = []
    for tid in weights.index:
        theme = str(topic_info.get(tid, {}).get("theme", "") or tid)
        labels.append(theme if len(theme) <= 18 else theme[:17] + "…")

    # horizontal bars read top-to-bottom; reverse so the largest sits on top
    labels = labels[::-1]
    values = weights.values[::-1]
    shades = np.linspace(0.45, 1.0, len(values))
    colors = [f"rgba(255,160,60,{a:.2f})" for a in shades]

    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}<br>weight = %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title=None, showgrid=True, gridcolor="rgba(16,26,38,0.06)",
                     zeroline=False, tickfont=dict(size=10, color=C["ink_soft"]))
    fig.update_yaxes(tickfont=dict(size=11, color=C["ink"]))
    return fig


# =========================================================================
# Detail panel
# =========================================================================
def _yaml_line(key, value, href=None, multiline=False):
    """Render one `key: value` line in a YAML-ish style."""
    if href:
        val = html.A(value, href=href, target="_blank", rel="noopener", className="yml-link")
    else:
        val = html.Span(value, className="yml-val")
    cls = "yml-line yml-col" if multiline else "yml-line"
    return html.Div([html.Span(f"{key}:", className="yml-key"), val], className=cls)


def build_detail_panel(selected_paper_id, selected_topic_id):
    blocks = []

    if selected_paper_id:
        row = merged_df.loc[merged_df["paper_id"] == selected_paper_id]
        if not row.empty:
            row = row.iloc[0]
            title = str(row.get("title", "") or "")
            event = str(row.get("event", "") or "")
            year = row.get("Publication_Year", np.nan)
            journal = str(row.get("journal", "") or "")
            doi = str(row.get("DOI", "") or "")
            authors = str(row.get("Author", "") or "")
            cites = row.get("Cited_by_count", np.nan)

            lines = []
            if pd.notna(year):
                lines.append(_yaml_line("year", f"{int(year)}"))
            if journal and journal.lower() != "nan":
                lines.append(_yaml_line("journal", journal))
            if HAS_CITES and pd.notna(cites):
                lines.append(_yaml_line("cited_by", f"{int(cites):,}"))
            if event and event.lower() != "nan":
                lines.append(_yaml_line("note", event, multiline=True))
            if doi and doi.lower() != "nan":
                lines.append(_yaml_line("doi", doi, href=doi))
            if authors and authors.lower() != "nan":
                lines.append(_yaml_line("authors", authors.replace(";", "; "),
                                        multiline=True))

            blocks.append(html.Div([
                html.Div("PAPER", className="eyebrow"),
                html.H4(title, className="detail-title"),
                html.Div(lines, className="yml-block"),
            ], className="detail-card"))

    if selected_topic_id:
        info = topic_info.get(selected_topic_id, {})
        theme = str(info.get("theme", "") or "")
        keywords = str(info.get("keywords", "") or "")
        kw_items = [k.strip() for k in keywords.split(",") if k.strip()]
        blocks.append(html.Div([
            html.Div("TOPIC", className="eyebrow"),
            html.H4(theme or selected_topic_id, className="detail-title"),
            html.Div("Keywords", className="field-label"),
            html.Div([html.Span(k, className="kw") for k in kw_items], className="kw-row")
            if kw_items else html.Div("—", className="authors"),
        ], className="detail-card"))

    if not blocks:
        return html.Div([
            html.Div("地図上の点、または下の表から", className="empty-line"),
            html.Div("論文・トピックを選択してください。", className="empty-line"),
        ], className="empty-state")

    return html.Div(blocks)


# =========================================================================
# Main UMAP figure
# =========================================================================
def make_figure(selected_paper_id, selected_topic_id):
    fig = go.Figure()

    # --- related papers (faint nebula) ---
    fig.add_trace(go.Scattergl(
        x=background_df["UMAP1"], y=background_df["UMAP2"],
        mode="markers", name="related",
        marker=dict(size=4.5, opacity=0.45, color=C["bg_point"]),
        hoverinfo="skip", showlegend=False,
    ))

    # --- ゲノビ年表 papers ---
    sizes = np.asarray(paper_marker_sizes(selected_topic_id), dtype=float).copy()
    opacity = np.full(len(merged_df), 0.85)
    line_w = np.zeros(len(merged_df))
    line_c = C["halo"]

    if selected_paper_id:
        opacity[:] = 0.45
        sel = (merged_df["paper_id"] == selected_paper_id).values
        sizes[sel] = np.maximum(sizes[sel], 20.0)
        opacity[sel] = 1.0
        line_w[sel] = 2.4
        line_c = C["select_ring"]

    fig.add_trace(go.Scattergl(
        x=merged_df["UMAP1"], y=merged_df["UMAP2"],
        mode="markers", name="papers",
        marker=dict(size=sizes, opacity=opacity, color=C["paper"],
                    line=dict(width=line_w, color=line_c)),
        text=paper_hover_text, hovertemplate="%{text}<extra></extra>",
        customdata=merged_df["paper_id"].astype(str), showlegend=False,
    ))

    # --- topics (amber landmarks with labels) ---
    t_sizes = topic_marker_sizes(selected_paper_id)
    text_sizes = topic_text_sizes(selected_paper_id)
    fig.add_trace(go.Scatter(
        x=topics_df["UMAP1"], y=topics_df["UMAP2"],
        mode="markers+text", name="topics",
        text=topics_df["theme"], textposition="top center",
        marker=dict(size=t_sizes, opacity=0.97, color=C["topic"],
                    line=dict(width=1.4, color=C["topic_ring"]), symbol="circle"),
        textfont=dict(size=12, color=C["topic_text"],
                      family="'Space Mono', ui-monospace, monospace"),
        texttemplate="%{text}", textfont_size=text_sizes,
        hovertext=topic_hover_text, hovertemplate="%{hovertext}<extra></extra>",
        customdata=topics_df["topic_id"].astype(str), showlegend=False,
    ))

    fig.update_layout(
        height=820,
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
        paper_bgcolor=C["map_bg"],
        plot_bgcolor=C["map_bg"],
        dragmode="pan",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(15,30,55,0.18)",
                        font=dict(color=C["ink"], family="'Noto Sans JP', sans-serif", size=12)),
        font=dict(color=C["ink"]),
        xaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False,
                   showticklabels=False, showline=False, title="", constrain="domain"),
        yaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False,
                   showticklabels=False, showline=False, title="",
                   scaleanchor="x", scaleratio=1),
    )
    return fig


# =========================================================================
# App + embedded styling (self-contained — no external CSS needed)
# =========================================================================
app = Dash(
    __name__,
    title="ゲノム微生物学の地図",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?"
        "family=Shippori+Mincho:wght@600;700;800&"
        "family=Noto+Sans+JP:wght@400;500;700&"
        "family=Space+Mono:wght@400;700&display=swap"
    ],
)
server = app.server

app.index_string = """<!DOCTYPE html>
<html lang="ja">
<head>
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  {%css%}
  <style>
    :root{
      --paper:#eef1f6; --paper2:#e6eaf1; --card:#ffffff; --ink:#101a26;
      --ink-soft:#5d6b7d; --line:#d9e0ea; --line-strong:#c4cdda;
      --map:#0c111c; --amber:#f0a23b; --amber-deep:#c2731a; --cyan:#1f93b3;
      --accent:#1d6f8b;
    }
    *{box-sizing:border-box}
    body{margin:0;background:
        radial-gradient(1200px 600px at 80% -10%, #e9edf4 0%, rgba(233,237,244,0) 60%),
        radial-gradient(900px 500px at -10% 110%, #e7ecf3 0%, rgba(231,236,243,0) 55%),
        var(--paper);
      color:var(--ink);
      font-family:'Noto Sans JP', system-ui, -apple-system, "Hiragino Kaku Gothic ProN", Meiryo, sans-serif;
      -webkit-font-smoothing:antialiased;}
    a{text-decoration:none}

    .app{max-width:1500px;margin:0 auto;padding:30px 26px 56px}

    /* ---- header ---- */
    .masthead{display:flex;justify-content:space-between;align-items:flex-end;
      gap:24px;flex-wrap:wrap;
      border-bottom:1.5px solid var(--line-strong);padding-bottom:20px;margin-bottom:22px}
    .eyebrow{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.22em;
      text-transform:uppercase;color:var(--accent);font-weight:700;margin-bottom:10px}
    .title{font-family:'Shippori Mincho',serif;font-weight:800;
      font-size:clamp(30px,4.4vw,52px);line-height:1.04;letter-spacing:.01em;
      margin:0;color:#0c151f}
    .subtitle{margin:14px 0 0;max-width:680px;font-size:14px;line-height:1.75;color:var(--ink-soft)}
    .stats{display:flex;gap:10px;flex-wrap:wrap}
    .stat{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:11px 16px;min-width:96px;box-shadow:0 1px 2px rgba(16,26,38,.04)}
    .stat .num{font-family:'Space Mono',monospace;font-size:24px;font-weight:700;
      line-height:1;color:#0c151f}
    .stat .lab{font-size:10.5px;letter-spacing:.08em;color:var(--ink-soft);
      margin-top:7px;text-transform:uppercase}

    /* ---- controls ---- */
    .controls{display:flex;align-items:center;gap:22px;flex-wrap:wrap;
      background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:13px 18px;margin-bottom:16px;box-shadow:0 1px 3px rgba(16,26,38,.04)}
    .ctrl-group{display:flex;align-items:center;gap:11px}
    .ctrl-label{font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.12em;
      text-transform:uppercase;color:var(--ink-soft);white-space:nowrap}
    .reset-btn{font-family:'Noto Sans JP',sans-serif;font-size:13px;font-weight:500;
      color:var(--ink);background:var(--paper2);border:1px solid var(--line-strong);
      border-radius:9px;padding:8px 15px;cursor:pointer;transition:.15s}
    .reset-btn:hover{background:#dde3ec;border-color:#aab6c6}
    .reset-btn:active{transform:translateY(1px)}
    .seg .dash-radio-items,.seg label{display:inline-flex;align-items:center}
    .seg label{font-size:12.5px;color:var(--ink-soft);margin-right:2px;
      padding:6px 12px;border:1px solid var(--line);border-right:none;cursor:pointer;
      transition:.15s;background:#fff}
    .seg label:first-of-type{border-radius:9px 0 0 9px}
    .seg label:last-of-type{border-radius:0 9px 9px 0;border-right:1px solid var(--line)}
    .seg input{position:absolute;opacity:0;width:0;height:0}
    .seg label:has(input:checked){background:var(--accent);color:#fff;
      border-color:var(--accent)}
    .year-wrap{flex:1;min-width:230px;display:flex;align-items:center;gap:14px}
    .year-slider{flex:1}
    .year-read{font-family:'Space Mono',monospace;font-size:12px;color:var(--ink);
      white-space:nowrap;min-width:104px;text-align:right}

    /* ---- layout ---- */
    .grid{display:flex;gap:18px;align-items:stretch}
    .left{flex:1.85;min-width:0}
    .right{flex:1;min-width:312px;display:flex;flex-direction:column;gap:16px}

    .viewport{position:relative;border-radius:16px;padding:8px;
      background:var(--card);
      border:1px solid var(--line);
      box-shadow:0 6px 22px rgba(16,26,38,.06), inset 0 0 0 1px rgba(16,26,38,.01)}
    .viewport::before,.viewport::after{content:"";position:absolute;width:15px;height:15px;
      border-color:var(--line-strong);pointer-events:none;z-index:2}
    .viewport::before{top:13px;left:13px;border-top:1.5px solid;border-left:1.5px solid}
    .viewport::after{bottom:13px;right:13px;border-bottom:1.5px solid;border-right:1.5px solid}
    .viewport .js-plotly-plot{border-radius:10px;overflow:hidden}
    .scope-tag{position:absolute;top:16px;right:20px;z-index:2;
      font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.16em;
      color:var(--ink-soft);opacity:.6;text-transform:uppercase}

    .panel{background:var(--card);border:1px solid var(--line);border-radius:16px;
      padding:18px 20px;box-shadow:0 1px 3px rgba(16,26,38,.04)}
    .panel-h{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.14em;
      text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 14px}

    /* ---- legend ---- */
    .legend-item{display:flex;gap:13px;align-items:flex-start;padding:9px 0;
      border-top:1px solid var(--line)}
    .legend-item:first-of-type{border-top:none}
    .dot{width:13px;height:13px;border-radius:50%;margin-top:3px;flex:none;
      box-shadow:0 0 0 3px rgba(0,0,0,.04)}
    .dot.t{background:var(--amber);box-shadow:0 0 0 3px rgba(239,154,46,.14)}
    .dot.p{background:#1593b8;box-shadow:0 0 0 3px rgba(21,147,184,.14)}
    .dot.r{background:#aeb8c8}
    .legend-name{font-weight:700;font-size:13.5px;color:var(--ink)}
    .legend-desc{font-size:12px;color:var(--ink-soft);line-height:1.55;margin-top:2px}

    /* ---- detail ---- */
    .panel-head-row{display:flex;align-items:center;justify-content:space-between;
      margin-bottom:14px}
    .clear-link{font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.08em;
      text-transform:uppercase;color:var(--ink-soft);background:none;border:none;
      cursor:pointer;padding:2px 4px;transition:.15s}
    .clear-link:hover{color:var(--accent)}
    .empty-state{padding:22px 4px;text-align:center}
    .empty-line{font-size:13px;color:var(--ink-soft);line-height:1.7}
    .detail-card{border-top:1px solid var(--line);padding-top:14px;margin-top:14px}
    .detail-card:first-of-type{border-top:none;padding-top:0;margin-top:0}
    .detail-title{font-family:'Shippori Mincho',serif;font-size:17px;font-weight:700;
      line-height:1.4;margin:4px 0 12px;color:#0c151f}

    /* ---- YAML-style metadata ---- */
    .yml-block{font-family:'Space Mono',ui-monospace,monospace;font-size:12.5px;
      line-height:1.85;background:#f8fafc;border-left:2px solid var(--amber);
      border-radius:0 8px 8px 0;padding:12px 14px;margin-top:4px}
    .yml-line{display:flex;gap:9px;padding:1px 0}
    .yml-line.yml-col{flex-direction:column;gap:2px}
    .yml-key{color:var(--accent);flex:none;min-width:70px}
    .yml-line.yml-col .yml-key{min-width:0}
    .yml-val{color:var(--ink);word-break:break-word}
    .yml-line.yml-col .yml-val{padding-left:18px;color:#33414f}
    .yml-link{color:var(--accent);text-decoration:underline;word-break:break-all}

    .field-label{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.12em;
      text-transform:uppercase;color:var(--ink-soft);margin:14px 0 8px}
    .authors{font-size:12.5px;line-height:1.6;color:#3a4656}
    .kw-row{display:flex;flex-wrap:wrap;gap:7px}
    .kw{font-size:12px;color:#2a3340;background:#fff5e8;border:1px solid #f3dcb6;
      border-radius:7px;padding:4px 10px}

    .helper{font-size:11.5px;color:var(--ink-soft);line-height:1.6;margin-top:11px}

    /* ---- table ---- */
    .table-section{margin-top:26px}
    .table-card{background:var(--card);border:1px solid var(--line);border-radius:16px;
      padding:18px 18px 8px;box-shadow:0 1px 3px rgba(16,26,38,.04)}
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table{
      border-collapse:separate!important;border-spacing:0}
    .dash-table-container .dash-header{
      background:#f4f6fa!important;color:#0c151f!important;font-weight:700!important;
      font-family:'Space Mono',monospace!important;font-size:11px!important;
      letter-spacing:.06em;text-transform:uppercase;
      border-bottom:1.5px solid var(--line-strong)!important;
      position:sticky!important;top:0;z-index:5}
    .dash-table-container .dash-spreadsheet-inner th{
      position:sticky!important;top:0;z-index:5;background:#f4f6fa!important}
    /* native filter row sticks just below the header */
    .dash-table-container .dash-filter{position:sticky!important;top:34px;z-index:4;
      background:#f4f6fa!important}
    .dash-table-container .dash-cell{
      background:#fff!important;color:var(--ink)!important;
      border-bottom:1px solid #eef1f6!important}
    .dash-table-container .dash-cell.focused,
    .dash-table-container .dash-cell.cell--selected{
      background:#e9f3f7!important;border:1px solid #9fcfdd!important}
    .dash-table-container input.dash-filter--case{color:var(--ink-soft)}

    /* ---- footer ---- */
    .foot{margin-top:30px;padding-top:18px;border-top:1px solid var(--line);
      font-size:11.5px;color:var(--ink-soft);line-height:1.7}
    .foot a{color:var(--accent)}

    @media (max-width:1080px){
      .grid{flex-direction:column}
      .left,.right{width:100%}
      .right{min-width:0}
    }
  </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

# Plotly config: clean toolbar
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def legend_row(cls, name, desc):
    return html.Div([
        html.Div(className=f"dot {cls}"),
        html.Div([html.Div(name, className="legend-name"),
                  html.Div(desc, className="legend-desc")]),
    ], className="legend-item")


table_cols = ["paper_id", "Publication_Year", "title", "journal"]

# Highlight the selected paper by paper_id (not by row index) so it survives
# sorting/filtering without any selected_rows round-trip — this avoids the
# selection<->table feedback loop that made sorting flicker.
TABLE_STRIPE = [{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfd"}]


def table_style(selected_paper_id):
    styles = list(TABLE_STRIPE)
    if selected_paper_id:
        styles.append({
            "if": {"filter_query": '{paper_id} = "%s"' % selected_paper_id},
            "backgroundColor": "#e3f1f6",
            "borderLeft": "3px solid #1593b8",
        })
    return styles

app.layout = html.Div([
    dcc.Store(id="selected-paper-id", data=None),
    dcc.Store(id="selected-topic-id", data=None),

    # ---------- masthead ----------
    html.Div([
        html.Div([
            html.Div("Genome Microbiology · Topic Atlas", className="eyebrow"),
            html.H1("ゲノム微生物学の地図", className="title"),
            html.P(
                "ゲノビ年表の論文群とその引用ネットワークを、トピックモデル（LDA）と UMAP で"
                "二次元に配置した研究アトラス。点をクリックして、分野の構造と個々の研究の"
                "位置づけを探索できます。",
                className="subtitle"),
        ]),
    ], className="masthead"),

    # ---------- main grid ----------
    html.Div([
        html.Div([
            html.Div([
                html.Div("UMAP · LDA", className="scope-tag"),
                dcc.Graph(id="umap-plot", figure=make_figure(None, None),
                          config=PLOT_CONFIG, style={"height": "820px"}),
            ], className="viewport"),
        ], className="left"),

        html.Div([
            html.Div([
                html.Div("凡例 — Legend", className="panel-h"),
                legend_row("t", "トピック", "23 の LDA トピック。クリックすると、そのトピックを強く含む論文が拡大します。"),
                legend_row("p", "ゲノビ年表論文", "年表由来の論文。クリックすると、その論文のトピック組成を表示します。"),
                legend_row("r", "関連論文", "年表論文と引用・被引用関係にある論文（背景）。"),
            ], className="panel"),

            html.Div([
                html.Div([
                    html.Div("選択 — Selection", className="panel-h", style={"margin": "0"}),
                    html.Button("クリア", id="reset-btn", className="clear-link", n_clicks=0),
                ], className="panel-head-row"),
                html.Div(id="detail-panel"),
            ], className="panel"),

            html.Div([
                html.Div("トピック組成 — Composition", className="panel-h"),
                dcc.Graph(id="topic-bar-plot", figure=make_topic_bar_figure(None),
                          config={"displayModeBar": False}, style={"height": "240px"}),
            ], className="panel"),
        ], className="right"),
    ], className="grid"),

    # ---------- table ----------
    html.Div([
        html.Div([
            html.Div("論文メタデータ — Metadata", className="panel-h"),
            dash_table.DataTable(
                id="meta-table",
                columns=[
                    {"name": "paper_id", "id": "paper_id"},
                    {"name": "Year", "id": "Publication_Year"},
                    {"name": "Title", "id": "title"},
                    {"name": "Journal", "id": "journal"},
                ],
                hidden_columns=["paper_id"],
                data=merged_df.sort_values("Publication_Year")[table_cols].to_dict("records"),
                filter_action="native", sort_action="native", sort_mode="multi",
                sort_by=[{"column_id": "Publication_Year", "direction": "asc"}],
                page_action="none",
                style_table={"height": "560px", "overflowY": "auto", "overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "10px 12px", "fontSize": 13,
                            "whiteSpace": "normal", "height": "auto", "lineHeight": "1.5",
                            "border": "none", "fontFamily": "'Noto Sans JP', sans-serif"},
                style_data={"borderBottom": "1px solid #eef1f6"},
                style_cell_conditional=[
                    {"if": {"column_id": "title"}, "minWidth": "420px", "maxWidth": "700px"},
                    {"if": {"column_id": "Publication_Year"}, "width": "84px",
                     "fontFamily": "'Space Mono', monospace"},
                    {"if": {"column_id": "journal"}, "minWidth": "150px", "maxWidth": "240px"},
                ],
                style_data_conditional=table_style(None),
            ),
            html.Div("行を選ぶと地図上で該当論文がハイライトされます。"
                     "ヘッダーで並べ替え・絞り込みができます。", className="helper"),
        ], className="table-card"),
    ], className="table-section"),

    # ---------- footer ----------
    html.Div([
        html.Span("データ：ゲノビ年表（ver.20250814）収録 264 報 ＋ 引用・被引用論文 7,643 報。"),
        html.Br(),
        html.Span("手法：Abstract を入力としたトピックモデル（LDA）／二次元可視化（UMAP）。"),
    ], className="foot"),
], className="app")


# =========================================================================
# Callbacks
# =========================================================================
@app.callback(
    Output("selected-paper-id", "data"),
    Output("selected-topic-id", "data"),
    Output("meta-table", "active_cell"),
    Input("umap-plot", "clickData"),
    Input("meta-table", "active_cell"),
    Input("reset-btn", "n_clicks"),
    State("meta-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def update_selection(clickData, active_cell, _n_reset, virtual_rows):
    trigger = ctx.triggered_id

    if trigger == "reset-btn":
        # clear active_cell too, so the same row can be re-selected afterwards
        return None, None, None

    if trigger == "umap-plot":
        if clickData and clickData.get("points"):
            pt = clickData["points"][0]
            curve = pt.get("curveNumber")
            cd = pt.get("customdata")
            if curve == 1 and cd is not None:   # papers trace
                return str(cd), None, no_update
            if curve == 2 and cd is not None:   # topics trace
                return None, str(cd), no_update
        return no_update, no_update, no_update

    if trigger == "meta-table":
        if active_cell and virtual_rows:
            r = active_cell.get("row")
            if r is not None and 0 <= r < len(virtual_rows):
                pid = virtual_rows[r].get("paper_id")
                if pid is not None:
                    return str(pid), None, no_update
        return no_update, no_update, no_update

    return no_update, no_update, no_update


@app.callback(
    Output("umap-plot", "figure"),
    Output("detail-panel", "children"),
    Output("topic-bar-plot", "figure"),
    Output("meta-table", "style_data_conditional"),
    Input("selected-paper-id", "data"),
    Input("selected-topic-id", "data"),
)
def refresh_view(selected_paper_id, selected_topic_id):
    fig = make_figure(selected_paper_id, selected_topic_id)
    detail = build_detail_panel(selected_paper_id, selected_topic_id)
    bar_fig = make_topic_bar_figure(selected_paper_id)
    return fig, detail, bar_fig, table_style(selected_paper_id)


if __name__ == "__main__":
    app.run(debug=True)
