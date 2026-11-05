import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env from the app folder BEFORE importing modules that read the environment.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import analyst  # noqa: E402
import charts  # noqa: E402
import data_utils as du  # noqa: E402
import llm  # noqa: E402
from charts import SpecError  # noqa: E402

st.set_page_config(page_title="DataMind AI", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem;}
h1 {margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)

NONE = "(none)"
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def show_fig(fig, key):
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def md(text):
    """Escape $ so currency isn't rendered as LaTeX."""
    return str(text).replace("$", "\\$")


def fig_html(fig):
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


def render_output(fig, table, notes, spec, key, html=None):
    if fig is not None:
        show_fig(fig, f"{key}_fig")
    for n in notes or []:
        st.caption(f"ℹ️ {n}")
    if table is not None and len(table):
        if fig is None:  # a table *is* the answer
            st.dataframe(table, width="stretch")
        else:
            with st.expander("📄 Data behind this chart"):
                st.dataframe(table, width="stretch")
    bcols = st.columns([1, 1, 3])
    if fig is not None:
        bcols[0].download_button("⬇️ Chart (HTML)", html or fig_html(fig), f"{key}.html", "text/html", key=f"{key}_dlh")
    if table is not None and len(table):
        bcols[1].download_button("⬇️ Data (CSV)", table.to_csv(index=False).encode(), f"{key}.csv", "text/csv", key=f"{key}_dlc")
    if spec:
        with st.expander("🔧 How this was computed (chart spec)"):
            st.caption("The AI only proposes this spec; DataMind validates it and computes every number with pandas.")
            st.json(spec)


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes, filename):
    return du.prepare(du.read_file(file_bytes, filename))


@st.cache_resource(show_spinner=False)
def get_index(data_key, _df):
    return du.build_index(_df)


@st.cache_data(show_spinner=False)
def cached_findings(data_key, _df):
    return du.auto_findings(_df)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_fig(data_key, kind, params_json, _df):
    p = json.loads(params_json)
    if kind == "corr":
        return charts.fig_corr(_df, p["method"])
    if kind == "hist":
        return charts.fig_hist_grid(_df, p["cols"])
    if kind == "box":
        return charts.fig_box_grid(_df, p["cols"])
    if kind == "cat":
        return charts.fig_cat_grid(_df, p["cols"])
    if kind == "smatrix":
        return charts.fig_scatter_matrix(_df, p["cols"])
    if kind == "missbar":
        return charts.fig_missing_bar(_df)
    if kind == "missmat":
        return charts.fig_missing_matrix(_df)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Header + sidebar
# --------------------------------------------------------------------------
st.title("🧠 DataMind AI")
st.caption("Upload data → explore automatically → ask questions or request charts in plain English.")

cfg = llm.get_llm_config()
with st.sidebar:
    st.header("⚙️ Settings")
    if cfg and llm.OpenAI is not None:
        st.success(f"LLM: {cfg['name']} · {cfg['model']}")
    else:
        st.warning("No API key found. Charts, EDA and exact statistics still work.")
    st.caption("Set GROQ_API_KEY (or OPENAI_API_KEY) in your .env file.")
    if cfg and llm.OpenAI is not None and st.button("🔌 Test LLM connection"):
        try:
            with st.spinner("Contacting provider..."):
                available = [m.id for m in llm.make_client(cfg).models.list().data]
            if cfg["model"] in available:
                st.success(f"Connected. Key is valid and model '{cfg['model']}' is available.")
            else:
                alt = llm.pick_fallback_model(available, cfg["model"])
                if alt:
                    st.session_state["model_override"] = alt
                    st.warning(f"Key is valid, but '{cfg['model']}' isn't available to your account. "
                               f"Switched to '{alt}' for this session. To make it permanent, set GROQ_MODEL={alt} in .env.")
                else:
                    st.warning(f"Key is valid, but model '{cfg['model']}' was not found. Available: {', '.join(available[:12])}")
        except Exception as exc:
            st.error(llm.explain_error(exc))
    if llm.OpenAI is None:
        st.error("The 'openai' package is missing or too old. Run: pip install -U openai")
    st.toggle("Explain charts with AI", value=True, key="explain_toggle",
              help="After building a chart, the AI writes a short insight based on the computed numbers.")

# --------------------------------------------------------------------------
# Data input
# --------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx"])
if uploaded:
    file_bytes, filename = uploaded.getvalue(), uploaded.name
elif st.session_state.get("use_demo"):
    file_bytes, filename = du.demo_dataset().to_csv(index=False).encode(), "demo_sales.csv"
else:
    st.info("Upload a dataset to begin, or try the built-in demo.")
    if st.button("🚀 Load demo dataset (sales data)"):
        st.session_state["use_demo"] = True
        st.rerun()
    st.markdown("""
### What DataMind AI does
- ✨ **Auto insights** — correlation heatmap, distributions, outliers, category breakdowns, trends
- 🎨 **Chart Builder** — build any chart with dropdowns (bar, line, scatter, box, violin, pie, heatmaps…)
- 🤖 **AI Analyst** — ask *“heatmap of average revenue by region and category”* and get a real chart
- 🧮 **Exact numbers** — the AI plans, pandas computes; nothing is guessed
- 🧹 **Data quality** — missing values, duplicates, outliers at a glance
""")
    st.stop()

try:
    df = load_dataset(file_bytes, filename)
except Exception as exc:
    st.error(f"Could not read **{filename}**: {exc}")
    st.stop()
if df.empty or len(df.columns) == 0:
    st.error("The file has no data. Please upload a dataset with a header row and at least one row.")
    st.stop()

data_key = hashlib.md5(file_bytes).hexdigest()
if st.session_state.get("data_key") != data_key:  # new dataset -> reset conversation state
    st.session_state.update(data_key=data_key, messages=[], last_spec=None, ai_report=None)

kinds = du.col_kinds(df)
summary = du.basic_summary(df)
index = get_index(data_key, df)
K = lambda name: f"{name}_{data_key[:8]}"  # noqa: E731  (widget keys tied to the dataset)

st.success(f"Loaded **{filename}** — {summary['rows']:,} rows × {summary['columns']} columns.")
cells = max(summary["rows"] * summary["columns"], 1)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{summary['rows']:,}", border=True)
c2.metric("Columns", summary["columns"], border=True)
c3.metric("Missing Values", f"{summary['missing']:,}", f"{100 * summary['missing'] / cells:.1f}% of cells", delta_color="off", border=True)
c4.metric("Duplicates", f"{summary['duplicates']:,}", border=True)
c5.metric("Numeric / Category / Date", f"{len(kinds['numeric'])} / {len(kinds['categorical'])} / {len(kinds['datetime'])}", border=True)

tab_data, tab_ins, tab_build, tab_ai, tab_q = st.tabs(
    ["📋 Data", "✨ Auto Insights", "🎨 Chart Builder", "🤖 AI Analyst", "🧹 Data Quality"])


# --------------------------------------------------------------------------
# 📋 Data
# --------------------------------------------------------------------------
with tab_data:
    n_prev = st.slider("Rows to preview", 10, min(500, max(len(df), 11)), min(100, max(len(df), 10)), key=K("prev"))
    st.dataframe(df.head(n_prev), width="stretch")
    st.download_button("⬇️ Download preview as CSV", df.head(n_prev).to_csv(index=False).encode(),
                       "datamind_preview.csv", "text/csv")
    st.subheader("Descriptive statistics")
    if kinds["numeric"]:
        st.dataframe(df[kinds["numeric"]].describe().T.round(3), width="stretch")
    else:
        st.caption("No numeric columns.")

# --------------------------------------------------------------------------
# ✨ Auto Insights
# --------------------------------------------------------------------------
with tab_ins:
    st.subheader("Key findings")
    findings = cached_findings(data_key, df)
    for f in findings:
        st.markdown(f"- {f}")
    if llm.llm_available():
        if st.button("✨ Generate AI briefing", key=K("report_btn")):
            try:
                with st.spinner("Writing briefing..."):
                    st.session_state["ai_report"] = llm.write_report(du.schema_text(df), findings)
            except Exception as exc:
                st.error(llm.explain_error(exc))
        if st.session_state.get("ai_report"):
            st.info(md(st.session_state["ai_report"]))

    num = [c for c in kinds["numeric"] if df[c].nunique() > 1]
    if len(num) >= 2:
        st.divider()
        st.subheader("🔥 Correlation heatmap")
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key=K("corr_method"),
                          help="Pearson = linear relationships. Spearman = any consistently rising/falling relationship.")
        show_fig(cached_fig(data_key, "corr", json.dumps({"method": method}), df), K("auto_corr"))
        pairs = du.top_correlations(df, n=8, method=method)
        if len(pairs):
            with st.expander("Top correlated pairs"):
                st.dataframe(pairs, width="stretch")

    if num:
        st.divider()
        st.subheader("📊 Distributions & outliers")
        mode = st.radio("Plot type", ["Histograms", "Box plots"], horizontal=True, key=K("dist_mode"))
        cols = st.multiselect("Columns", num, default=num[:6], key=K("dist_cols"))
        if cols:
            fig = cached_fig(data_key, "hist" if mode == "Histograms" else "box", json.dumps({"cols": cols}), df)
            show_fig(fig, K("auto_dist"))

    cats = [c for c in kinds["categorical"] if 1 < df[c].nunique() <= 40]
    if cats:
        st.divider()
        st.subheader("🧩 Category breakdown")
        ccols = st.multiselect("Columns", cats, default=cats[:6], key=K("cat_cols"))
        if ccols:
            show_fig(cached_fig(data_key, "cat", json.dumps({"cols": ccols}), df), K("auto_cat"))

    if kinds["datetime"] and kinds["numeric"]:
        st.divider()
        st.subheader("📈 Trend over time")
        t1, t2, t3, t4 = st.columns(4)
        dcol = t1.selectbox("Date column", kinds["datetime"], key=K("t_date"))
        tval = t2.selectbox("Metric", kinds["numeric"], key=K("t_val"))
        tagg = t3.selectbox("Aggregation", ["sum", "mean", "count", "median"], key=K("t_agg"))
        tbucket = t4.selectbox("Group by", ["auto", "day", "week", "month", "quarter", "year"], key=K("t_bucket"))
        try:
            res = charts.build_chart({"chart_type": "line", "x": dcol, "y": tval, "agg": tagg,
                                      "time_bucket": None if tbucket == "auto" else tbucket}, df)
            show_fig(res.fig, K("auto_trend"))
        except SpecError as exc:
            st.warning(str(exc))

    if len(num) >= 3:
        st.divider()
        st.subheader("🔗 Pairwise relationships")
        pcols = st.multiselect("Pick 2–6 numeric columns", num, default=num[:4], max_selections=6, key=K("pair_cols"))
        if len(pcols) >= 2:
            show_fig(cached_fig(data_key, "smatrix", json.dumps({"cols": pcols}), df), K("auto_pairs"))

# --------------------------------------------------------------------------
# 🎨 Chart Builder (no AI needed)
# --------------------------------------------------------------------------
with tab_build:
    st.caption("Build any chart with dropdowns. Works without an API key.")
    TYPES = {"Bar": "bar", "Line": "line", "Area": "area", "Scatter": "scatter", "Histogram": "histogram",
             "Box": "box", "Violin": "violin", "Pie": "pie", "Correlation heatmap": "correlation_heatmap",
             "Pivot heatmap": "pivot_heatmap"}
    all_cols = list(df.columns)
    groupable = [c for c in all_cols if du.is_groupable(df[c])] or all_cols
    numc = kinds["numeric"]

    def opt(col, label, options, key, index=0):
        v = col.selectbox(label, [NONE] + options, index=index, key=K(f"b_{key}"))
        return None if v == NONE else v

    top = st.columns(3)
    label = top[0].selectbox("Chart type", list(TYPES), key=K("b_type"))
    ct = TYPES[label]
    spec = {"chart_type": ct}
    row = st.columns(4)
    ok = True
    if ct == "correlation_heatmap":
        spec["method"] = row[0].radio("Method", ["pearson", "spearman"], horizontal=True, key=K("b_method"))
        spec["columns"] = row[1].multiselect("Columns (default: all numeric)", numc, key=K("b_ccols")) or None
    elif ct == "histogram":
        if numc:
            spec["x"] = row[0].selectbox("Column", numc, key=K("b_hx"))
            spec["color"] = opt(row[1], "Colour by (optional)", groupable, "hcolor")
            spec["bins"] = row[2].slider("Bins", 5, 100, 25, key=K("b_bins"))
        else:
            ok = False
    elif ct in ("box", "violin"):
        if numc:
            spec["y"] = row[0].selectbox("Value (numeric)", numc, key=K("b_boxy"))
            spec["x"] = opt(row[1], "Group by (optional)", groupable, "boxx")
        else:
            ok = False
    elif ct == "scatter":
        if len(numc) >= 2:
            spec["x"] = row[0].selectbox("X (numeric)", numc, key=K("b_sx"))
            spec["y"] = row[1].selectbox("Y (numeric)", numc, index=1, key=K("b_sy"))
            spec["color"] = opt(row[2], "Colour by (optional)", groupable, "scolor")
            spec["trendline"] = row[3].checkbox("Trendline", value=True, key=K("b_trend"))
        else:
            ok = False
    elif ct in ("bar", "pie"):
        spec["x"] = row[0].selectbox("Group by (X)", groupable, key=K("b_bx"))
        spec["y"] = opt(row[1], "Metric (optional)", numc, "by")
        agg_opts = ["mean", "sum", "median", "min", "max", "count"]
        spec["agg"] = row[2].selectbox("Aggregation", agg_opts, index=5 if not spec["y"] else 0, key=K("b_bagg"))
        spec["top_n"] = row[3].slider("Max groups", 3, 50, 15, key=K("b_top"))
        row2 = st.columns(4)
        spec["sort"] = row2[0].selectbox("Sort", ["desc", "asc"], key=K("b_sort"))
        if ct == "bar":
            spec["color"] = opt(row2[1], "Split by (optional)", [c for c in groupable if c != spec["x"]], "bcolor")
            spec["orientation"] = "h" if row2[2].checkbox("Horizontal", key=K("b_hor")) else "v"
    elif ct in ("line", "area"):
        xopts = kinds["datetime"] + [c for c in all_cols if c not in kinds["datetime"]]
        spec["x"] = row[0].selectbox("X (date or ordered)", xopts, key=K("b_lx"))
        spec["y"] = opt(row[1], "Metric (optional)", numc, "ly", index=1 if numc else 0)
        spec["agg"] = row[2].selectbox("Aggregation", ["sum", "mean", "median", "min", "max", "count"], key=K("b_lagg"))
        spec["color"] = opt(row[3], "Split by (optional)", [c for c in groupable if c != spec["x"]], "lcolor")
        if spec["x"] in kinds["datetime"]:
            tb = st.selectbox("Group dates by", ["auto", "day", "week", "month", "quarter", "year"], key=K("b_lbucket"))
            spec["time_bucket"] = None if tb == "auto" else tb
    elif ct == "pivot_heatmap":
        spec["x"] = row[0].selectbox("Columns (X)", groupable, key=K("b_px"))
        spec["y"] = row[1].selectbox("Rows (Y)", groupable, index=min(1, len(groupable) - 1), key=K("b_py"))
        spec["value"] = opt(row[2], "Value (optional)", numc, "pval")
        spec["agg"] = row[3].selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"],
                                       index=5 if not spec["value"] else 0, key=K("b_pagg"))

    with st.expander("🔎 Filter rows (optional)"):
        fcol = st.selectbox("Filter column", [NONE] + all_cols, key=K("b_fcol"))
        if fcol != NONE:
            fk = du.kind_of(df[fcol])
            if fk == "numeric":
                f1, f2 = st.columns(2)
                fop = f1.selectbox("Condition", [">", ">=", "<", "<=", "==", "!="], key=K("b_fop"))
                fval = f2.number_input("Value", value=float(df[fcol].median()), key=K("b_fval"))
                spec["filters"] = [{"column": fcol, "op": fop, "value": fval}]
            elif fk == "categorical" and df[fcol].nunique() <= 300:
                vals = st.multiselect("Keep only these values", sorted(df[fcol].dropna().astype(str).unique()), key=K("b_fvals"))
                if vals:
                    spec["filters"] = [{"column": fcol, "op": "in", "value": vals}]
            else:
                f1, f2 = st.columns(2)
                fop = f1.selectbox("Condition", [">=", "<=", "==", "!="], key=K("b_fop2"))
                fval = f2.text_input("Value (e.g. 2025-01-31)", key=K("b_fval2"))
                if fval:
                    spec["filters"] = [{"column": fcol, "op": fop, "value": fval}]

    if not ok:
        st.warning("This chart type needs more numeric columns than the dataset has.")
    else:
        try:
            res = charts.build_chart(spec, df)
            render_output(res.fig, res.table, res.notes, res.spec, K("builder"))
        except SpecError as exc:
            st.warning(str(exc))

# --------------------------------------------------------------------------
# 🤖 AI Analyst
# --------------------------------------------------------------------------
def render_message(m, i):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
            return
        if m.get("badge"):
            st.caption(m["badge"])
        st.markdown(md(m["content"]))
        render_output(m.get("fig"), m.get("table"), m.get("notes"), m.get("spec"), f"{K('msg')}_{i}", m.get("html"))
        if m.get("retrieved"):
            with st.expander("🔎 Similar rows used as context"):
                for doc, score in m["retrieved"]:
                    st.write(f"**Similarity:** {score:.3f} — {doc}")


with tab_ai:
    if not llm.llm_available():
        st.info("No API key configured: I can still build simple charts (heatmap, histogram, *average X by Y*, trends) "
                "and answer basic stats. Add a Groq key to `.env` for open-ended questions.")
    st.caption("Ask for numbers, insights or **charts** — e.g. *“heatmap of average revenue by region and category”*, "
               "*“box plot of price by category”*, *“make it a pie chart”*, *“only for 2025”*.")
    sug = du.suggest_questions(df)
    scols = st.columns(3)
    for i, s_q in enumerate(sug):
        if scols[i % 3].button(s_q, key=K(f"sug_{i}")):
            st.session_state["queued_q"] = s_q

    msgs = st.session_state["messages"]
    for i, m in enumerate(msgs):
        render_message(m, i)

    q = st.chat_input("Ask about your data or request a chart…", key=K("chat"))
    q = q or st.session_state.pop("queued_q", None)
    if q:
        history = [m["content"] for m in msgs if m["role"] == "user"]
        with st.spinner("Analyzing…"):
            reply = analyst.answer_question(q, df, index, st.session_state.get("last_spec"), history,
                                            explain=st.session_state.get("explain_toggle", True))
        if reply.get("fig") is not None:
            reply["html"] = fig_html(reply["fig"])
        msgs.append({"role": "user", "content": q})
        msgs.append(reply)
        del msgs[:-40]  # keep the conversation bounded
        if reply.get("spec"):
            st.session_state["last_spec"] = reply["spec"]
        st.rerun()
    if msgs and st.button("🗑️ Clear conversation", key=K("clear")):
        st.session_state["messages"], st.session_state["last_spec"] = [], None
        st.rerun()

# --------------------------------------------------------------------------
# 🧹 Data Quality
# --------------------------------------------------------------------------
with tab_q:
    st.subheader("Columns")
    dtype_df = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(x) for x in df.dtypes],
        "Kind": [du.kind_of(df[c]) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(100 * df[c].isna().mean(), 1) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(dtype_df, width="stretch")

    st.subheader("Missing values")
    mb = cached_fig(data_key, "missbar", "{}", df)
    if mb is None:
        st.success("No missing values found.")
    else:
        show_fig(mb, K("q_missbar"))
        mm = cached_fig(data_key, "missmat", "{}", df)
        if mm is not None:
            show_fig(mm, K("q_missmat"))

    st.subheader("Duplicates")
    ndup = summary["duplicates"]
    if ndup:
        st.warning(f"{ndup:,} duplicate rows found.")
        with st.expander("Show duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].head(200), width="stretch")
        st.download_button("⬇️ Download data without duplicates", df.drop_duplicates().to_csv(index=False).encode(),
                           "deduplicated.csv", "text/csv")
    else:
        st.success("No duplicate rows.")

    ot = du.outlier_table(df)
    if len(ot):
        st.subheader("Outliers & skew (IQR rule)")
        st.dataframe(ot, width="stretch")
