"""Safe chart engine.

The LLM never executes code. It proposes a small JSON *spec*; this module validates every
field against the real DataFrame, aggregates with pandas and renders with Plotly, so the
numbers on screen are computed exactly and a hallucinated column name becomes a clear error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_utils import col_kinds, is_groupable, kind_of, norm_text, top_correlations, LOW_CARD

CHART_TYPES = ("bar", "line", "area", "scatter", "histogram", "box", "violin", "pie",
               "correlation_heatmap", "pivot_heatmap", "table")
AGGS = ("mean", "sum", "count", "median", "min", "max", "nunique")
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains")
MAX_GROUPS = 30
SAMPLE_LIMIT = 20_000  # max rows embedded in a distribution plot (keeps the page light)
BUCKET_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}

_CHART_ALIASES = {
    "bar_chart": "bar", "barchart": "bar", "column": "bar", "columns": "bar", "count": "bar", "countplot": "bar",
    "grouped_bar": "bar", "stacked_bar": "bar", "horizontal_bar": "bar", "hbar": "bar",
    "hist": "histogram", "distribution": "histogram", "density": "histogram",
    "line_chart": "line", "trend": "line", "time_series": "line", "timeseries": "line", "area_chart": "area",
    "scatter_plot": "scatter", "scatterplot": "scatter",
    "boxplot": "box", "box_plot": "box", "violin_plot": "violin",
    "donut": "pie", "doughnut": "pie", "pie_chart": "pie",
    "heatmap": "correlation_heatmap", "correlation": "correlation_heatmap", "corr": "correlation_heatmap",
    "correlation_matrix": "correlation_heatmap", "corr_heatmap": "correlation_heatmap",
    "pivot": "pivot_heatmap", "crosstab": "pivot_heatmap", "pivot_table": "pivot_heatmap",
    "data_table": "table", "summary_table": "table",
}
_AGG_ALIASES = {"average": "mean", "avg": "mean", "total": "sum", "unique": "nunique", "distinct": "nunique",
                "number": "count", "rate": "mean", "percentage": "mean", "proportion": "mean", "maximum": "max",
                "minimum": "min", "len": "count", "size": "count"}
_OP_ALIASES = {"=": "==", "eq": "==", "equals": "==", "is": "==", "<>": "!=", "ne": "!=", "not_equals": "!=",
               "gt": ">", "gte": ">=", "ge": ">=", "lt": "<", "lte": "<=", "le": "<=", "=>": ">=", "=<": "<=",
               "≥": ">=", "≤": "<=", "isin": "in", "notin": "not_in", "not in": "not_in",
               "like": "contains", "includes": "contains"}
_BUCKET_ALIASES = {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter",
                   "yearly": "year", "annual": "year", "annually": "year", "days": "day", "weeks": "week",
                   "months": "month", "quarters": "quarter", "years": "year"}


class SpecError(ValueError):
    """The chart request cannot be executed (bad column, wrong type, no data...)."""


@dataclass
class ChartResult:
    fig: object
    table: pd.DataFrame
    title: str
    spec: dict
    notes: list = field(default_factory=list)
    facts: str | None = None  # compact text used to ask the LLM for an explanation


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------
def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _first(v):
    return v[0] if isinstance(v, (list, tuple)) and v else v


def resolve_col(name, df):
    """Map an LLM-provided column name to a real one (case/spacing tolerant)."""
    name = _first(name)
    if name is None:
        return None
    name = str(name).strip()
    if name.lower() in ("", "none", "null", "nan", "n/a", "na"):
        return None
    if name in df.columns:
        return name
    lower = {str(c).lower(): c for c in df.columns}
    if name.lower() in lower:
        return lower[name.lower()]
    k = _key(name)
    matches = [c for c in df.columns if _key(c) == k]
    if matches:
        return matches[0]
    raise SpecError(f"Column '{name}' does not exist. Available columns: {', '.join(map(str, df.columns))}.")


def _int(v, lo, hi, default=None):
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError):
        return default


def normalize_spec(spec, df):
    if not isinstance(spec, dict):
        raise SpecError("The chart specification must be a JSON object.")
    raw_ct = str(spec.get("chart_type") or spec.get("type") or "").lower().strip().replace(" ", "_").replace("-", "_")
    ct = _CHART_ALIASES.get(raw_ct, raw_ct)
    if ct not in CHART_TYPES:
        raise SpecError(f"Unknown chart type '{raw_ct}'. Supported: {', '.join(CHART_TYPES)}.")

    s = {"chart_type": ct}
    for f in ("x", "y", "color", "value", "sort_by"):
        s[f] = resolve_col(spec.get(f), df)
    if raw_ct == "heatmap" and s["x"] and s["y"]:
        s["chart_type"] = "pivot_heatmap"
    if raw_ct in ("horizontal_bar", "hbar"):
        spec = {**spec, "orientation": "h"}
    if raw_ct == "count" and not spec.get("agg"):
        spec = {**spec, "agg": "count"}

    cols = spec.get("columns")
    s["columns"] = None
    if isinstance(cols, (list, tuple)):
        seen = []
        for c in cols:
            rc = resolve_col(c, df)
            if rc and rc not in seen:
                seen.append(rc)
        s["columns"] = seen or None

    agg = str(spec.get("agg") or "").lower().strip()
    agg = _AGG_ALIASES.get(agg, agg)
    s["agg"] = agg if agg in AGGS else None
    s["top_n"] = _int(spec.get("top_n"), 1, 200)
    sort = str(spec.get("sort") or "").lower()
    s["sort"] = "asc" if sort.startswith("asc") else "desc" if sort.startswith("desc") else None
    s["bins"] = _int(spec.get("bins"), 2, 200, 20)
    s["orientation"] = "h" if str(spec.get("orientation") or "").lower().startswith("h") else "v"
    tb = str(spec.get("time_bucket") or "").lower().strip()
    tb = _BUCKET_ALIASES.get(tb, tb)
    s["time_bucket"] = tb if tb in BUCKET_FREQ else None
    s["trendline"] = bool(spec.get("trendline")) and str(spec.get("trendline")).lower() not in ("false", "0", "none")
    s["method"] = "spearman" if str(spec.get("method") or "").lower().startswith("spear") else "pearson"
    title = spec.get("title")
    s["title"] = str(title).strip()[:110] if title else None

    filters = []
    for f in spec.get("filters") or []:
        if not isinstance(f, dict):
            continue
        col = resolve_col(f.get("column") or f.get("col"), df)
        op = str(f.get("op") or f.get("operator") or "==").lower().strip()
        op = _OP_ALIASES.get(op, op)
        if col is None:
            continue
        if op not in OPS:
            raise SpecError(f"Unsupported filter operator '{op}'. Use one of: {', '.join(OPS)}.")
        filters.append({"column": col, "op": op, "value": f.get("value")})
    s["filters"] = filters
    return s


def _coerce(v, series):
    k = kind_of(series)
    try:
        if k == "numeric":
            if isinstance(v, str):
                v = v.replace(",", "").replace("%", "").strip()
            return float(v)
        if k == "datetime":
            return pd.Timestamp(v)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"Value {v!r} is not valid for column '{series.name}' ({k}).") from exc
    return v


def apply_filters(df, filters):
    notes = []
    for f in filters:
        col, op, val = f["column"], f["op"], f["value"]
        s = df[col]
        k = kind_of(s)
        try:
            if op in ("in", "not_in"):
                vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
                if k == "categorical":
                    mask = s.astype(str).str.lower().isin([str(v).lower() for v in vals])
                else:
                    mask = s.isin([_coerce(v, s) for v in vals])
                mask = mask if op == "in" else ~mask
            elif op == "contains":
                mask = s.astype(str).str.contains(str(val), case=False, na=False, regex=False)
            else:
                v = _coerce(val, s)
                if k == "categorical" and op in ("==", "!="):
                    eq = s.astype(str).str.lower() == str(v).lower()
                    mask = eq if op == "==" else ~eq
                else:
                    mask = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
        except (TypeError, ValueError) as exc:
            raise SpecError(f"Could not apply the filter on '{col}' ({op} {val!r}): {exc}") from exc
        df = df[mask.fillna(False).astype(bool)]
        shown = ", ".join(map(str, val)) if isinstance(val, (list, tuple, set)) else val
        notes.append(f"{col} {op} {shown}")
    return df, notes


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------
def aggregate(df, groups, y, agg):
    groups = [g for g in groups if g]
    if agg is None:
        agg = "count" if y is None else "mean"
    if y is None or agg == "count":
        label = "Count"
        if label in groups:
            label = "Count (rows)"
        res = df.groupby(groups, observed=True).size().reset_index(name=label)
        return res, label
    data = df
    if pd.api.types.is_bool_dtype(df[y]):
        data = df.assign(**{y: df[y].astype(int)})
    if agg != "nunique" and kind_of(data[y]) != "numeric":
        raise SpecError(f"Cannot compute the {agg} of '{y}' because it is not numeric. "
                        "Use a numeric column, or count / nunique instead.")
    label = f"{agg.title()} of {y}"
    if label in groups:
        label += " (value)"
    res = data.groupby(groups, observed=True)[y].agg(agg).reset_index(name=label)
    return res, label


def _fmt_for(values):
    try:
        v = pd.Series(values).dropna()
        return ",.0f" if (v % 1 == 0).all() else ",.2f"
    except Exception:
        return ",.2f"


def _sample(d, limit=SAMPLE_LIMIT):
    """Random sample used for plotting only; statistics are always computed on all rows."""
    if len(d) <= limit:
        return d, None
    return d.sample(limit, random_state=0), f"Plot drawn from a random sample of {limit:,} of {len(d):,} rows (statistics use all rows)."


def _cat_to_str(res, col, source):
    if kind_of(source[col]) == "categorical":
        res = res.copy()
        res[col] = res[col].astype(str)
    return res


# --------------------------------------------------------------------------
# Builders: each returns (fig, table, title, notes, facts)
# --------------------------------------------------------------------------
def _bar(d, s, pie=False):
    x, y, color = s["x"], s["y"], s["color"]
    if not x:
        raise SpecError("This chart needs an 'x' column to group by.")
    if color == x or pie:
        color = None
    notes = []
    res, label = aggregate(d, [x] + ([color] if color else []), y, s["agg"])
    if res.empty:
        raise SpecError("There is no data to plot after grouping.")
    xk = kind_of(d[x])
    sort = s["sort"] or ("desc" if xk == "categorical" else None)
    tot = res.groupby(x, observed=True)[label].sum()
    order = list((tot.sort_values(ascending=(sort == "asc"), kind="mergesort") if sort else tot.sort_index()).index)
    limit = s["top_n"] or (12 if pie else (MAX_GROUPS if (xk == "categorical" or sort) else 60))
    if len(order) > limit:
        notes.append(f"Showing the {'top' if sort else 'first'} {limit} of {len(order)} groups.")
        order = order[:limit]
    res = res[res[x].isin(order)].copy()
    rank = {v: i for i, v in enumerate(order)}
    res["_o"] = res[x].astype(object).map(rank).astype(float)
    res = res.sort_values("_o", kind="mergesort").drop(columns="_o").reset_index(drop=True)

    title = f"{label} by {x}" + (f" and {color}" if color else "")
    plot = _cat_to_str(res, x, d)
    if pie:
        plot = plot.assign(**{x: plot[x].astype(str)})
        fig = px.pie(plot, names=x, values=label, hole=0.35)
        fig.update_traces(sort=False, textinfo="percent+label")
    else:
        text = _fmt_for(plot[label]) if len(plot) <= 25 else False
        kw = dict(color=color, barmode="group", text_auto=text)
        if s["orientation"] == "h":
            fig = px.bar(plot, x=label, y=x, orientation="h", **kw)
            fig.update_yaxes(autorange="reversed")
            if xk == "numeric":
                fig.update_yaxes(type="category")
        else:
            fig = px.bar(plot, x=x, y=label, **kw)
            if xk == "numeric":
                fig.update_xaxes(type="category")
    return fig, res, title, notes, res.head(40).to_csv(index=False)


def _line(d, s, area=False):
    x, y, color = s["x"], s["y"], s["color"]
    if not x:
        raise SpecError("A line/area chart needs an 'x' column (usually a date).")
    if color == x:
        color = None
    notes, d2 = [], d
    if kind_of(d[x]) == "datetime":
        bucket = s["time_bucket"]
        if not bucket and d[x].nunique() > 150:
            span = (d[x].max() - d[x].min()).days
            bucket = "month" if span > 1095 else "week"
        if bucket:
            d2 = d.copy()
            d2[x] = d2[x].dt.to_period(BUCKET_FREQ[bucket]).dt.to_timestamp()
            notes.append(f"Grouped by {bucket}.")
    res, label = aggregate(d2, [x] + ([color] if color else []), y, s["agg"])
    if color and res[color].nunique() > 12:
        keep = res.groupby(color, observed=True)[label].sum().nlargest(12).index
        res = res[res[color].isin(keep)]
        notes.append("Showing the 12 largest series only.")
    res = res.sort_values(x).reset_index(drop=True)
    plot = res.assign(**{color: res[color].astype(str)}) if color else res
    if area:
        fig = px.area(plot, x=x, y=label, color=color)
    else:
        fig = px.line(plot, x=x, y=label, color=color, markers=bool(res[x].nunique() <= 60))
    return fig, res, f"{label} over {x}", notes, res.head(60).to_csv(index=False)


def _scatter(d, s):
    x, y, color = s["x"], s["y"], s["color"]
    if not x or not y:
        raise SpecError("A scatter plot needs both an 'x' and a 'y' numeric column.")
    for c in (x, y):
        if kind_of(d[c]) != "numeric":
            raise SpecError(f"Scatter plots need numeric columns, but '{c}' is not numeric.")
    cols = [x, y] + ([color] if color and color not in (x, y) else [])
    data = d[cols].dropna(subset=[x, y])
    if data.empty:
        raise SpecError("There are no rows with both values present.")
    n, notes = len(data), []
    plot = data
    if n > 5000:
        plot = data.sample(5000, random_state=0)
        notes.append(f"Plotted a random sample of 5,000 of {n:,} points.")
    ccol = color if color in cols else None
    if ccol and kind_of(plot[ccol]) == "categorical":
        plot = plot.assign(**{ccol: plot[ccol].astype(str)})
    fig = px.scatter(plot, x=x, y=y, color=ccol, opacity=0.7)
    xv, yv = data[x].astype(float).to_numpy(), data[y].astype(float).to_numpy()
    r = float(np.corrcoef(xv, yv)[0, 1]) if n > 2 and xv.std() > 0 and yv.std() > 0 else float("nan")
    slope = intercept = float("nan")
    if n >= 3 and xv.std() > 0:
        slope, intercept = np.polyfit(xv, yv, 1)
        if s["trendline"]:
            xs = np.array([xv.min(), xv.max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Trend",
                                     line=dict(color="#d62728", dash="dash")))
    table = pd.DataFrame({"Points": [n], "Pearson r": [round(r, 4)], "Slope": [round(float(slope), 4)],
                          "Intercept": [round(float(intercept), 4)]})
    facts = f"{n} points; Pearson r = {r:.3f}; slope = {slope:.4f}; intercept = {intercept:.4f}"
    return fig, table, f"{y} vs {x}", notes, facts


def _hist(d, s):
    x = s["x"]
    if not x:
        raise SpecError("A histogram needs an 'x' column.")
    if kind_of(d[x]) == "categorical":
        fig, table, title, notes, facts = _bar(d, dict(s, chart_type="bar", agg="count", y=None))
        notes.insert(0, f"'{x}' is categorical, so counts per category are shown instead of a histogram.")
        return fig, table, title, notes, facts
    color, notes = (s["color"] if s["color"] != x else None), []
    if color and d[color].nunique() > 8:
        color = None
        notes.append("Colour grouping ignored (too many groups).")
    base, snote = _sample(d[[x] + ([color] if color else [])])
    if snote:
        notes.append(snote)
    plot = base.assign(**{color: base[color].astype(str)}) if color else base
    fig = px.histogram(plot, x=x, color=color, nbins=s["bins"], marginal="box", barmode="overlay",
                       opacity=0.65 if color else 0.9)
    desc = d[x].describe()
    facts = ""
    if kind_of(d[x]) == "numeric":
        desc = desc.round(3)
        facts = f"skew = {d[x].skew():.2f}; "
    table = desc.to_frame(name="Value").rename_axis("Statistic").reset_index()
    return fig, table, f"Distribution of {x}", notes, facts + table.to_csv(index=False)


def _box(d, s, violin=False):
    x, y, color = s["x"], s["y"], s["color"]
    if y is None and x is not None and kind_of(d[x]) == "numeric":
        x, y = None, x
    if y is None or kind_of(d[y]) != "numeric":
        raise SpecError("Box / violin plots need a numeric 'y' column.")
    if x is not None and not is_groupable(d[x]):
        raise SpecError(f"'{x}' has too many distinct numeric values to group by. Pick a categorical column.")
    notes, data = [], d
    if x:
        counts = d[x].value_counts()
        if len(counts) > 20:
            data = d[d[x].isin(counts.index[:20])]
            notes.append("Showing the 20 most frequent groups.")
        data = data.assign(**{x: data[x].astype(str)})
    color = color or x
    order = None
    if x:
        vals = list(pd.unique(data[x]))
        try:
            vals = sorted(vals, key=float)
        except ValueError:
            vals = sorted(vals)
        order = {x: vals}
    plot, snote = _sample(data[[c for c in dict.fromkeys([x, y, color]) if c]])
    if snote:
        notes.append(snote)
    if violin:
        fig = px.violin(plot, x=x, y=y, color=color, box=True, points=False, category_orders=order)
    else:
        fig = px.box(plot, x=x, y=y, color=color, points="outliers", category_orders=order)
    if x:
        table = data.groupby(x, observed=True)[y].agg(["count", "mean", "median", "min", "max", "std"]).round(3).reset_index()
        table = table.sort_values(x, key=lambda c: c.map(lambda v: order[x].index(v)))
        title = f"{y} by {x}"
    else:
        table = d[y].describe().round(3).to_frame(name="Value").rename_axis("Statistic").reset_index()
        title = f"Distribution of {y}"
    return fig, table.reset_index(drop=True), title, notes, table.to_csv(index=False)


def _corr(d, s):
    num = [c for c in (s["columns"] or col_kinds(d)["numeric"]) if kind_of(d[c]) == "numeric" and d[c].nunique() > 1]
    if len(num) < 2:
        raise SpecError("A correlation heatmap needs at least two numeric columns with more than one distinct value.")
    notes = []
    if len(num) > 25:
        num = num[:25]
        notes.append("Showing the first 25 numeric columns.")
    corr = d[num].corr(method=s["method"])
    fig = px.imshow(corr, text_auto=".2f" if len(num) <= 14 else False, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_xaxes(tickangle=-40)
    pairs = top_correlations(d[num], n=8, method=s["method"])
    facts = "Strongest pairs:\n" + pairs.to_csv(index=False)
    table = corr.round(3).rename_axis("Column").reset_index()
    return fig, table, f"Correlation heatmap ({s['method'].title()})", notes, facts


def _pivot(d, s):
    x, y = s["x"], s["y"]
    if not x or not y:
        raise SpecError("A pivot heatmap needs both 'x' and 'y' grouping columns.")
    for c in (x, y):
        if not is_groupable(d[c]):
            raise SpecError(f"'{c}' has too many distinct values for a heatmap axis.")
    notes = []
    for c in (x, y):
        top = d[c].value_counts().index[:25]
        if d[c].nunique() > 25:
            notes.append(f"'{c}' limited to its 25 most frequent values.")
        d = d[d[c].isin(top)]
    res, label = aggregate(d, [x, y], s["value"], s["agg"])
    piv = res.pivot(index=y, columns=x, values=label)
    piv = piv.rename(index=str, columns=str)
    fig = px.imshow(piv, text_auto=_fmt_for(res[label]) if piv.size <= 150 else False,
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_xaxes(title_text=x, side="bottom")
    fig.update_yaxes(title_text=y)
    table = piv.rename_axis(y).reset_index()
    return fig, table, f"{label} by {x} and {y}", notes, table.to_csv(index=False)


def _table(d, s):
    x, y, color, agg = s["x"], s["y"], s["color"], s["agg"]
    notes = []
    if x:
        groups = [x] + ([color] if color and color != x else [])
        res, label = aggregate(d, groups, y, agg)
        if s["sort"] or kind_of(d[x]) == "categorical":
            res = res.sort_values(label, ascending=(s["sort"] == "asc"), kind="mergesort")
        else:
            res = res.sort_values(groups)
        n = s["top_n"] or 50
        if len(res) > n:
            notes.append(f"Showing {n} of {len(res)} rows.")
            res = res.head(n)
        res = res.reset_index(drop=True)
        return None, res, f"{label} by {x}", notes, res.head(40).to_csv(index=False)
    if not s["columns"] and not s["sort_by"]:
        if y is None or agg == "count":
            res = pd.DataFrame({"Metric": ["Rows"], "Value": [len(d)]})
        else:
            a = agg or "mean"
            ser = d[y].astype(int) if pd.api.types.is_bool_dtype(d[y]) else d[y]
            if a != "nunique" and kind_of(ser) != "numeric":
                raise SpecError(f"Cannot compute the {a} of '{y}' because it is not numeric.")
            val = ser.nunique() if a == "nunique" else getattr(ser, a)()
            res = pd.DataFrame({"Metric": [f"{a.title()} of {y}"], "Value": [round(float(val), 4)]})
        return None, res, "Result", notes, res.to_csv(index=False)
    cols = s["columns"] or list(d.columns)[:15]
    sort_by = s["sort_by"] or y
    dd = d
    if sort_by:
        dd = d.sort_values(sort_by, ascending=(s["sort"] == "asc"), na_position="last")
    n = s["top_n"] or 20
    res = dd[cols].head(n).reset_index(drop=True)
    return None, res, "Rows", notes, res.head(30).to_csv(index=False)


_BUILDERS = {
    "bar": _bar, "pie": lambda d, s: _bar(d, s, pie=True),
    "line": _line, "area": lambda d, s: _line(d, s, area=True),
    "scatter": _scatter, "histogram": _hist,
    "box": _box, "violin": lambda d, s: _box(d, s, violin=True),
    "correlation_heatmap": _corr, "pivot_heatmap": _pivot, "table": _table,
}


def _style(fig, title):
    fig.update_layout(title=dict(text=title, x=0.0, xanchor="left"), margin=dict(l=10, r=10, t=60, b=10),
                      height=440, font=dict(size=13))


def build_chart(spec, df):
    """Validate a spec, apply filters, aggregate and render. Raises SpecError with a friendly message."""
    s = normalize_spec(spec, df)
    notes, d = [], df
    if s["filters"]:
        d, fnotes = apply_filters(d, s["filters"])
        if d.empty:
            raise SpecError("No rows match the filters: " + "; ".join(fnotes))
        notes.append(f"Filtered to {len(d):,} of {len(df):,} rows ({'; '.join(fnotes)}).")
    fig, table, title, bnotes, facts = _BUILDERS[s["chart_type"]](d, s)
    title = s["title"] or title
    if fig is not None:
        _style(fig, title)
    return ChartResult(fig=fig, table=table, title=title, spec=s, notes=notes + bnotes, facts=facts)


def describe_spec(spec):
    bits = [str(spec.get("chart_type"))]
    for k in ("x", "y", "color", "value"):
        if spec.get(k):
            bits.append(f"{k}={spec[k]}")
    if spec.get("agg"):
        bits.append(f"agg={spec['agg']}")
    if spec.get("filters"):
        bits.append("filters: " + ", ".join(f"{f['column']} {f['op']} {f['value']}" for f in spec["filters"]))
    return " · ".join(bits)


# --------------------------------------------------------------------------
# Rule-based chart requests (used when there is no API key or the LLM fails)
# --------------------------------------------------------------------------
_SUM_HINT = re.compile(r"(revenue|sales|amount|total|units|quantity|qty|profit|cost|spend|orders|count)", re.I)


def find_columns(text, df):
    """Column names mentioned in `text`, in order of appearance (longest names matched first)."""
    t, found = text, []
    for c in sorted(df.columns, key=lambda c: -len(str(c))):
        name = norm_text(c)
        if not name:
            continue
        m = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", t)
        if m:
            found.append((m.start(), c))
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    return [c for _, c in sorted(found)]


def _agg_from_words(ql, metric):
    if re.search(r"\b(total|sum)\b", ql):
        return "sum"
    if re.search(r"\b(average|mean|avg)\b", ql):
        return "mean"
    if re.search(r"\bmedian\b", ql):
        return "median"
    if re.search(r"\b(max|maximum|highest)\b", ql):
        return "max"
    if re.search(r"\b(min|minimum|lowest)\b", ql):
        return "min"
    return "sum" if metric and _SUM_HINT.search(metric) else "mean"


def heuristic_spec(question, df):
    ql = norm_text(question)
    kinds = col_kinds(df)
    num, dt = set(kinds["numeric"]), set(kinds["datetime"])
    mentioned = find_columns(ql, df)
    m_num = [c for c in mentioned if c in num]
    groupable = [c for c in mentioned if is_groupable(df[c], LOW_CARD + 18) and c not in dt]
    has = lambda p: re.search(p, ql) is not None  # noqa: E731

    if has(r"\bheat ?map\b") and len(groupable) >= 2:
        x, y = groupable[:2]
        val = next((c for c in m_num if c not in (x, y)), None)
        return {"chart_type": "pivot_heatmap", "x": x, "y": y, "value": val,
                "agg": _agg_from_words(ql, val) if val else "count"}
    if has(r"\b(heat ?map|correlation|correlations|correlate|correlated)\b") and not has(r"\b(scatter|vs|versus)\b"):
        return {"chart_type": "correlation_heatmap"}
    if has(r"\b(histogram|distribution|distributed|spread)\b") and m_num:
        return {"chart_type": "histogram", "x": m_num[0]}
    if has(r"\b(scatter|vs|versus|against|relationship)\b") and len(m_num) >= 2:
        return {"chart_type": "scatter", "x": m_num[0], "y": m_num[1], "trendline": True}
    if has(r"\b(box ?plot|boxplot|outliers?)\b") and m_num:
        grp = next((c for c in groupable if c not in m_num), None)
        return {"chart_type": "box", "y": m_num[0], "x": grp}
    if has(r"\bviolin\b") and m_num:
        grp = next((c for c in groupable if c not in m_num), None)
        return {"chart_type": "violin", "y": m_num[0], "x": grp}
    if has(r"\b(trend|over time|timeline|line|monthly|weekly|yearly|daily|quarterly|area)\b"):
        x = next((c for c in mentioned if c in dt), None) or (kinds["datetime"][0] if kinds["datetime"] else None)
        y = next((c for c in m_num), None)
        if x and y:
            bucket = next((b for b in ("daily", "weekly", "monthly", "quarterly", "yearly") if has(rf"\b{b}\b")), None)
            return {"chart_type": "area" if has(r"\barea\b") else "line", "x": x, "y": y,
                    "agg": _agg_from_words(ql, y), "time_bucket": bucket}
    if has(r"\b(pie|donut|share|proportion|percentage)\b|\b(bar|bars|count|counts|breakdown|compare|comparison|by|per)\b|\b(chart|plot|graph)\b"):
        by = re.search(r"\b(by|per)\b(.*)$", ql)
        right = find_columns(by.group(2), df) if by else []
        grp = next((c for c in right if is_groupable(df[c], LOW_CARD + 18)), None) or (groupable[0] if groupable else None)
        metric = next((c for c in m_num if c != grp), None)
        if grp:
            ct = "pie" if has(r"\b(pie|donut|share|proportion)\b") else "bar"
            if metric:
                return {"chart_type": ct, "x": grp, "y": metric, "agg": _agg_from_words(ql, metric)}
            return {"chart_type": ct, "x": grp, "agg": "count"}
    return None
# --------------------------------------------------------------------------
# Automatic dashboard figures
# --------------------------------------------------------------------------
def _grid(titles, cols=3, height_per_row=290):
    rows = -(-len(titles) // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.14 / max(rows, 1) * 3)
    fig.update_layout(height=height_per_row * rows + 40, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig, cols


def fig_corr(df, method="pearson"):
    return build_chart({"chart_type": "correlation_heatmap", "method": method}, df).fig


def fig_hist_grid(df, cols):
    fig, ncols = _grid(list(cols))
    for i, c in enumerate(cols):
        vals = df[c].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        counts, edges = np.histogram(vals, bins=30)  # pre-binned: exact counts, tiny payload
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), marker_color="#4c78a8",
                             hovertemplate=f"{c}: %{{x:.3g}}<br>count: %{{y}}<extra></extra>"),
                      row=i // ncols + 1, col=i % ncols + 1)
    fig.update_layout(bargap=0.02)
    return fig


def fig_box_grid(df, cols):
    fig, ncols = _grid(list(cols))
    for i, c in enumerate(cols):
        s = df[c].dropna()
        if len(s) > SAMPLE_LIMIT // 2:
            s = s.sample(SAMPLE_LIMIT // 2, random_state=0)
        fig.add_trace(go.Box(y=s, name=str(c), boxpoints="outliers", marker_color="#f58518"),
                      row=i // ncols + 1, col=i % ncols + 1)
    return fig


def fig_cat_grid(df, cols, top=8):
    fig, ncols = _grid(list(cols))
    for i, c in enumerate(cols):
        vc = df[c].value_counts().head(top)[::-1]
        vc.index = vc.index.astype(str)
        fig.add_trace(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color="#54a24b"),
                      row=i // ncols + 1, col=i % ncols + 1)
    return fig


def fig_missing_bar(df):
    pct = (df.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    if pct.empty:
        return None
    fig = px.bar(x=pct.values, y=pct.index.astype(str), orientation="h", labels={"x": "% missing", "y": "Column"},
                 text=[f"{v:.1f}%" for v in pct.values])
    fig.update_layout(title=dict(text="Missing values by column", x=0), height=max(240, 34 * len(pct) + 100),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_missing_matrix(df, max_rows=300, max_cols=60):
    d = df.iloc[:, :max_cols]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0).sort_index()
    z = d.isna().astype(int).to_numpy()
    if z.sum() == 0:
        return None
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in d.columns], y=[str(i) for i in d.index],
                               colorscale=[[0, "#dfe8f3"], [1, "#d62728"]], showscale=False))
    fig.update_yaxes(autorange="reversed", showticklabels=False, title_text="Rows (sampled)")
    fig.update_layout(title=dict(text="Where values are missing (red = missing)", x=0), height=420,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_scatter_matrix(df, cols, max_points=2000):
    d = df[list(cols)].dropna()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)
    fig = px.scatter_matrix(d, dimensions=list(cols), opacity=0.5)
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10))
    return fig
