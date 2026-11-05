"""Data loading, profiling, exact (no-LLM) answers and TF-IDF retrieval."""
import io
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_RAG_ROWS = 1500
LOW_CARD = 12  # numeric columns with <= this many distinct values behave like categories

_DATE_RE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})")


# --------------------------------------------------------------------------
# Loading & preparation
# --------------------------------------------------------------------------
def read_file(file_bytes, filename):
    def bio():
        return io.BytesIO(file_bytes)

    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(bio())
    try:
        df = pd.read_csv(bio())
    except UnicodeDecodeError:
        df = pd.read_csv(bio(), encoding="latin-1")
    if df.shape[1] == 1 and any(sep in str(df.columns[0]) for sep in (";", "\t", "|")):
        try:  # semicolon / tab / pipe separated files
            df = pd.read_csv(bio(), sep=None, engine="python", encoding_errors="replace")
        except Exception:
            pass
    return df


def kind_of(s):
    """'numeric' | 'datetime' | 'categorical' (bool counts as categorical)."""
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def prepare(df):
    """Clean column names (unique strings) and auto-detect date columns."""
    df = df.copy()
    used, cols = set(), []
    for c in df.columns:
        base = str(c).strip() or "column"
        name, i = base, 1
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        cols.append(name)
    df.columns = cols

    for c in df.columns:
        s = df[c]
        if kind_of(s) != "categorical" or pd.api.types.is_bool_dtype(s):
            continue
        sample = s.dropna().head(50).astype(str)
        if len(sample) == 0 or sample.str.match(_DATE_RE).mean() < 0.9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().sum() >= 0.9 * s.notna().sum():
            df[c] = parsed
    return df


def col_kinds(df):
    out = {"numeric": [], "datetime": [], "categorical": []}
    for c in df.columns:
        out[kind_of(df[c])].append(c)
    return out


def is_groupable(s, limit=30):
    return kind_of(s) == "categorical" or s.nunique(dropna=True) <= limit


def basic_summary(df):
    k = col_kinds(df)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": k["numeric"],
        "datetime_columns": k["datetime"],
        "categorical_columns": k["categorical"],
    }


def _fmt(x):
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


def schema_text(df, max_cols=60):
    """Compact description of the dataset that is sent to the LLM."""
    lines = [f"Rows: {len(df)}; Columns: {df.shape[1]}"]
    for c in list(df.columns)[:max_cols]:
        s, k = df[c], kind_of(df[c])
        miss, nun = int(s.isna().sum()), int(s.nunique(dropna=True))
        if k == "numeric":
            if nun <= LOW_CARD and nun < len(df):
                vals = ", ".join(_fmt(v) for v in sorted(s.dropna().unique()))
                desc = f"numeric, few distinct values: {vals}"
            elif nun:
                desc = f"numeric, min {_fmt(s.min())}, max {_fmt(s.max())}, mean {_fmt(s.mean())}"
            else:
                desc = "numeric (all missing)"
        elif k == "datetime":
            desc = "datetime" if s.notna().sum() == 0 else f"datetime, {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
        else:
            top = s.value_counts().head(5)
            desc = f"categorical, {nun} unique, top values: " + ", ".join(f"{i} ({n})" for i, n in top.items())
        lines.append(f"- {c}: {desc}" + (f"; {miss} missing" if miss else ""))
    if df.shape[1] > max_cols:
        lines.append(f"(+{df.shape[1] - max_cols} more columns not shown)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Insights (rule-based, no LLM)
# --------------------------------------------------------------------------
def top_correlations(df, n=10, method="pearson"):
    num = [c for c in col_kinds(df)["numeric"] if df[c].nunique() > 1][:40]
    cols = ["Column A", "Column B", "Correlation"]
    if len(num) < 2:
        return pd.DataFrame(columns=cols)
    corr = df[num].corr(method=method)
    rows = [(a, b, corr.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if pd.notna(corr.loc[a, b])]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out["_abs"] = out["Correlation"].abs()
    return out.sort_values("_abs", ascending=False).drop(columns="_abs").head(n).round(3).reset_index(drop=True)


def outlier_table(df):
    rows = []
    for c in col_kinds(df)["numeric"]:
        s = df[c].dropna()
        if s.nunique() <= 2:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        rows.append({"Column": c, "Outliers": n_out, "Outlier %": round(100 * n_out / len(s), 2),
                     "Skew": round(float(s.skew()), 2)})
    return pd.DataFrame(rows, columns=["Column", "Outliers", "Outlier %", "Skew"])


def auto_findings(df):
    out, n = [], len(df)
    kinds = col_kinds(df)
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    high = miss[miss >= 20]
    if len(high):
        out.append(f"⚠️ **{len(high)} column(s) have 20%+ missing values:** "
                   + ", ".join(f"{c} ({v:.0f}%)" for c, v in high.head(5).items()))
    elif len(miss) and miss.iloc[0] > 0:
        out.append(f"🟡 Missing values are limited (worst: **{miss.index[0]}** at {miss.iloc[0]:.1f}%).")
    else:
        out.append("✅ No missing values.")
    dup = int(df.duplicated().sum())
    if dup:
        out.append(f"⚠️ **{dup:,} duplicate rows** ({dup / max(n, 1):.1%}).")
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        out.append("🟡 Constant columns (carry no information): " + ", ".join(const[:6]))
    ids = [c for c in kinds["categorical"] if n >= 20 and df[c].nunique() >= 0.9 * n]
    if ids:
        out.append("🆔 Likely ID / free-text columns (almost every value unique): " + ", ".join(ids[:6]))
    ot = outlier_table(df)
    if len(ot):
        skewed = ot[ot["Skew"].abs() > 1].sort_values("Skew", key=lambda x: x.abs(), ascending=False).head(3)
        if len(skewed):
            out.append("📈 Strongly skewed: " + ", ".join(f"{r.Column} ({r.Skew:+.1f})" for r in skewed.itertuples())
                       + " — consider a log transform.")
        flagged = ot[ot["Outlier %"] >= 5].head(3)
        if len(flagged):
            out.append("🎯 Many outliers (IQR rule): " + ", ".join(f"{r.Column} ({r._3:.1f}%)" for r in flagged.itertuples()))
    pairs = top_correlations(df, n=3)
    strong = pairs[pairs["Correlation"].abs() >= 0.5] if len(pairs) else pairs
    if len(strong):
        out.append("🔗 Strongest relationships: "
                   + "; ".join(f"**{r._1} ↔ {r._2}** (r = {r.Correlation:+.2f})" for r in strong.itertuples()))
    return out


def suggest_questions(df):
    k = col_kinds(df)
    num = [c for c in k["numeric"] if df[c].nunique() > LOW_CARD] or k["numeric"]
    cats = [c for c in df.columns if is_groupable(df[c], LOW_CARD) and 1 < df[c].nunique() <= LOW_CARD]
    out = ["Show a correlation heatmap"]
    if num:
        out.append(f"Show the distribution of {num[0]}")
    if num and cats:
        out.append(f"Average {num[0]} by {cats[0]} as a bar chart")
    if len(num) >= 2:
        out.append(f"Scatter plot of {num[0]} vs {num[1]} with a trendline")
    if num and k["datetime"]:
        out.append(f"Show the monthly trend of {num[0]}")
    if len(cats) >= 2 and num:
        out.append(f"Heatmap of average {num[0]} by {cats[0]} and {cats[1]}")
    out.append("Summarize this dataset and its data quality")
    return out[:6]


# --------------------------------------------------------------------------
# Demo data
# --------------------------------------------------------------------------
def demo_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    category = rng.choice(["Electronics", "Clothing", "Home", "Sports", "Beauty"], n, p=[.28, .25, .2, .15, .12])
    base = {"Electronics": 320, "Clothing": 60, "Home": 110, "Sports": 85, "Beauty": 40}
    region = rng.choice(["North", "South", "East", "West"], n)
    segment = rng.choice(["Consumer", "Corporate", "Small Business"], n, p=[.55, .25, .2])
    units = rng.integers(1, 9, n)
    season = 1 + 0.25 * np.sin((dates.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi)
    price = np.array([base[c] for c in category]) * rng.normal(1, .12, n) * season
    discount = rng.choice([0, .05, .1, .15, .2, .3], n, p=[.35, .2, .2, .1, .1, .05])
    revenue = units * price * (1 - discount)
    age = np.clip(rng.normal(38, 12, n), 18, 80).round()
    satisfaction = np.clip(np.round(3.5 + rng.normal(0, .9, n) + 0.9 * (discount > .15)), 1, 5)
    df = pd.DataFrame({
        "Date": dates, "Region": region, "Category": category, "Segment": segment,
        "Units": units, "Unit_Price": price.round(2), "Discount": discount, "Revenue": revenue.round(2),
        "Customer_Age": age.astype(int), "Satisfaction": satisfaction,
    }).sort_values("Date").reset_index(drop=True)
    df.loc[rng.choice(n, 60, replace=False), "Satisfaction"] = np.nan
    return df


# --------------------------------------------------------------------------
# Exact answers without an LLM
# --------------------------------------------------------------------------
def norm_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(text).lower())).strip()


def _has(pattern, text):
    return re.search(pattern, text) is not None


_CHART_RE = (r"\b(chart|plot|graph|visuali[sz]e|visual|visualization|heat ?map|histogram|scatter|pie|donut|bars?|lines?|"
             r"box ?plot|violin|distribution|trend|trends|correlat\w*|draw|dashboard|breakdown|versus|vs|"
             r"compare|comparison)\b")
_GROUP_RE = r"\b(by|per|each|across|for every|grouped|top \d+|bottom \d+|rank|ranking|where|only|among|for)\b"


def needs_llm(question):
    """True when a question is about charts / group-by / filters, so the simple pandas shortcut would be wrong."""
    q = norm_text(question)
    return _has(_CHART_RE, q) or _has(_GROUP_RE, q)


def local_answer(query, df):
    """Exact answer for simple whole-dataset questions, or None if not handled."""
    q = norm_text(query)
    if _has(r"\b(missing|null|nan|empty)\b", q):
        return f"The dataset contains **{int(df.isna().sum().sum())} missing values** across all columns."
    if _has(r"\bduplicat", q):
        return f"The dataset contains **{int(df.duplicated().sum())} duplicate rows**."
    if _has(r"\b(how many|number of|count of|total)\s+rows\b", q) or q in ("rows", "row count"):
        return f"The dataset has **{len(df):,} rows**."
    if _has(r"\bhow many columns\b", q):
        return f"The dataset has **{len(df.columns)} columns**."
    if _has(r"\b(list|show|what are|which)\b.*\bcolumns\b", q) or q in ("columns", "column names"):
        return "The dataset columns are: " + ", ".join(map(str, df.columns))

    numeric = df[col_kinds(df)["numeric"]]
    if numeric.empty:
        return None
    for col in sorted(numeric.columns, key=lambda c: -len(str(c))):
        name = norm_text(col)
        if not name or not _has(rf"\b{re.escape(name)}\b", q):
            continue
        rest = re.sub(rf"\b{re.escape(name)}\b", " ", q)
        s = numeric[col]
        if _has(r"\b(average|mean|avg)\b", rest):
            return f"The mean of **{col}** is **{s.mean():.2f}**."
        if _has(r"\bmedian\b", rest):
            return f"The median of **{col}** is **{s.median():.2f}**."
        if _has(r"\b(max|maximum|highest|largest|biggest)\b", rest):
            return f"The maximum of **{col}** is **{s.max():.2f}**."
        if _has(r"\b(min|minimum|lowest|smallest)\b", rest):
            return f"The minimum of **{col}** is **{s.min():.2f}**."
        if _has(r"\b(sum|total)\b", rest):
            return f"The sum of **{col}** is **{s.sum():.2f}**."
        if _has(r"\b(std|standard deviation)\b", rest):
            return f"The standard deviation of **{col}** is **{s.std():.2f}**."
    return None


# --------------------------------------------------------------------------
# TF-IDF retrieval (used for row look-up questions)
# --------------------------------------------------------------------------
def build_documents(df, max_rows=MAX_RAG_ROWS):
    docs = []
    for idx, row in df.head(max_rows).iterrows():
        parts = [f"Row {idx}"]
        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        docs.append(" | ".join(parts))
    return docs


def build_index(df):
    documents = build_documents(df)
    if not documents:
        return documents, None, None
    try:
        vec = TfidfVectorizer(stop_words="english")
        return documents, vec, vec.fit_transform(documents)
    except ValueError:  # empty vocabulary
        return documents, None, None


def retrieve(query, documents, vectorizer, matrix, k=6):
    if vectorizer is None or matrix is None:
        return []
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ids = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in ids if scores[i] > 0]
