"""Run with:  python -m pytest -q   (from the project folder)"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyst  # noqa: E402
import charts  # noqa: E402
import data_utils as du  # noqa: E402
import llm  # noqa: E402
from charts import SpecError, build_chart  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return du.prepare(du.demo_dataset())


# ---------------------------------------------------------------- data prep
def test_prepare_detects_dates_and_makes_unique_columns():
    raw = pd.DataFrame([["2024-01-05", 1, 2], ["2024-02-10", 3, 4], ["2024-03-15", 5, 6]], columns=["when", "a", "a"])
    out = du.prepare(raw)
    assert list(out.columns) == ["when", "a", "a_1"]
    assert du.kind_of(out["when"]) == "datetime"


def test_semicolon_csv_is_parsed():
    df = du.read_file(b"a;b;c\n1;2;x\n3;4;y\n", "s.csv")
    assert list(df.columns) == ["a", "b", "c"]


# ---------------------------------------------------------------- exactness
def test_bar_matches_pandas(df):
    res = build_chart({"chart_type": "bar", "x": "Region", "y": "Revenue", "agg": "mean"}, df)
    exp = df.groupby("Region")["Revenue"].mean()
    got = res.table.set_index("Region").iloc[:, 0]
    assert all(got[k] == pytest.approx(exp[k]) for k in exp.index)


def test_table_top_n_is_sorted_and_exact(df):
    res = build_chart({"chart_type": "table", "x": "Category", "y": "Revenue", "agg": "sum", "top_n": 3, "sort": "desc"}, df)
    exp = df.groupby("Category")["Revenue"].sum().nlargest(3)
    assert list(res.table["Category"]) == list(exp.index)
    assert res.table.iloc[:, 1].tolist() == pytest.approx(exp.tolist())


def test_pivot_matches_pandas(df):
    res = build_chart({"chart_type": "pivot_heatmap", "x": "Region", "y": "Category", "value": "Revenue", "agg": "mean"}, df)
    exp = df.pivot_table(index="Category", columns="Region", values="Revenue", aggfunc="mean")
    got = res.table.set_index("Category")
    assert np.allclose(got.loc[exp.index, exp.columns].to_numpy(), exp.to_numpy())


def test_scalar_with_filter(df):
    res = build_chart({"chart_type": "table", "y": "Revenue", "agg": "mean",
                       "filters": [{"column": "region", "op": "==", "value": "NORTH"}]}, df)
    assert res.table["Value"].iloc[0] == pytest.approx(df.loc[df.Region == "North", "Revenue"].mean(), abs=1e-3)


def test_correlation_matches_pandas(df):
    res = build_chart({"chart_type": "correlation_heatmap"}, df)
    assert res.table.set_index("Column").loc["Units", "Revenue"] == pytest.approx(df["Units"].corr(df["Revenue"]), abs=1e-3)


# ---------------------------------------------------------------- validation
def test_column_names_are_resolved_case_insensitively(df):
    assert build_chart({"chart_type": "bar", "x": "region ", "agg": "count"}, df).spec["x"] == "Region"


@pytest.mark.parametrize("spec", [
    {"chart_type": "bar", "x": "Regoin"},
    {"chart_type": "sankey", "x": "Region"},
    {"chart_type": "bar"},
    {"chart_type": "scatter", "x": "Region", "y": "Revenue"},
    {"chart_type": "bar", "x": "Region", "y": "Category", "agg": "mean"},
    {"chart_type": "bar", "x": "Region", "filters": [{"column": "Revenue", "op": ">", "value": 1e12}]},
    {"chart_type": "bar", "x": "Region", "filters": [{"column": "Region", "op": "~~", "value": 1}]},
    "not a dict",
])
def test_bad_specs_raise_friendly_errors(df, spec):
    with pytest.raises(SpecError):
        build_chart(spec, df)


def test_histogram_of_categorical_falls_back_to_counts(df):
    res = build_chart({"chart_type": "histogram", "x": "Region"}, df)
    assert any("categorical" in n for n in res.notes)


def test_every_chart_type_serialises(df):
    specs = [
        {"chart_type": "bar", "x": "Region", "y": "Revenue", "agg": "sum", "color": "Segment"},
        {"chart_type": "pie", "x": "Category"}, {"chart_type": "line", "x": "Date", "y": "Revenue", "agg": "sum"},
        {"chart_type": "area", "x": "Date", "y": "Units", "agg": "sum", "time_bucket": "quarter"},
        {"chart_type": "scatter", "x": "Units", "y": "Revenue", "trendline": True},
        {"chart_type": "histogram", "x": "Revenue"}, {"chart_type": "box", "y": "Revenue", "x": "Category"},
        {"chart_type": "violin", "y": "Revenue", "x": "Segment"}, {"chart_type": "correlation_heatmap"},
    ]
    for spec in specs:
        assert "data" in json.loads(build_chart(spec, df).fig.to_json())


def test_large_data_is_sampled_for_plots_but_stats_use_all_rows():
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"g": rng.choice(["a", "b"], 60_000), "v": rng.normal(size=60_000)})
    res = build_chart({"chart_type": "box", "y": "v", "x": "g"}, big)
    assert any("random sample" in n for n in res.notes)
    assert res.table["count"].sum() == 60_000


# ---------------------------------------------------------------- exact answers & routing
def test_local_answer_does_not_confuse_age_with_average():
    d = pd.DataFrame({"Age": [10, 20, 30], "Fare": [1.0, 2.0, 6.0]})
    assert "Fare" in du.local_answer("what is the average fare", d) and "3.00" in du.local_answer("what is the average fare", d)
    assert "Age" in du.local_answer("average age", d)


def test_needs_llm_routes_group_by_and_charts():
    assert du.needs_llm("average fare by class") and du.needs_llm("show a heatmap")
    assert not du.needs_llm("what is the average fare")


# ---------------------------------------------------------------- offline heuristics
@pytest.mark.parametrize("q,ctype", [
    ("show a correlation heatmap", "correlation_heatmap"),
    ("histogram of Revenue", "histogram"),
    ("scatter revenue vs units", "scatter"),
    ("average revenue by region", "bar"),
    ("total revenue by category as pie", "pie"),
    ("monthly revenue trend", "line"),
    ("boxplot of revenue by category", "box"),
    ("heatmap of average revenue by region and category", "pivot_heatmap"),
])
def test_heuristics_build_valid_charts(df, q, ctype):
    spec = charts.heuristic_spec(q, df)
    assert spec and spec["chart_type"] == ctype
    build_chart(spec, df)


def test_heuristics_return_none_for_chatter(df):
    assert charts.heuristic_spec("tell me a joke", df) is None


# ---------------------------------------------------------------- llm helpers
def test_parse_json_handles_fences_and_trailing_commas():
    assert llm.parse_json('```json\n{"a": 1, "b": [1, 2,],}\n```') == {"a": 1, "b": [1, 2]}
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def test_pick_fallback_model():
    avail = ["whisper-large-v3", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
    assert llm.pick_fallback_model(avail, "openai/gpt-oss-120b") == "openai/gpt-oss-20b"
    assert llm.pick_fallback_model(["whisper-large-v3"], "x") is None


# ---------------------------------------------------------------- orchestration with a fake LLM
def _index(df):
    return du.build_index(df)


def test_analyst_builds_chart_and_repairs_bad_spec(df, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    bad = json.dumps({"type": "chart", "answer": "x", "spec": {"chart_type": "bar", "x": "Regoin", "y": "Revenue", "agg": "mean"}})
    good = json.dumps({"type": "chart", "answer": "ok", "spec": {"chart_type": "bar", "x": "Region", "y": "Revenue", "agg": "mean"}})
    replies = iter([bad, good, "North leads."])
    monkeypatch.setattr(llm, "complete", lambda messages, temperature=0.1: next(replies))
    msg = analyst.answer_question("average revenue by region chart", df, _index(df))
    assert msg["fig"] is not None and msg["content"] == "North leads." and msg["table"].shape[0] == 4


def test_analyst_falls_back_to_rules_when_llm_fails(df, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")

    def boom(messages, temperature=0.1):
        raise ConnectionError("offline")

    monkeypatch.setattr(llm, "complete", boom)
    msg = analyst.answer_question("show a correlation heatmap", df, _index(df))
    assert msg["fig"] is not None and "Rule-based" in msg["badge"]


def test_analyst_without_key_still_charts(df, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert analyst.answer_question("average revenue by region", df, _index(df))["fig"] is not None
    assert "Exact" in analyst.answer_question("what is the average revenue", df, _index(df))["badge"]
