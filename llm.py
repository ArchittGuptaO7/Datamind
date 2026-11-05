"""LLM access (Groq by default, OpenAI as fallback) + the planner that turns a question into a chart spec."""
import json
import os
import re

import streamlit as st

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_PREFERRED = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.6-27b",
]
_NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "safeguard", "embed", "playai")


class PlanError(RuntimeError):
    """The model did not return a usable plan."""


# --------------------------------------------------------------------------
# Provider config
# --------------------------------------------------------------------------
def _clean(value):
    return (value or "").strip().strip('"').strip("'").strip()


def get_llm_config():
    groq_key = _clean(os.getenv("GROQ_API_KEY"))
    if groq_key:
        return {
            "name": "Groq",
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": st.session_state.get("model_override") or _clean(os.getenv("GROQ_MODEL")) or DEFAULT_GROQ_MODEL,
        }
    openai_key = _clean(os.getenv("OPENAI_API_KEY"))
    if openai_key:
        return {
            "name": "OpenAI",
            "api_key": openai_key,
            "base_url": None,
            "model": _clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini",
        }
    return None


def llm_available():
    return OpenAI is not None and get_llm_config() is not None


def make_client(cfg):
    kwargs = {"api_key": cfg["api_key"], "timeout": 45.0}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def explain_error(exc):
    name = type(exc).__name__
    hints = {
        "AuthenticationError": "The provider rejected the key (invalid, revoked, or copied incorrectly).",
        "PermissionDeniedError": "Access denied: the key lacks permission, or your network/region is blocked.",
        "APIConnectionError": "Could not reach the server: check internet, VPN/proxy, or whether your network blocks api.groq.com.",
        "APITimeoutError": "The request timed out: network is slow or blocked.",
        "RateLimitError": "Rate limit or quota reached: wait a minute or check your plan.",
        "NotFoundError": "Model or endpoint not found: the model name may be retired.",
        "BadRequestError": "The request was rejected: the model name may be wrong or retired.",
    }
    return f"{name}: {exc}\n\n{hints.get(name, '')}".strip()


def pick_fallback_model(available, current):
    for m in GROQ_PREFERRED:
        if m in available and m != current:
            return m
    for m in available:
        if m != current and not any(x in m.lower() for x in _NOT_CHAT):
            return m
    return None


def is_model_error(exc):
    text = str(exc).lower()
    return type(exc).__name__ == "NotFoundError" or "model_not_found" in text or "does not exist" in text


def complete(messages, temperature=0.1):
    """One chat completion. Retries once with an available model if the configured one is missing."""
    cfg = get_llm_config()
    if cfg is None or OpenAI is None:
        raise PlanError("No API key configured.")
    client = make_client(cfg)

    def call(model):
        r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        text = r.choices[0].message.content
        if not text or not text.strip():
            raise PlanError("The model returned an empty response.")
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    try:
        return call(cfg["model"])
    except Exception as exc:
        if cfg["name"] != "Groq" or not is_model_error(exc):
            raise
        available = [m.id for m in client.models.list().data]
        alt = pick_fallback_model(available, cfg["model"])
        if not alt:
            raise
        text = call(alt)
        st.session_state["model_override"] = alt
        return text


# --------------------------------------------------------------------------
# JSON handling
# --------------------------------------------------------------------------
def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", t))  # tolerate trailing commas


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
_SYSTEM = """You are DataMind AI, a data analyst that answers questions about a pandas DataFrame by choosing exactly ONE action.
You never see the full data and you never write code: you reply with a single JSON object and nothing else.

JSON format:
{"type": "chart" | "table" | "answer", "answer": "<text>", "spec": {...}}

"spec" (only for type chart/table) has these fields; use column names EXACTLY as listed in the schema, or null:
- chart_type: bar | line | area | scatter | histogram | box | violin | pie | correlation_heatmap | pivot_heatmap | table
- x, y, color: column names. value: numeric column (pivot_heatmap only).
- agg: mean | sum | count | median | min | max | nunique  (how y is aggregated for each x group; use count when there is no y)
- filters: [{"column": "...", "op": "==|!=|>|>=|<|<=|in|not_in|contains", "value": ...}]
- top_n (int), sort ("asc"|"desc"), bins (int, histogram), orientation ("v"|"h"), trendline (true|false, scatter),
  time_bucket (day|week|month|quarter|year, for line/area with a date x), method (pearson|spearman),
  columns (list) and sort_by (column) for row listings, title (short).

How each chart type uses the fields:
- bar / pie: x = category to group by, y = numeric metric (optional), agg. color adds grouped bars (bar only).
- line / area: x = date or ordered column, y = metric, agg, optional color and time_bucket.
- scatter: x and y both numeric; optional color; trendline true when the user asks about relationship/trend.
- histogram: x numeric. box / violin: y numeric, x optional category.
- correlation_heatmap: no columns needed (optionally columns=[...]). pivot_heatmap: x and y categories, value numeric, agg.
- table: with x -> aggregated table; with sort_by/columns/top_n -> row listing; with only y+agg -> a single number.

Rules:
1. Use "chart" when the user asks to show/plot/draw/visualize/graph anything or when a picture clearly answers best.
2. Use "table" for exact numbers: group-by results, rankings, top-N rows, crosstabs, a single filtered statistic.
3. Use "answer" for explanations, summaries, advice, and row look-ups. Base it ONLY on the schema and the similar rows; never invent numbers.
4. Any group-by / "by X" / rate / share question must be a chart or table so the numbers are computed exactly.
   A rate of a 0/1 column is agg "mean". Category values in filters must match the schema examples.
5. If the request cannot be done with the available columns, use "answer" and say what is missing.
6. If the user is refining the previous chart ("make it a pie", "now by region", "only 2025"), start from the previous spec and change only what was asked.
7. Keep "answer" short (one sentence) for chart/table; a few sentences for type answer.

Examples:
Q: average Fare by Pclass as a bar chart
{"type":"chart","answer":"Average fare per class.","spec":{"chart_type":"bar","x":"Pclass","y":"Fare","agg":"mean","title":"Average fare by class"}}
Q: how does revenue relate to units sold?
{"type":"chart","answer":"Revenue vs units.","spec":{"chart_type":"scatter","x":"Units","y":"Revenue","trendline":true}}
Q: top 5 regions by total revenue in 2025
{"type":"table","answer":"Top regions in 2025.","spec":{"chart_type":"table","x":"Region","y":"Revenue","agg":"sum","sort":"desc","top_n":5,"filters":[{"column":"Date","op":">=","value":"2025-01-01"},{"column":"Date","op":"<=","value":"2025-12-31"}]}}
Q: which columns look most useful for predicting Survived?
{"type":"answer","answer":"...","spec":null}"""


def plan_request(question, schema, retrieved=(), history=(), last_spec=None, repair=None):
    """Ask the model for a plan. `repair` = (previous_json_text, error_message) to request a corrected spec."""
    parts = [f"Dataset schema:\n{schema}"]
    if last_spec:
        parts.append("Previous chart spec (for follow-up requests):\n" + json.dumps(last_spec, default=str))
    if history:
        parts.append("Recent user questions:\n" + "\n".join(f"- {h}" for h in history[-3:]))
    if retrieved:
        parts.append("Rows most similar to the question (a sample, not the whole data):\n" + "\n".join(retrieved))
    parts.append(f"User question: {question}\n\nReply with the JSON object only.")
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
    if repair:
        messages += [{"role": "assistant", "content": repair[0]},
                     {"role": "user", "content": f"That spec failed: {repair[1]}\nReturn a corrected JSON object only."}]

    text = complete(messages)
    try:
        plan = parse_json(text)
    except ValueError:
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object."}]
        try:
            plan = parse_json(complete(messages))
        except ValueError as exc:
            raise PlanError("The model did not return valid JSON.") from exc
    if not isinstance(plan, dict):
        raise PlanError("The model returned an unexpected format.")
    ptype = str(plan.get("type") or "").lower()
    if ptype not in ("chart", "table", "answer"):
        ptype = "chart" if plan.get("spec") else "answer"
    plan["type"] = ptype
    plan["_raw"] = json.dumps(plan.get("spec"), default=str)
    return plan


def explain_result(question, title, facts, notes=()):
    """2-4 sentence insight grounded in the numbers that were actually computed. Returns None on failure."""
    if not facts:
        return None
    prompt = (
        f"The user asked: {question}\nChart/table: {title}\n"
        + ("Notes: " + "; ".join(notes) + "\n" if notes else "")
        + f"\nComputed data (exact, from the full dataset):\n{facts[:3500]}\n\n"
        "Write 2-4 sentences explaining what this shows: name the biggest/smallest values or the key trend, "
        "with numbers. Use ONLY numbers that appear above. No headings, no preamble."
    )
    try:
        return complete([
            {"role": "system", "content": "You are a precise data analyst. Never invent numbers."},
            {"role": "user", "content": prompt},
        ])
    except Exception:
        return None


def write_report(schema, findings):
    prompt = (
        f"Dataset schema:\n{schema}\n\nAutomatic findings:\n" + "\n".join(f"- {f}" for f in findings)
        + "\n\nWrite a concise analyst briefing (max 180 words) in markdown with three short sections: "
          "**What this data is**, **Key observations**, **Suggested next steps** (3 bullets: cleaning, visuals, models). "
          "Use only facts from above."
    )
    return complete([
        {"role": "system", "content": "You are a careful, business-friendly data analyst."},
        {"role": "user", "content": prompt},
    ])
