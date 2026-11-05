"""Turns a user question into a chat message: exact answer, chart/table, or LLM answer."""
import json

import charts
import data_utils as du
import llm
from charts import SpecError, build_chart, describe_spec


def _msg(content, **kw):
    return {"role": "assistant", "content": content, "fig": None, "table": None, "spec": None,
            "notes": [], "retrieved": [], "badge": None, **kw}


def _from_result(res, content, badge):
    return _msg(content, fig=res.fig, table=res.table, spec=res.spec, notes=res.notes, badge=badge)


def _offline(question, df, reason):
    """Rule-based chart when the LLM is unavailable or failed."""
    spec = charts.heuristic_spec(question, df)
    if spec:
        try:
            res = build_chart(spec, df)
            return _from_result(res, f"Here is **{res.title}**. _(built with simple rules — {reason})_",
                                "🎨 Rule-based chart")
        except SpecError:
            pass
    return None


def answer_question(question, df, index, last_spec=None, history=(), explain=True):
    """Returns an assistant message dict (see _msg)."""
    # 1) Cheap exact answers (no LLM) for whole-dataset stats.
    if not du.needs_llm(question):
        exact = du.local_answer(question, df)
        if exact:
            return _msg(exact, badge="📊 Exact answer (pandas)")

    # 2) No API key: rule-based charts only.
    if not llm.llm_available():
        m = _offline(question, df, "no API key configured")
        return m or _msg(
            "I can build simple charts and answer basic statistics without an API key "
            "(try *“show a correlation heatmap”*, *“histogram of Revenue”* or *“average Revenue by Region”*). "
            "For open-ended questions add a Groq/OpenAI key in `.env`.", badge="ℹ️ No API key")

    # 3) LLM planning.
    documents, vectorizer, matrix = index
    retrieved = du.retrieve(question, documents, vectorizer, matrix)
    ctx = [d for d, _ in retrieved]
    schema = du.schema_text(df)
    try:
        plan = llm.plan_request(question, schema, ctx, history, last_spec)
    except Exception as exc:
        m = _offline(question, df, "the AI request failed")
        return m or _msg(f"The AI request failed.\n\n```\n{llm.explain_error(exc)}\n```", badge="⚠️ AI error")

    if plan["type"] == "answer":
        text = str(plan.get("answer") or "").strip() or "I could not produce an answer from the available information."
        return _msg(text, badge="🤖 AI answer", retrieved=retrieved)

    # 4) Chart / table: validate + execute the spec (one automatic repair attempt).
    res, err = None, None
    try:
        res = build_chart(plan.get("spec") or {}, df)
    except SpecError as exc:
        err = exc
        try:
            plan2 = llm.plan_request(question, schema, ctx, history, last_spec, repair=(plan["_raw"], str(exc)))
            if plan2["type"] != "answer":
                res = build_chart(plan2.get("spec") or {}, df)
                plan, err = plan2, None
        except Exception as exc2:
            err = exc2 if isinstance(exc2, SpecError) else err
    if res is None:
        m = _offline(question, df, "the AI spec was invalid")
        return m or _msg(f"I couldn't build that chart: **{err}**\n\nTry rephrasing, or use the **Chart Builder** tab.",
                         badge="⚠️ Could not build chart")

    insight = llm.explain_result(question, res.title, res.facts, res.notes) if explain else None
    text = insight or str(plan.get("answer") or "").strip() or f"Here is **{res.title}**."
    return _from_result(res, text, "🤖 AI-planned chart · computed with pandas" if res.fig is not None
                        else "🤖 AI-planned table · computed with pandas")
