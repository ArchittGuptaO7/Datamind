# 🧠 DataMind AI : https://archittguptao7.github.io/Datamind/
### AI-powered data analysis, automatic visuals & a chat analyst that draws charts

Upload a CSV/Excel file (or load the built-in demo) and DataMind AI profiles it, draws the right visuals
automatically, and lets you **ask for charts and numbers in plain English**.

## Features

| Tab | What it does |
|---|---|
| 📋 **Data** | Preview, descriptive statistics, CSV download |
| ✨ **Auto Insights** | Rule-based key findings · **correlation heatmap** (Pearson/Spearman) · histograms / box plots · category breakdowns · time trends · pairwise scatter matrix · optional AI briefing |
| 🎨 **Chart Builder** | Build bar, line, area, scatter, histogram, box, violin, pie, correlation & pivot heatmaps with dropdowns + row filters (no API key needed) |
| 🤖 **AI Analyst** | Chat: *“heatmap of average revenue by region and category”*, *“make it a pie chart”*, *“top 5 regions by revenue in 2025”*. Returns real charts/tables + a short grounded insight |
| 🧹 **Data Quality** | Missing-value bar + matrix, duplicates (with de-duplicated download), outliers & skew |

Every chart can be downloaded as interactive HTML and every table as CSV.

## How the AI makes charts (safe by design)

```text
 "average revenue by region as a bar chart"
              │
              ▼
   LLM planner (Groq / OpenAI)  ── sees only the schema + sample rows, never writes code
              │  JSON spec: {chart_type:"bar", x:"Region", y:"Revenue", agg:"mean"}
              ▼
   Validator (charts.py)  ── unknown column? wrong type? → error sent back to the LLM once to self-repair
              ▼
   pandas computes the numbers (exact)  →  Plotly renders the figure
              ▼
   LLM explains the *computed* table in 2–4 sentences (numbers must come from the table)
```

* **No `exec`/`eval` of model output** – the LLM can only choose from a fixed set of chart types and
  columns that really exist, so it cannot run arbitrary code or invent data.
* **Exact answers** – group-by numbers are computed by pandas, not guessed by the model.
* **Graceful fallbacks** – no key or API failure → rule-based chart engine for common requests;
  model retired → the app switches to an available model automatically.
* **Follow-ups work** – the previous chart spec is passed back, so “make it a pie” / “only 2025” just modify it.
* **Retrieval** – TF-IDF row retrieval still powers look-up questions (“show passengers named Smith”).

## Quick start (Windows)

1. Unzip the folder, open **cmd** inside it.
2. Double-click `run_app.bat`, **or** run:

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux: `source venv/bin/activate` instead of the `activate` line.

## AI setup (optional)

Copy `.env.example` to `.env` and add your key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

OpenAI also works (`OPENAI_API_KEY`, `OPENAI_MODEL`); if both are set, Groq is used.
Use the sidebar **🔌 Test LLM connection** button to verify the key and model.
Without any key, everything except open-ended AI answers still works.

## Things to try (demo dataset)

* “Show a correlation heatmap”
* “Average revenue by category as a horizontal bar chart”
* “Monthly revenue trend split by region”
* “Box plot of revenue by segment”
* “Scatter plot of units vs revenue with a trendline”
* “Heatmap of average satisfaction by region and category”
* “Top 3 segments by total revenue in 2025” → exact table
* then: “make it a pie chart”, “only the West region”

## Project structure

```text
app.py         Streamlit UI (5 tabs, chat, caching)
analyst.py     Orchestrator: exact answer → rule-based chart → LLM plan → validate → explain
charts.py      Spec validation, aggregation (pandas), Plotly figures, rule-based chart heuristics
llm.py         Groq/OpenAI client, model fallback, planner prompt, explanation writer
data_utils.py  Loading, type detection, profiling, insights, demo data, TF-IDF retrieval
tests/         35 pytest tests (python -m pytest -q)
```

## Limits

* Retrieval covers the first 1,500 rows; statistics and charts always use all rows.
* Large files: distribution plots use a 20,000-row random sample (labelled), statistics use every row.
* TF-IDF is lexical retrieval; sentence embeddings + a vector DB are the natural next step.

## Suggested future improvements

* Sentence embeddings + FAISS/Chroma, PDF/document RAG
* Predictive models (AutoML tab) and automated PDF reports
* Snowflake / SQL as the data layer

## Security

Never commit API keys. `.env` is listed in `.gitignore` (make sure the file is named `.gitignore`).
If a key is ever exposed, revoke it in the provider console and create a new one.

## Academic positioning

**Project:** DataMind AI — AI-Powered Data Analysis, Visualisation and Conversational Analytics
**Core concepts:** LLM tool-planning (structured output), RAG, EDA, data visualisation, prompt engineering,
input validation / safe execution, responsible AI (no hallucinated numbers).
