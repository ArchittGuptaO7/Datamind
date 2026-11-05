# DataMind AI 📊🤖

**AI-Powered Data Analytics, Automated Visualizations & Conversational Data Analysis**

DataMind AI is an interactive data analytics platform that transforms raw CSV and Excel datasets into meaningful insights, visualizations, and actionable information. It combines Python-based data analysis with AI-powered conversational analytics, allowing users to explore their data using natural language.

Upload a dataset, explore automatic insights, build custom charts, and ask questions about your data — all through a simple Streamlit interface.

## ✨ Features

- **Dataset Upload:** Import CSV and Excel files or explore demo datasets.
- **Automated Data Profiling:** Get an overview of columns, data types, statistics, and dataset structure.
- **Auto Insights:** Discover patterns and useful information automatically.
- **Interactive Chart Builder:** Create dynamic visualizations using Plotly.
- **AI Data Analyst:** Ask questions about your dataset in plain English and receive data-driven answers with relevant charts.
- **Data Quality Analysis:** Identify missing values and potential data-quality issues.
- **Safe AI Processing:** Uses structured LLM responses and chart validation rather than executing AI-generated Python code.
- **Fallback Support:** Core functionality remains usable when the AI API is unavailable.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas & NumPy | Data processing and analysis |
| Plotly | Interactive visualizations |
| Scikit-learn | TF-IDF-based retrieval |
| Groq API / OpenAI client | AI-powered analytics |
| OpenPyXL | Excel file support |

## 📁 Project Structure

```text
DataMind-AI/
│
├── app.py              # Main Streamlit application
├── analyst.py          # AI analyst functionality
├── charts.py           # Chart generation and visualization
├── data_utils.py       # Data processing utilities
├── llm.py              # LLM integration
│
├── requirements.txt    # Project dependencies
├── .gitignore          # Files excluded from Git
├── .env.example        # Environment variable template
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DataMind-AI.git
cd DataMind-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Add your own API key. Never upload your `.env` file or expose API credentials publicly.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 How It Works

1. Upload a CSV or Excel dataset, or select demo data.
2. Explore the dataset through the Data and Auto Insights sections.
3. Generate interactive visualizations using the Chart Builder.
4. Ask questions in natural language through the AI Analyst.
5. Review data-quality information to identify potential issues.

## 🔮 Future Improvements

- Support for additional data sources and formats.
- Advanced predictive analytics.
- More sophisticated natural-language querying.
- Exportable analytical reports.
- Enhanced automated insight generation.

## 👨‍💻 Author

**Architt Gupta**  
B.Tech Computer Science & Engineering (AI & ML)

## 📄 License

This project is available under the MIT License.
