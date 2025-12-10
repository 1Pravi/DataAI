from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os
import google.generativeai as genai
import textwrap
import uuid
from dotenv import load_dotenv
import json

load_dotenv()

# ============== CONFIG ==============

# Set your Gemini API key here or via environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDt1skZujCULgDk-aUIqmkq5pKPb-S1e_4")
genai.configure(api_key=GEMINI_API_KEY)

# Folder to save plot images
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Global store for dataset (simple single-user demo)
df_store = {
    "df": None,
    "filename": None,
    "columns": None
}

# ============== FASTAPI APP SETUP ==============

app = FastAPI()

# CORS so React (localhost:3000) can talk to FastAPI (localhost:8000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://datachatpro.vercel.app",
    "https://fffwqwq-jjnuzm8v8-hhhhs-projects-d30a9823.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve plots as static files
app.mount("/plots", StaticFiles(directory=PLOT_DIR), name="plots")


# ============== MODELS ==============

class AskRequest(BaseModel):
    question: str


# ============== HELPERS ==============

def build_pandas_summary(df: pd.DataFrame, filename: str):
    summary = {
        "filename": filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(5).to_dict(orient="records"),
    }
    # describe can fail if no numeric columns; handle gracefully
    try:
        summary["describe_numeric"] = df.describe().to_dict()
    except Exception:
        summary["describe_numeric"] = {}
    return summary


def summarize_with_gemini(df: pd.DataFrame, filename: str):
    """Ask Gemini to explain the dataset in natural language."""
    cols = list(df.columns)
    sample_rows = df.head(5).to_dict(orient="records")

    prompt = f"""
You are a helpful data analyst.

You are given a dataset named: {filename}

Columns:
{cols}

Here are the first 5 rows (as JSON records):
{sample_rows}

INSTRUCTIONS:
1. Analyze the dataset to understand its contents.
2. Provide a concise natural language summary of what the dataset is about.
3. Generate 4 specific, interesting questions the user could ask about this data.

IMPORTANT: Output your response as a valid JSON object with two keys: "summary" and "questions".
Do not wrap the JSON in markdown code blocks.

Example Output:
{{
  "summary": "This dataset contains...",
  "questions": [
    "What is the total revenue?",
    "Which product sells the most?",
    "Are there any trends over time?",
    "Compare sales by region"
  ]
}}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(textwrap.dedent(prompt))
    
    # Parse response
    full_text = response.text
    print(f"DEBUG: Full Gemini response:\n{full_text}") # Debug print
    
    summary = ""
    questions = []
    
    try:
        # Clean up potential markdown code blocks
        clean_text = full_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        summary = data.get("summary", "")
        questions = data.get("questions", [])
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON parse error: {e}")
        # Fallback to text parsing if JSON fails
        summary = full_text
        questions = []
    
    return summary, questions
            
def ask_gemini_for_answer_and_code(df: pd.DataFrame, question: str, filename: str):
    """
    Ask Gemini:
    - to give a natural language answer
    - to generate Python code using df, pandas, numpy, seaborn, matplotlib
    We separate answer and code using special markers.
    """
    cols = list(df.columns)

    prompt = f"""
You are an expert Python data analyst.

You are working with a pandas DataFrame named df loaded from a dataset: {filename}

The DataFrame has the following columns:
{cols}

User question about the dataset:
\"\"\"{question}\"\"\"

INSTRUCTIONS:
1. Answer the user's question DIRECTLY and CONCISELY.
2. If the user asks "is there any..." or "are there any...", check the data and answer "Yes" or "No" immediately, followed by a LIST of the items if the answer is Yes.
3. DO NOT explain *how* you found the answer (e.g. "I checked the column...", "The code filtered..."). Just give the answer.
4. If the answer is a list, format it as a bulleted list.
5. **CRITICAL**: When writing code, DO NOT try to read the dataset file (e.g. do NOT use `pd.read_csv`). The DataFrame `df` is ALREADY LOADED and available in the environment. Use `df` directly.
6. **CRITICAL**: DO NOT generate code using matplotlib or seaborn. ONLY generate `chart_data` and `chart_config` for the frontend to render.
7. **CRITICAL**: The `chart_config` MUST have a `type` (bar, line, scatter, pie), `xKey`, `yKey`, and `title`.

EXAMPLES:
User: "Are there any repeated clients?"
Assistant:
ANSWER_START
Yes, the following clients appear multiple times:
- Client A (3 times)
- Client B (2 times)
ANSWER_END

User: "What is the total revenue?"
Assistant:
ANSWER_START
The total revenue is $1,234,567.
ANSWER_END
CODE_START
CODE_END
FOLLOWUP_QUESTIONS_START
<question 1>
<question 2>
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(textwrap.dedent(prompt))
    full_text = response.text

    answer_text = ""
    code_text = ""

    # Simple parsing using markers
    if "ANSWER_START" in full_text and "ANSWER_END" in full_text:
        answer_text = full_text.split("ANSWER_START", 1)[1].split("ANSWER_END", 1)[0].strip()
    else:
        answer_text = full_text  # fallback

    if "CODE_START" in full_text and "CODE_END" in full_text:
        code_text = full_text.split("CODE_START", 1)[1].split("CODE_END", 1)[0].strip()

    followup_questions = []
    if "FOLLOWUP_QUESTIONS_START" in full_text:
        questions_text = full_text.split("FOLLOWUP_QUESTIONS_START", 1)[1].strip()
        followup_questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

    return answer_text, code_text, followup_questions


def execute_generated_code(df: pd.DataFrame, code: str):
    """
    Executes Gemini-generated code with df, pd, np, sns, plt available.
    Saves the last plot (if any) to a PNG and returns its filename.
    """
    # Environment in which the code will run
    exec_env = {
        "df": df,
        "pd": pd,
        "np": np,
    }

    try:
        exec(code, exec_env)
    except Exception as e:
        return None, None, str(e)

    # Extract chart data and config
    chart_data = exec_env.get("chart_data")
    chart_config = exec_env.get("chart_config")

    return chart_data, chart_config, None


# ============== ROUTES ==============

@app.get("/")
def root():
    return {"message": "Hi, this is the AI Data Assistant backend."}


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    1. Upload CSV/Excel
    2. Read into pandas
    3. Save df globally
    4. Return pandas summary + Gemini natural language summary
    """
    print(f"DEBUG: Received upload request for {file.filename}")
    content = await file.read()
    file_bytes = BytesIO(content)
    filename = file.filename.lower()

    # Read file with pandas
    if filename.endswith(".csv"):
        df = pd.read_csv(file_bytes)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file_bytes)
    else:
        return {"error": "Only CSV or Excel files are supported"}
    
    print(f"DEBUG: DataFrame created. Shape: {df.shape}")

    # Save globally (simple demo – not multi-user safe)
    df_store["df"] = df
    df_store["filename"] = file.filename
    df_store["columns"] = list(df.columns)

    pandas_summary = build_pandas_summary(df, file.filename)
    print("DEBUG: Pandas summary built")

    # Gemini summary (if API key is set)
    gemini_summary = ""
    sample_questions = []
    if GEMINI_API_KEY:
        try:
            print("DEBUG: Calling Gemini for summary...")
            gemini_summary, sample_questions = summarize_with_gemini(df, file.filename)
            print("DEBUG: Gemini summary received")
        except Exception as e:
            print(f"DEBUG: Gemini summary failed: {e}")
            gemini_summary = f"(Gemini summary failed: {e})"
    else:
        gemini_summary = "(Gemini API key not configured – set GEMINI_API_KEY to enable AI summary.)"

    return {
        "pandas_summary": pandas_summary,
        "gemini_summary": gemini_summary,
        "sample_questions": sample_questions
    }


@app.post("/ask")
async def ask_question(payload: AskRequest):
    """
    Ask a question about the currently uploaded dataset.
    Returns:
      - answer_text: natural language explanation
      - code: generated Python code
      - plot_image_url: URL to generated plot (if any)
    """
    df = df_store.get("df")
    filename = df_store.get("filename")

    if df is None:
        return {
            "error": "No dataset uploaded yet. Upload a dataset first using /upload-dataset."
        }

    if not GEMINI_API_KEY:
        return {
            "error": "Gemini API key not configured. Set GEMINI_API_KEY env variable."
        }

    question = payload.question

    # 1. Ask Gemini for answer + code
    try:
        answer_text, code_text, followup_questions = ask_gemini_for_answer_and_code(df, question, filename)
    except Exception as e:
        return {
            "answer_text": f"I encountered an error while processing your request: {str(e)}",
            "code": "",
            "chart_data": None,
            "chart_config": None,
            "followup_questions": []
        }

    # 2. Execute the code and capture data
    chart_data, chart_config, exec_error = (None, None, None)

    if code_text:
        chart_data, chart_config, exec_error = execute_generated_code(df, code_text)

    response = {
        "answer_text": answer_text,
        "code": code_text,
        "chart_data": chart_data,
        "chart_config": chart_config,
        "followup_questions": followup_questions
    }

    if exec_error:
        response["execution_error"] = exec_error

    return response
