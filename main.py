import os
import sys
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
os.environ["DEEPSEEK_MODEL"] = "deepseek-reasoner"

# os.environ["ENABLE_SELF_CONSISTENCY"] = "true"
# os.environ["SELF_CONSISTENCY_SAMPLES"] = "3"

# -----------------------------
# Project path (Windows-safe)
# -----------------------------
PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

os.chdir(PROJECT)
print("cwd:", os.getcwd())

# -----------------------------
# Run evaluation
# -----------------------------
from scripts.report import evaluate_and_report

res, figs = evaluate_and_report(
    os.path.join(PROJECT, "tests", "Ticketing_DB.jsonl"),
    frac=0.1,
    workers=6,
    accuracy_html=os.path.join(PROJECT, "accuracy_report.html"),
    cost_html=os.path.join(PROJECT, "cost_report.html"),
    accuracy_png=os.path.join(PROJECT, "accuracy_report.png"),
    errors_out=os.path.join(PROJECT, "errors.jsonl"),
    errors_xlsx=os.path.join(PROJECT, "errors.xlsx"),
    show=False,
)
print("Done!")