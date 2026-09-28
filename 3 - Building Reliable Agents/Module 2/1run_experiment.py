import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client, evaluate

load_dotenv()
client = Client()

csv_path = Path(__file__).parent / "officeflow-dataset.csv"

# 1. Upload CSV to LangSmith Cloud (only needs to be run once)
DATASET_NAME = "officeflow-dataset"
if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.upload_csv(
        csv_file=str(csv_path),
        input_keys=["question"],  # name of the question/input column in your CSV
        output_keys=[],           # or ground-truth columns if you have them
        name=DATASET_NAME,
        description="OfficeFlow Customer Support Evaluation Dataset"
    )
    print(f"Uploaded dataset: {DATASET_NAME}")

# 2. Dummy app & Evaluator
def dummy_app(inputs: dict) -> dict:
    return {"response": "Sure! In OfficeFlow, you can reset your password from the settings page."}

def mentions_officeflow(outputs: dict) -> bool:
    return "officeflow" in outputs["response"].lower()

# 3. Now passing the string name works!
results = evaluate(
    dummy_app,
    data=DATASET_NAME,  # <--- Looks up your cloud dataset
    evaluators=[mentions_officeflow],
    experiment_prefix="dummy-app-eval"
)