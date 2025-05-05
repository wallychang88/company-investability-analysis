# api/analyze.py
"""Back‑end endpoint for generating investability scores in a memory‑safe, streamable
   fashion.  Tweaks in this build:
   • Model bumped to **gpt‑4o‑mini**
   • max_tokens lowered to 1024 (fits the model’s 12k context window comfortably)
   • Global timeout of 30 s on every OpenAI call
   • Robust progress accounting even when a batch errors
   • Chunk‑reads the CSV so large uploads don’t blow RAM
   • CORS restricted to the production front‑end
"""
from __future__ import annotations

import json, os, time
from typing import Dict, Generator

import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from openai import OpenAI, APIError

###############################################################################
# Flask + CORS setup
###############################################################################

app = Flask(__name__)
CORS(app, origins=["https://company-investability-score.vercel.app"])  # 🚦 prod only

###############################################################################
# OpenAI client configuration
###############################################################################

MODEL_NAME = "gpt-4o-mini"  # ⬅️ switched from gpt‑3.5‑turbo
MAX_TOKENS = 1024
TEMPERATURE = 0.3
BATCH_SIZE = 5  # rows / OpenAI request
TIMEOUT = 30  # seconds

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=TIMEOUT)

###############################################################################
# Helper functions
###############################################################################

def build_system_prompt(criteria: str) -> str:
    """Returns the system prompt with user‑supplied criteria embedded."""
    return (
        "You are an expert venture analyst. Using the criteria below, rate each "
        "company’s *investability* from 1 to 10 and give a one‑sentence rationale. "
        "Return a JSON list where each element has keys: company_name, "
        "investability_score (integer 1‑10), rationale.\n\nCriteria:\n" + criteria
    )

def score_batch(
    df_slice: pd.DataFrame, column_map: Dict[str, str], criteria: str
) -> str:
    """Calls the chat model on a slice of the dataframe and returns raw JSON‑text."""
    system_prompt = build_system_prompt(criteria)

    messages = [{"role": "system", "content": system_prompt}]
    for _, row in df_slice.iterrows():
        company_name = row[column_map["company_name"]]
        description = row.get(column_map.get("description", ""), "")
        messages.append(
            {
                "role": "user",
                "content": f"Company: {company_name}\nDescription: {description}\nScore:",
            }
        )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content


def stream_analysis(
    csv_stream, column_map: Dict[str, str], criteria: str
) -> Generator[str, None, None]:
    """Yields NDJSON strings as each batch is processed."""
    processed = 0

    for chunk in pd.read_csv(csv_stream, dtype=str, chunksize=1000):
        chunk = chunk.fillna("")
        num_rows = len(chunk)

        for start in range(0, num_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_rows)
            batch = chunk.iloc[start:end]

            try:
                result = score_batch(batch, column_map, criteria)
                payload = {"progress": processed + len(batch), "result": json.loads(result)}
            except APIError as e:
                payload = {
                    "progress": processed + len(batch),
                    "error": f"OpenAI error: {e.__class__.__name__}: {e}",
                }
            except Exception as e:  # catch‑all so stream never dies
                payload = {"progress": processed + len(batch), "error": str(e)}

            processed += len(batch)
            yield json.dumps(payload) + "\n"

###############################################################################
# Flask route
###############################################################################


@app.route("/api/analyze", methods=["POST"])
def analyze_endpoint():
    """HTTP endpoint that streams NDJSON back to the front‑end."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        column_map = json.loads(request.form["columnMap"])
    except (KeyError, json.JSONDecodeError):
        return jsonify({"error": "Invalid or missing columnMap"}), 400

    criteria = request.form.get("criteria", "Return your best estimate.")

    ndjson_stream = stream_analysis(request.files["file"].stream, column_map, criteria)
    return Response(ndjson_stream, mimetype="application/x-ndjson")


if __name__ == "__main__":
    # Only for local debugging; use gunicorn/uvicorn in prod.
    app.run(host="0.0.0.0", port=8080, debug=True)
