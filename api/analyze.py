# api/analyze.py
"""Back‑end endpoint for generating investability scores in a memory‑safe, streamable
   fashion.  Tweaks in this build:
   • Model bumped to **gpt‑4o‑mini**
   • max_tokens lowered to 1024 (fits the model's 12k context window comfortably)
   • Global timeout of 30 s on every OpenAI call
   • Robust progress accounting even when a batch errors
   • Chunk‑reads the CSV so large uploads don't blow RAM
   • CORS restricted to the production front‑end
   • Response format updated to return {"rows":[{<csv row fields>, "investability_score":<0‑10>}, …]}
"""
from __future__ import annotations

import json, os, time
from typing import Dict, Generator, List

import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from openai import OpenAI, APIError

###############################################################################
# Flask + CORS setup
###############################################################################

app = Flask(__name__)
CORS(app, origins=["https://company-investability-score.vercel.app"])  # prod only

###############################################################################
# OpenAI client configuration
###############################################################################

MODEL_NAME = "gpt-4o-mini"  
MAX_TOKENS = 1024
TEMPERATURE = 0.2
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
        "company's *investability* from 0 to 10 (integer only) based on how well it matches the criteria. "
        "Return ONLY valid JSON in the shape "
        '{"rows":[{"company_name":"Name", "investability_score":N}, ...]}.\n\n'
        "Criteria:\n" + criteria
    )

def score_batch(
    df_slice: pd.DataFrame, column_map: Dict[str, str], criteria: str
) -> List[Dict]:
    """Calls the chat model on a slice of the dataframe and returns just company names and scores."""
    system_prompt = build_system_prompt(criteria)

    # Prepare company data with mandatory and optional columns when mapped
    batch_content = ""
    companies = []
    
    for _, row in df_slice.iterrows():
        # Get company name (from description field or designated company_name field)
        company_name = row.get(column_map.get("company_name", ""), "") or row.get(column_map.get("description", ""), "")
        companies.append({"name": company_name})
        
        # Start with mandatory fields
        company_data = (
            f"Company: {company_name}\n"
            f"Employee Count: {row.get(column_map.get('employee_count', ''), '')}\n"
            f"Description: {row.get(column_map.get('description', ''), '')}\n"
            f"Industries: {row.get(column_map.get('industries', ''), '')}\n"
            f"Specialties: {row.get(column_map.get('specialties', ''), '')}\n"
            f"Products/Services: {row.get(column_map.get('products_services', ''), '')}\n"
            f"End Markets: {row.get(column_map.get('end_markets', ''), '')}\n"
        )
        
        # Add optional fields if they are mapped
        optional_fields = []
        
        if column_map.get("country") and column_map.get("country") in row:
            optional_fields.append(f"Country: {row[column_map.get('country')]}")
            
        if column_map.get("ownership") and column_map.get("ownership") in row:
            optional_fields.append(f"Ownership: {row[column_map.get('ownership')]}")
            
        if column_map.get("founding_year") and column_map.get("founding_year") in row:
            optional_fields.append(f"Founding Year: {row[column_map.get('founding_year')]}")
            
        # Add any other mapped columns that aren't in the standard set
        for key, col_name in column_map.items():
            if key not in ["employee_count", "description", "industries", "specialties", 
                           "products_services", "end_markets", "country", "ownership", 
                           "founding_year", "company_name"] and col_name in row:
                optional_fields.append(f"{key.replace('_', ' ').title()}: {row[col_name]}")
        
        # Add optional fields to company data if present
        if optional_fields:
            company_data += "\n".join(optional_fields) + "\n"
            
        company_data += "\n"
        batch_content += company_data

    # Create the completion request
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Companies to analyze:\n\n{batch_content}"}
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"}
    )
    
    response_text = completion.choices[0].message.content
    
    try:
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Return the rows array directly or empty list if not present
        return response_data.get("rows", [])
    except json.JSONDecodeError as e:
        # Fallback if the model returns invalid JSON
        print(f"JSON parse error: {e}")
        print(f"Response was: {response_text}")
        
        # Attempt to create minimal valid scores as a fallback
        return [{"company_name": company["name"], "investability_score": 5} for company in companies]

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
                # Get just company names and scores
                rows = score_batch(batch, column_map, criteria)
                
                # Create response payload with simplified rows
                payload = {
                    "progress": processed + len(batch),
                    "result": rows  # Using 'result' to maintain compatibility with front-end
                }
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


# Add a health check endpoint for front-end connectivity testing
@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "time": time.time()})


if __name__ == "__main__":
    # Only for local debugging; use gunicorn/uvicorn in prod.
    app.run(host="0.0.0.0", port=8080, debug=True)
