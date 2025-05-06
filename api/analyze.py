# api/analyze.py

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
CORS(app, origins=["https://company-investability-score.vercel.app", "http://localhost:3000"])  # Allow local testing

###############################################################################
# OpenAI client configuration
###############################################################################

MODEL_NAME = "gpt-4o-mini"  
MAX_TOKENS = 1024
TEMPERATURE = 0.2
BATCH_SIZE = 5  # rows / OpenAI request
TIMEOUT = 30  # seconds

# Initialize OpenAI client with error handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set!")

client = OpenAI(api_key=api_key, timeout=TIMEOUT)

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
    print(f"Scoring batch of {len(df_slice)} companies")
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

    print(f"Prepared batch content ({len(batch_content)} chars) for OpenAI API")

    # Create the completion request
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Companies to analyze:\n\n{batch_content}"}
    ]

    try:
        print(f"Calling OpenAI API with model {MODEL_NAME}...")
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        
        elapsed = time.time() - start_time
        print(f"OpenAI API call completed in {elapsed:.2f} seconds")
        
        response_text = completion.choices[0].message.content
        
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Get rows and validate
        rows = response_data.get("rows", [])
        print(f"Received {len(rows)} company scores from OpenAI API")
        
        return rows
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response was: {response_text}")
        
        # Attempt to create minimal valid scores as a fallback
        print("Using fallback scoring due to JSON parse error")
        return [{"company_name": company["name"], "investability_score": 5} for company in companies]
    except Exception as e:
        print(f"Error in score_batch: {type(e).__name__}: {str(e)}")
        raise  # Re-raise to be caught by stream_analysis

def stream_analysis(
    csv_stream, column_map: Dict[str, str], criteria: str
) -> Generator[str, None, None]:
    """Yields NDJSON strings as each batch is processed."""
    processed = 0
    total_rows = 0
    
    print("Starting stream analysis")
    
    try:
        # Count rows first
        print("Counting total rows in CSV...")
        df_chunks = pd.read_csv(csv_stream, dtype=str, chunksize=1000)
        chunks = list(df_chunks)
        total_rows = sum(len(chunk) for chunk in chunks)
        print(f"Found {total_rows} total rows in CSV")
        
        # Reset file pointer
        csv_stream.seek(0)
        
        # Process in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_stream, dtype=str, chunksize=1000)):
            chunk = chunk.fillna("")
            num_rows = len(chunk)
            print(f"Processing chunk {chunk_idx+1} with {num_rows} rows")

            for start in range(0, num_rows, BATCH_SIZE):
                end = min(start + BATCH_SIZE, num_rows)
                batch = chunk.iloc[start:end]
                batch_size = len(batch)
                
                print(f"Processing batch: rows {start+1}-{end} (batch size: {batch_size})")

                try:
                    # Get company scores from OpenAI
                    rows = score_batch(batch, column_map, criteria)
                    
                    # Create response payload with rows
                    payload = {
                        "progress": processed + batch_size,
                        "result": rows
                    }
                    
                    processed_pct = round((processed + batch_size) / total_rows * 100)
                    print(f"Progress: {processed + batch_size}/{total_rows} ({processed_pct}%)")
                    
                except APIError as e:
                    print(f"OpenAI API error: {type(e).__name__}: {str(e)}")
                    payload = {
                        "progress": processed + batch_size,
                        "error": f"OpenAI error: {e.__class__.__name__}: {e}",
                    }
                except Exception as e:
                    print(f"Unexpected error: {type(e).__name__}: {str(e)}")
                    payload = {
                        "progress": processed + batch_size,
                        "error": f"Error: {type(e).__name__}: {str(e)}",
                    }

                processed += batch_size
                json_payload = json.dumps(payload)
                print(f"Yielding payload with {len(rows) if 'rows' in locals() else 0} results")
                
                yield json_payload + "\n"
    except Exception as e:
        print(f"Stream analysis error: {type(e).__name__}: {str(e)}")
        yield json.dumps({"error": f"Stream error: {str(e)}"}) + "\n"

###############################################################################
# Flask route
###############################################################################

@app.route("/api/analyze", methods=["POST"])
def analyze_endpoint():
    """HTTP endpoint that streams NDJSON back to the front‑end."""
    print("\n==== ANALYZE ENDPOINT CALLED ====")
    print(f"Request from: {request.remote_addr}")
    
    # Debug OpenAI setup
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API KEY PRESENT: {'Yes' if api_key else 'No'}")
    if not api_key:
        error_msg = "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
        print(f"Error: {error_msg}")
        return jsonify({"error": error_msg}), 500
    
    # Test OpenAI connectivity with the actual model we'll use
    try:
        print(f"Testing OpenAI connection with model {MODEL_NAME}...")
        test_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'Connection established' in 5 words or less."}],
            max_tokens=10
        )
        print(f"OpenAI API test successful! Response: {test_response.choices[0].message.content}")
    except Exception as e:
        error_msg = f"OpenAI API test failed: {type(e).__name__}: {str(e)}"
        print(f"Error: {error_msg}")
        return jsonify({"error": error_msg}), 500
    
    # Process uploaded file
    if "file" not in request.files:
        print("Error: No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    try:
        print("Parsing column mappings...")
        column_map = json.loads(request.form["columnMap"])
        print(f"Column mappings: {column_map}")
    except (KeyError, json.JSONDecodeError) as e:
        error_msg = f"Invalid column mappings: {str(e)}"
        print(f"Error: {error_msg}")
        return jsonify({"error": error_msg}), 400

    criteria = request.form.get("criteria", "Return your best estimate.")
    print(f"Investment criteria: {criteria[:100]}...")
    
    # Get the weights if provided
    weights_str = request.form.get("weights")
    if weights_str:
        try:
            weights = json.loads(weights_str)
            print(f"Criteria weights provided: {weights}")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid weights JSON: {str(e)}")
    
    print("Starting analysis stream...")
    ndjson_stream = stream_analysis(request.files["file"].stream, column_map, criteria)
    
    print("Returning streaming response")
    return Response(ndjson_stream, mimetype="application/x-ndjson")


# Add a health check endpoint for front-end connectivity testing
@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    print("Health check requested")
    return jsonify({
        "status": "ok", 
        "time": time.time(),
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY"))
    })


if __name__ == "__main__":
    # Only for local debugging; use gunicorn/uvicorn in prod.
    print(f"Starting Flask development server on port 8080")
    print(f"OpenAI API KEY present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    print(f"Using OpenAI model: {MODEL_NAME}")
    app.run(host="0.0.0.0", port=8080, debug=True)
