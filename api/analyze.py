# api/analyze.py

from __future__ import annotations

import json, os, time
from typing import Dict, Generator, List
from io import BytesIO

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
BATCH_SIZE = 2  # rows / OpenAI request
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
        "You are an expert venture analyst. Your task is to evaluate companies based on the investment criteria provided below.\n\n"
        "For each company in the input, you MUST assign an investability score from 0 to 10 (integer only).\n\n"
        "IMPORTANT: You must return your analysis in VALID JSON format with this exact structure:\n"
        '{"rows":[{"company_name":"Company Name 1", "investability_score":8}, {"company_name":"Company Name 2", "investability_score":5}, ...]}\n\n'
        "Each company MUST have both a company_name and investability_score field.\n\n"
        "Criteria for evaluation:\n" + criteria
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

    # Create user message with explicit request for JSON format
    user_message = (
        f"Analyze the following companies based on the investment criteria. "
        f"For each company, provide an investability score from 0-10.\n\n"
        f"Return ONLY a JSON object with this structure: {{\"rows\":[{{\"company_name\":\"Name\", \"investability_score\":N}}, ...]}}\n\n"
        f"Companies to analyze:\n\n{batch_content}"
    )
    
    # Create the completion request
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
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
        print(f"Raw API response: {response_text[:100]}...")
        
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Get rows and validate
        rows = response_data.get("rows", [])
        
        # Verify we got correct data
        if not rows:
            print(f"WARNING: No rows found in response. Full response: {response_text}")
            # Fall back to generating scores ourselves
            print("Using fallback scoring due to empty rows")
            return [{"company_name": company["name"], "investability_score": 5} for company in companies]
            
        # Validate each row has required fields
        valid_rows = []
        for i, row in enumerate(rows):
            if "company_name" not in row or "investability_score" not in row:
                print(f"WARNING: Row {i} missing required fields: {row}")
                continue
                
            # Ensure score is an integer
            try:
                row["investability_score"] = int(row["investability_score"])
                valid_rows.append(row)
            except (ValueError, TypeError):
                print(f"WARNING: Invalid score format in row {i}: {row}")
        
        print(f"Received {len(valid_rows)} valid company scores from OpenAI API")
        
        # If we didn't get any valid rows, use fallback
        if not valid_rows and companies:
            print("No valid rows found, using fallback scoring")
            return [{"company_name": company["name"], "investability_score": 5} for company in companies]
            
        return valid_rows
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response was: {response_text}")
        
        # Attempt to create minimal valid scores as a fallback
        print("Using fallback scoring due to JSON parse error")
        return [{"company_name": company["name"], "investability_score": 5} for company in companies]
    except Exception as e:
        print(f"Error in score_batch: {type(e).__name__}: {str(e)}")
        
        # If we have companies, return fallback scores rather than raising the error
        if companies:
            print("Using fallback scoring due to exception")
            return [{"company_name": company["name"], "investability_score": 5} for company in companies]
        else:
            raise  # Re-raise only if we don't have company info

def stream_analysis(
    csv_stream, column_map: Dict[str, str], criteria: str
) -> Generator[str, None, None]:
    """Yields NDJSON strings as each batch is processed."""
    processed = 0
    total_rows = 0
    
    print("Starting stream analysis")
    
    try:
        # Instead of reading the stream twice, read it once into a DataFrame
        print("Reading CSV data...")
        all_data = pd.read_csv(csv_stream, dtype=str)
        total_rows = len(all_data)
        all_data = all_data.fillna("")
        print(f"Found {total_rows} total rows in CSV")
        
        # Process in chunks
        chunk_size = 1000
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = all_data.iloc[chunk_start:chunk_end]
            num_rows = len(chunk)
            chunk_idx = chunk_start // chunk_size + 1
            print(f"Processing chunk {chunk_idx} with {num_rows} rows")

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
    
    # Read the file into memory to prevent "I/O operation on closed file" error
    print("Reading uploaded file into memory...")
    uploaded_file = request.files["file"]
    file_content = uploaded_file.read()
    
    # Create a BytesIO object that can be safely read multiple times
    csv_data = BytesIO(file_content)
    
    print("Starting analysis stream...")
    ndjson_stream = stream_analysis(csv_data, column_map, criteria)
    
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
