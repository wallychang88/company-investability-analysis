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
        "DO NOT change the company names in any way - use them exactly as provided.\n\n"
        "Criteria for evaluation:\n" + criteria
    )

def truncate_text(text, max_length):
    """Helper function to truncate text to a maximum length."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def score_batch(
    df_slice: pd.DataFrame, column_map: Dict[str, str], criteria: str
) -> List[Dict]:
    """Calls the chat model on a slice of the dataframe with token efficiency."""
    print(f"Scoring batch of {len(df_slice)} companies")
    print(f"Column map received: {column_map}")
    
    # Debug first row of data
    if len(df_slice) > 0:
        print(f"First row keys: {list(df_slice.iloc[0].keys())}")
        print(f"First row values: {list(df_slice.iloc[0].values())}")
    
    system_prompt = build_system_prompt(criteria)

    # Prepare company data with mandatory and optional columns when mapped
    batch_content = ""
    companies = []
    
    # Store original company names to ensure exact matching later
    company_names = []
    
    # Debug column mapping and dataframe columns
    for col_key, col_name in column_map.items():
        print(f"Mapping '{col_key}' to '{col_name}'")
        if col_name and col_name in df_slice.columns:
            print(f"  Column '{col_name}' exists in dataframe")
            # Sample first few values
            sample = df_slice[col_name].head(2).tolist()
            print(f"  Sample values: {sample}")
        else:
            print(f"  WARNING: Column '{col_name}' NOT FOUND in dataframe")
    
    for idx, (_, row) in enumerate(df_slice.iterrows()):
        # Debug row processing
        print(f"Processing row {idx+1}")
        
        # Get company name (from description field or designated company_name field)
        desc_col = column_map.get("description", "")
        name_col = column_map.get("company_name", "")
        
        # Debug column values
        if desc_col:
            print(f"Description column: '{desc_col}', Value: '{row.get(desc_col, 'NOT FOUND')}'")
        if name_col:
            print(f"Company name column: '{name_col}', Value: '{row.get(name_col, 'NOT FOUND')}'")
            
        company_name = row.get(name_col, "") or row.get(desc_col, "")
        print(f"Using company name: '{company_name}'")
        
        companies.append({"name": company_name})
        company_names.append(company_name)  # Store exact name for later matching
        
        # Get description without truncation
        description = row.get(desc_col, '')
        
        # Start with mandatory fields - more concise format
        company_data = [f"Company: {company_name}"]
        
        # Add mandatory fields (without truncating description)
        emp_col = column_map.get('employee_count', '')
        ind_col = column_map.get('industries', '')
        spec_col = column_map.get('specialties', '')
        prod_col = column_map.get('products_services', '')
        end_col = column_map.get('end_markets', '')
        
        fields = {
            "Employee Count": row.get(emp_col, ''),
            "Description": description,
            "Industries": truncate_text(row.get(ind_col, ''), 200),
            "Specialties": truncate_text(row.get(spec_col, ''), 200),
            "Products/Services": truncate_text(row.get(prod_col, ''), 200),
            "End Markets": truncate_text(row.get(end_col, ''), 200)
        }
        
        # Debug field values
        print(f"Field values for company '{company_name}':")
        for field_name, field_value in fields.items():
            print(f"  {field_name}: '{field_value[:20]}{'...' if len(field_value) > 20 else ''}'")
        
        # Add each field if it has content
        for label, value in fields.items():
            if value.strip():
                company_data.append(f"{label}: {value}")
        
        # Add optional fields if they're mapped and have content
        country_col = column_map.get("country", "")
        if country_col and country_col in row and row[country_col]:
            company_data.append(f"Country: {row[country_col]}")
            
        ownership_col = column_map.get("ownership", "")
        if ownership_col and ownership_col in row and row[ownership_col]:
            company_data.append(f"Ownership: {row[ownership_col]}")
            
        founding_col = column_map.get("founding_year", "")
        if founding_col and founding_col in row and row[founding_col]:
            company_data.append(f"Founding Year: {row[founding_col]}")
            
        raised_col = column_map.get("total_raised", "")
        if raised_col and raised_col in row and row[raised_col]:
            company_data.append(f"Total Raised: {row[raised_col]}")
        
        # Join all company data and add to batch
        company_text = "\n".join(company_data) + "\n\n"
        batch_content += company_text
        
        print(f"Added {len(company_data)} data fields for this company")
        
    print(f"Final batch_content length: {len(batch_content)}")
    print(f"batch_content first 200 chars: {batch_content[:200]}")
    print(f"Total companies processed: {len(company_names)}")

    # Estimate token count
    tokens_per_char = 0.25  # Rough estimate: 4 chars per token on average
    estimated_tokens = int(len(batch_content) * tokens_per_char)
    
    print(f"Prepared batch content ({len(batch_content)} chars, ~{estimated_tokens} estimated tokens) for OpenAI API")
    
    # Provide a warning if token count seems high
    if estimated_tokens > 6000:
        print(f"WARNING: Estimated token count ({estimated_tokens}) is high and may exceed model limits.")
    
    print(f"First 500 chars of company data: {batch_content[:500]}...")

    # Create user message with explicit request for JSON format
    user_message = (
        f"Analyze these companies based on our investment criteria. "
        f"For each company, assign an investability score from 0-10.\n\n"
        f"Return ONLY a JSON object with this structure: {{\"rows\":[{{\"company_name\":\"Name\", \"investability_score\":N}}, ...]}}\n\n"
        f"IMPORTANT: Use the EXACT company names as provided. Do not modify or summarize the company names.\n\n"
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
        print(f"Raw API response: {response_text[:200]}...")
        
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Get rows and validate
        rows = response_data.get("rows", [])
        
        # Verify we got correct data
        if not rows:
            print(f"WARNING: No rows found in response. Full response: {response_text}")
            # Return companies with -1 scores instead of fallback scoring
            return [{"company_name": name, "investability_score": -1} for name in company_names]
            
        # Validate and fix each row
        valid_rows = []
        for i, row in enumerate(rows):
            # If we have more companies than rows, stop processing
            if i >= len(company_names):
                break
                
            # Ensure company name matches exactly
            if "company_name" not in row or not row["company_name"]:
                print(f"Missing company name in row {i}, using original: {company_names[i]}")
                row["company_name"] = company_names[i]
            
            # Ensure score is a valid integer between 0-10
            try:
                score = int(row.get("investability_score", -1))
                row["investability_score"] = max(0, min(10, score))
            except (ValueError, TypeError):
                print(f"Invalid score in row {i}, using -1")
                row["investability_score"] = -1
                
            valid_rows.append(row)
        
        # If we have fewer rows than companies, add missing companies with -1 scores
        if len(valid_rows) < len(company_names):
            processed_names = [row["company_name"] for row in valid_rows]
            for i, name in enumerate(company_names):
                if name not in processed_names:
                    print(f"Adding missing company: {name}")
                    valid_rows.append({"company_name": name, "investability_score": -1})
        
        print(f"Returning {len(valid_rows)} valid company scores")
        
        return valid_rows
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response was: {response_text}")
        
        # Return companies with -1 scores instead of fallback scoring
        print("Using -1 scores due to JSON parse error")
        return [{"company_name": name, "investability_score": -1} for name in company_names]
    except Exception as e:
        print(f"Error in score_batch: {type(e).__name__}: {str(e)}")
        
        # Return companies with -1 scores instead of fallback scoring
        print("Using -1 scores due to exception")
        return [{"company_name": name, "investability_score": -1} for name in company_names]

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
