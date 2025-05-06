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
        "Pay special attention to employee count, country, ownership, total raised, latest round, date of most recent investment, and industry fit.\n\n"
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
    
    system_prompt = build_system_prompt(criteria)

    # Prepare company data with mandatory and optional columns when mapped
    batch_content = ""
    company_names = []
    
    for _, row in enumerate(df_slice.iterrows()):
        # Get company name (from description field or designated company_name field)
        desc_col = column_map.get("description", "")
        name_col = column_map.get("company_name", "")
            
        # Safe row access
        company_name = ""
        if name_col and name_col in row[1]:
            company_name = row[1][name_col]
        if not company_name and desc_col and desc_col in row[1]:
            company_name = row[1][desc_col]
        
        company_names.append(company_name)  # Store exact name for later matching
        
        # Get description without truncation
        description = row[1][desc_col] if desc_col in row[1] else ''
        
        # Start with mandatory fields - more concise format
        company_data = [f"Company: {company_name}"]
        
        # Get column references
        emp_col = column_map.get('employee_count', '')
        ind_col = column_map.get('industries', '')
        spec_col = column_map.get('specialties', '')
        prod_col = column_map.get('products_services', '')
        end_col = column_map.get('end_markets', '')
        country_col = column_map.get('country', '')
        ownership_col = column_map.get('ownership', '')
        total_raised_col = column_map.get('total_raised', '')
        latest_raised_col = column_map.get('latest_raised', '')
        recent_investment_col = column_map.get('date_of_most_recent_investment', '')
        
        # Safe data access
        fields = {
            "Employee Count": row[1][emp_col] if emp_col in row[1] else '',
            "Description": description,
            "Industries": truncate_text(row[1][ind_col] if ind_col in row[1] else '', 200),
            "Specialties": truncate_text(row[1][spec_col] if spec_col in row[1] else '', 200),
            "Products/Services": truncate_text(row[1][prod_col] if prod_col in row[1] else '', 200),
            "End Markets": truncate_text(row[1][end_col] if end_col in row[1] else '', 200),
            "Country": row[1][country_col] if country_col in row[1] else '',
            "Ownership": row[1][ownership_col] if ownership_col in row[1] else '',
            "Total Raised": row[1][total_raised_col] if total_raised_col in row[1] else '',
            "Latest Raised": row[1][latest_raised_col] if latest_raised_col in row[1] else '',
            "Date of Most Recent Investment": row[1][recent_investment_col] if recent_investment_col in row[1] else ''
        }
        
        # Add each field if it has content
        for label, value in fields.items():
            if value and value.strip():
                company_data.append(f"{label}: {value}")
        
        # Add optional fields if they're mapped and have content
        field_1_col = column_map.get("field_1", "")
        if field_1_col and field_1_col in row[1] and row[1][field_1_col]:
            company_data.append(f"Field 1: {row[1][field_1_col]}")
            
        field_2_col = column_map.get("field_2", "")
        if field_2_col and field_2_col in row[1] and row[1][field_2_col]:
            company_data.append(f"Field 2: {row[1][field_2_col]}")
            
        field_3_col = column_map.get("field_3", "")
        if field_3_col and field_3_col in row[1] and row[1][field_3_col]:
            company_data.append(f"Field 3: {row[1][field_3_col]}")
        
        # Join all company data and add to batch
        company_text = "\n".join(company_data) + "\n\n"
        batch_content += company_text
        
    # Estimate token count
    tokens_per_char = 0.25  # Rough estimate: 4 chars per token on average
    estimated_tokens = int(len(batch_content) * tokens_per_char)
    
    if estimated_tokens > 6000:
        print(f"WARNING: Estimated token count ({estimated_tokens}) is high and may exceed model limits.")
    
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
        
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Get rows and validate
        rows = response_data.get("rows", [])
        
        # Verify we got correct data
        if not rows:
            print(f"WARNING: No rows found in response")
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
                row["company_name"] = company_names[i]
            
            # Ensure score is a valid integer between 0-10
            try:
                score = int(row.get("investability_score", -1))
                row["investability_score"] = max(0, min(10, score))
            except (ValueError, TypeError):
                row["investability_score"] = -1
                
            valid_rows.append(row)
        
        # If we have fewer rows than companies, add missing companies with -1 scores
        if len(valid_rows) < len(company_names):
            processed_names = [row["company_name"] for row in valid_rows]
            for name in company_names:
                if name not in processed_names:
                    valid_rows.append({"company_name": name, "investability_score": -1})
        
        print(f"Returning {len(valid_rows)} valid company scores")
        
        return valid_rows
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        
        # Return companies with -1 scores instead of fallback scoring
        return [{"company_name": name, "investability_score": -1} for name in company_names]
    except Exception as e:
        print(f"Error in score_batch: {type(e).__name__}: {str(e)}")
        
        # Return companies with -1 scores instead of fallback scoring
        return [{"company_name": name, "investability_score": -1} for name in company_names]

def stream_analysis(
    csv_stream, column_map: Dict[str, str], criteria: str
) -> Generator[str, None, None]:
    """Yields NDJSON strings as each batch is processed."""
    processed = 0
    total_rows = 0
    
    print("==== DETAILED DEBUG STREAM ANALYSIS ====")
    print("Starting stream analysis with extensive debugging")
    
    try:
        # Create a temporary clean copy of the file to avoid metadata/encoding issues
        print("Creating clean copy of the uploaded file...")
        
        # Read the file content and get basic info
        csv_content = csv_stream.read()
        print(f"File size: {len(csv_content)} bytes")
        print(f"First 100 bytes (hex): {csv_content[:100].hex()}")
        
        # Create a new BytesIO object with the content
        clean_csv = BytesIO()
        
        # Check for BOM and other encoding issues
        if csv_content.startswith(b'\xef\xbb\xbf'):
            # Remove BOM if present
            csv_content = csv_content[3:]
            print("Removed BOM from file")
            print(f"New first 100 bytes (hex): {csv_content[:100].hex()}")
        
        # Check for other common BOMs
        if csv_content.startswith(b'\xfe\xff') or csv_content.startswith(b'\xff\xfe'):
            print("WARNING: Detected UTF-16 BOM")
            if csv_content.startswith(b'\xfe\xff'):
                csv_content = csv_content[2:]
                print("Removed UTF-16 BE BOM")
            else:
                csv_content = csv_content[2:]
                print("Removed UTF-16 LE BOM")
        
        if csv_content.startswith(b'\x00\x00\xfe\xff') or csv_content.startswith(b'\xff\xfe\x00\x00'):
            print("WARNING: Detected UTF-32 BOM")
            if csv_content.startswith(b'\x00\x00\xfe\xff'):
                csv_content = csv_content[4:]
                print("Removed UTF-32 BE BOM")
            else:
                csv_content = csv_content[4:]
                print("Removed UTF-32 LE BOM")
        
        # Check for unusual characters in the first few lines
        try:
            first_few_lines = csv_content[:1000].decode('utf-8').split('\n')[:10]
            print("First few lines of the file:")
            for i, line in enumerate(first_few_lines):
                print(f"Line {i}: {repr(line)}")
        except Exception as e:
            print(f"Error decoding first few lines: {e}")
        
        # Write cleaned content to the new buffer
        clean_csv.write(csv_content)
        clean_csv.seek(0)  # Reset pointer to beginning of file
        
        # Try multiple encoding options if needed
        all_data = None
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                print(f"Trying to read CSV with encoding: {encoding}")
                clean_csv.seek(0)
                all_data = pd.read_csv(clean_csv, dtype=str, header=4, encoding=encoding)
                print(f"Successfully read CSV with encoding: {encoding}")
                break
            except Exception as e:
                print(f"Failed with encoding {encoding}: {e}")
        
        if all_data is None:
            print("All encoding attempts failed. Trying with chardet detection...")
            # Try to detect encoding
            import chardet
            clean_csv.seek(0)
            result = chardet.detect(csv_content)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            print(f"Detected encoding: {detected_encoding} (confidence: {confidence})")
            
            clean_csv.seek(0)
            all_data = pd.read_csv(clean_csv, dtype=str, header=4, encoding=detected_encoding)
            print(f"Read CSV with detected encoding: {detected_encoding}")
        
        # Print basic DataFrame info
        print(f"DataFrame info: {all_data.shape[0]} rows, {all_data.shape[1]} columns")
        print("Column names:", list(all_data.columns))
        
        # Fix any potential line ending issues - using updated methods instead of applymap
        print("Fixing line endings...")
        all_data = all_data.map(lambda x: x.replace('\r', '') if isinstance(x, str) else x)
        
        # Remove any non-printable characters - using updated methods instead of applymap
        print("Removing non-printable characters...")
        def remove_nonprintable(x):
            if isinstance(x, str):
                return ''.join(c for c in x if c.isprintable())
            return x
            
        all_data = all_data.map(remove_nonprintable)
        
        total_rows = len(all_data)
        all_data = all_data.fillna("")
        print(f"Successfully processed file. Found {total_rows} total rows in CSV")
        
        # Debug the first few rows of data
        print("First 2 rows of data:")
        for i in range(min(2, len(all_data))):
            row_data = all_data.iloc[i]
            print(f"Row {i}:")
            for col in all_data.columns:
                value = row_data[col]
                if isinstance(value, str) and len(value) > 50:
                    print(f"  {col}: {value[:50]}... (truncated)")
                else:
                    print(f"  {col}: {value}")
        
        # Debug column mapping
        print("Column mapping debug:")
        for key, value in column_map.items():
            print(f"  {key} -> {value}")
            if value and value in all_data.columns:
                print(f"    Column exists in DataFrame")
                sample = all_data[value].iloc[0] if not all_data.empty else "N/A"
                if isinstance(sample, str) and len(sample) > 50:
                    print(f"    Sample: {sample[:50]}... (truncated)")
                else:
                    print(f"    Sample: {sample}")
            else:
                print(f"    WARNING: Column not found in DataFrame")
        
        # Process in chunks
        chunk_size = 1000
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = all_data.iloc[chunk_start:chunk_end]
            num_rows = len(chunk)
            
            for start in range(0, num_rows, BATCH_SIZE):
                end = min(start + BATCH_SIZE, num_rows)
                batch = chunk.iloc[start:end]
                batch_size = len(batch)
                
                print(f"Processing batch: rows {start+1}-{end} (batch size: {batch_size})")

                try:
                    # Debug batch data before sending to OpenAI
                    print(f"Sending batch to OpenAI with {batch_size} rows:")
                    for i in range(min(1, batch_size)):
                        row_data = batch.iloc[i]
                        print(f"Sample row fields:")
                        for field_key in ['description', 'employee_count', 'industries']:
                            field_col = column_map.get(field_key, '')
                            if field_col and field_col in row_data:
                                value = row_data[field_col]
                                if isinstance(value, str) and len(value) > 50:
                                    print(f"  {field_key}: {value[:50]}... (truncated)")
                                else:
                                    print(f"  {field_key}: {value}")
                            else:
                                print(f"  {field_key}: NOT FOUND")
                    
                    # Get company scores from OpenAI
                    print("Calling score_batch function...")
                    rows = score_batch(batch, column_map, criteria)
                    
                    # Create response payload with rows
                    payload = {
                        "progress": processed + batch_size,
                        "result": rows
                    }
                    
                    # Debug API response
                    print(f"OpenAI API returned {len(rows)} results")
                    if len(rows) > 0:
                        print(f"First result: {rows[0]}")
                    else:
                        print("WARNING: No rows returned from OpenAI API")
                    
                    processed_pct = round((processed + batch_size) / total_rows * 100)
                    print(f"Progress: {processed + batch_size}/{total_rows} ({processed_pct}%)")
                    
                except APIError as e:
                    print(f"OpenAI API error: {type(e).__name__}: {str(e)}")
                    payload = {
                        "progress": processed + batch_size,
                        "error": f"OpenAI error: {e.__class__.__name__}: {e}",
                    }
                except Exception as e:
                    print(f"Unexpected error in batch processing: {type(e).__name__}: {str(e)}")
                    print(f"Error details: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    payload = {
                        "progress": processed + batch_size,
                        "error": f"Error: {type(e).__name__}: {str(e)}",
                    }

                processed += batch_size
                json_payload = json.dumps(payload)
                
                yield json_payload + "\n"
    except Exception as e:
        print(f"Stream analysis error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
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
