# api/analyze.py

from __future__ import annotations

import json, os, time
import io
from typing import Dict, Generator, List
from io import BytesIO
import csv
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
        "VERY IMPORTANT: DO NOT CHANGE THE COMPANY NAMES IN ANY WAY - USE THEM EXACTLY AS PROVIDED. DO NOT EXPAND ABBREIVIATIONS, DO NOT CHANGE CAPITALIZATION, DO NOT CHANGE SPACING OR PUNCTUATION. RETURN THE COMPANY NAMES EXACTLY AS PROVIDED. \n\n"
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
    
    # Add helper function to match columns more flexibly
    def find_best_column_match(df, col_name):
        """Find the best match for a column name in the DataFrame"""
        if not col_name:
            return None
            
        # 1. Direct match
        if col_name in df.columns:
            return col_name
            
        # 2. Case-insensitive match
        for col in df.columns:
            if col.lower() == col_name.lower():
                return col
                
        # 3. Partial match - column contains the name or vice versa
        for col in df.columns:
            if col_name.lower() in col.lower() or col.lower() in col_name.lower():
                return col
                
        return None
    
    # Print column mapping with improved matching
    print("Improved column mapping:")
    improved_map = {}
    for key, value in column_map.items():
        matched_col = find_best_column_match(df_slice, value)
        if matched_col:
            print(f"  {key} -> {value} -> MATCHED as {matched_col}")
            improved_map[key] = matched_col
        else:
            print(f"  {key} -> {value} -> NOT FOUND")
            improved_map[key] = value  # Keep original if no match
            
    # Use the improved map for the rest of the function
    column_map = improved_map
    
    system_prompt = build_system_prompt(criteria)

    # Prepare company data with mandatory and optional columns when mapped
    batch_content = ""
    company_names = []
    
    for idx, (_, row) in enumerate(df_slice.iterrows()):
        print(f"Processing row {idx+1}")
        
        # Get company name (from description field or designated company_name field)
        desc_col = column_map.get("description", "")
        name_col = column_map.get("company_name", "")
            
        # Safe row access
        company_name = ""
        if name_col and name_col in row:
            company_name = row[name_col]
        if not company_name:
            company_name = "blank"
            
        print(f"Using company name: '{company_name}'")
        
        company_names.append(company_name)  # Store exact name for later matching
        
        # Get description without truncation
        description = row[desc_col] if desc_col in row else ''
        
        # Start with mandatory fields - more concise format
        company_data = [f"Company: {company_name}"]
        
        # Add mandatory fields (without truncating description)
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
        
        # Safe data access with existence check for each column
        fields = {
            "Employee Count": row[emp_col] if emp_col in row else '',
            "Description": description,
            "Industries": truncate_text(row[ind_col] if ind_col in row else '', 200),
            "Specialties": truncate_text(row[spec_col] if spec_col in row else '', 200),
            "Products/Services": truncate_text(row[prod_col] if prod_col in row else '', 200),
            "End Markets": truncate_text(row[end_col] if end_col in row else '', 200),
            "Country": row[country_col] if country_col in row else '',
            "Ownership": row[ownership_col] if ownership_col in row else '',
            "Total Raised": row[total_raised_col] if total_raised_col in row else '',
            "Latest Raised": row[latest_raised_col] if latest_raised_col in row else '',
            "Date of Most Recent Investment": row[recent_investment_col] if recent_investment_col in row else ''
        }
        
        # Add each field if it has content
        for label, value in fields.items():
            if value and value.strip():
                company_data.append(f"{label}: {value}")
        
        # Add optional fields if they're mapped and have content
        field_1_col = column_map.get("field_1", "")
        if field_1_col and field_1_col in row and row[field_1_col]:
            company_data.append(f"Field 1: {row[field_1_col]}")
            
        field_2_col = column_map.get("field_2", "")
        if field_2_col and field_2_col in row and row[field_2_col]:
            company_data.append(f"Field 2: {row[field_2_col]}")
            
        field_3_col = column_map.get("field_3", "")
        if field_3_col and field_3_col in row and row[field_3_col]:
            company_data.append(f"Field 3: {row[field_3_col]}")
        
        # Join all company data and add to batch
        company_text = "\n".join(company_data) + "\n\n"
        batch_content += company_text
        
        print(f"Added {len(company_data)} data fields for this company")
        
    print(f"Final batch_content length: {len(batch_content)}")
    print(f"Total companies processed: {len(company_names)}")

    # If batch content is empty, return error results
    if not batch_content.strip():
        print("ERROR: Empty batch content. Returning error results.")
        return [{"company_name": name, "investability_score": -1} for name in company_names]

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
        print(f"Raw API response first 200 chars: {response_text[:200]}...")
        
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Get rows and validate
        rows = response_data.get("rows", [])
        
        # Verify we got correct data
        if not rows:
            print(f"WARNING: No rows found in response. Returning error results.")
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
        import traceback
        print(traceback.format_exc())
        
        # Return companies with -1 scores instead of fallback scoring
        return [{"company_name": name, "investability_score": -1} for name in company_names]

def normalize_csv(csv_content):
    """
    Normalize CSV content to ensure consistent delimiting.
    Handles SourceScrub CSV files with flexible header detection.
    
    Args:
        csv_content: The raw CSV content as bytes
        
    Returns:
        BytesIO: A buffer containing the normalized CSV content,
        int: The index of the header row in the normalized file (usually 4)
    """
    
    print("Starting CSV normalization process...")
    
    # Create a normalized version of the CSV
    normalized_buffer = BytesIO()
    
    try:
        # Decode the content
        try:
            text_content = csv_content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = csv_content.decode('utf-8', errors='replace')
        
        # Split the content into lines
        lines = text_content.splitlines()
        
        # First check if we can find "Company Name" in any of the first 10 rows
        header_row_index = -1
        for i in range(min(10, len(lines))):
            if "Company Name" in lines[i]:
                header_row_index = i
                print(f"Header row detected at line {i+1}")
                break
        
        # If not found, use default (SourceScrub typically has 4 metadata rows)
        if header_row_index < 0:
            header_row_index = 4  # Default to row 5 (index 4)
            print("Header not detected, using default row 5")
        
        # Split into metadata, header, and data
        metadata_lines = lines[:header_row_index]
        header_line = lines[header_row_index] if header_row_index < len(lines) else ""
        data_lines = lines[header_row_index+1:] if header_row_index+1 < len(lines) else []
        
        print(f"File has {len(lines)} lines total.")
        print(f"Using {len(metadata_lines)} metadata lines, 1 header line, and {len(data_lines)} data lines.")
        
        # Setup CSV writer for output
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        
        # Process metadata lines - write as single-column rows
        for line in metadata_lines:
            writer.writerow([line])
        
        # Determine the best delimiter for the header
        delimiters = [',', '\t', ';', '|']
        best_delimiter = ','  # Default
        max_columns = 0
        header_fields = []
        
        # Check which delimiter produces the most fields for the header line
        for delimiter in delimiters:
            # Create a CSV reader with this delimiter
            test_reader = csv.reader([header_line], delimiter=delimiter)
            fields = list(next(test_reader))
            
            print(f"Delimiter '{delimiter}' gives {len(fields)} fields for header")
            
            if len(fields) > max_columns:
                max_columns = len(fields)
                best_delimiter = delimiter
                header_fields = fields
        
        print(f"Selected delimiter '{best_delimiter}' with {max_columns} columns")
        
        # Write the header with properly separated fields
        writer.writerow(header_fields)
        print(f"Processed header with {len(header_fields)} fields")
        
        # Process the data rows with the same delimiter
        for i, line in enumerate(data_lines):
            try:
                # Create a CSV reader with the best delimiter
                row_reader = csv.reader([line], delimiter=best_delimiter)
                fields = list(next(row_reader))
                
                # Ensure each row has the right number of fields
                if len(fields) < max_columns:
                    # Pad with empty fields if needed
                    fields.extend([''] * (max_columns - len(fields)))
                elif len(fields) > max_columns:
                    # Truncate if there are too many fields
                    fields = fields[:max_columns]
                
                # Write the normalized row
                writer.writerow(fields)
                
            except Exception as e:
                # For problematic rows, write as a single field
                writer.writerow([line])
        
        # Get the resulting CSV content
        csv_output = output.getvalue().encode('utf-8')
        
        # Write to the output buffer
        normalized_buffer.write(csv_output)
        normalized_buffer.seek(0)
        
        print("CSV normalization completed successfully")
        return normalized_buffer, len(metadata_lines)
        
    except Exception as e:
        print(f"Error during CSV normalization: {type(e).__name__}: {str(e)}")
        # Return the original content
        original_buffer = BytesIO(csv_content)
        original_buffer.seek(0)
        return original_buffer, 4  # Default to assuming 4 metadata rows

def stream_analysis(
    csv_stream, column_map: Dict[str, str], criteria: str, resume_from: int = 0
) -> Generator[str, None, None]:
    """Yields NDJSON strings as each batch is processed."""
    processed = 0
    total_rows = 0
    
    print("Starting stream analysis")
    
    try:
        # Create a temporary clean copy of the file to avoid metadata/encoding issues
        print("Creating clean copy of the uploaded file...")
        
        # Read the file content
        csv_content = csv_stream.read()
        print(f"File size: {len(csv_content)} bytes")
        
        # Create a new BytesIO object with the content
        clean_csv = BytesIO()
        
        # Check for BOM and other encoding issues
        if csv_content.startswith(b'\xef\xbb\xbf'):
            # Remove BOM if present
            csv_content = csv_content[3:]
            print("Removed BOM from file")
        
        # Write cleaned content to the new buffer
        clean_csv.write(csv_content)
        clean_csv.seek(0)  # Reset pointer to beginning of file
        
        # Try to parse the file normally first
        try:
            # First read without headers to examine the file structure
            print("Examining file structure...")
            preview_data = pd.read_csv(
                clean_csv, 
                dtype=str, 
                header=None,
                nrows=10,  # Just read the first 10 rows to examine structure
                encoding='utf-8'
            )
            
            print(f"Preview data shape: {preview_data.shape}")
            
            # Find header row by looking for "Company Name"
            header_row_index = -1
            for i in range(min(10, len(preview_data))):
                row = preview_data.iloc[i]
                if len(row) >= 1:
                    if "Company Name" in str(row[0]):
                        header_row_index = i
                        print(f"Header row detected at row {i+1}")
                        break
            
            # If not found, use default
            if header_row_index < 0:
                header_row_index = 4  # Default to row 5 (index 4)
                print("Header row not detected, using default row 5")
            
            # Print the first few rows to help diagnose
            for i in range(min(7, len(preview_data))):
                print(f"Row {i}: First few values: {[str(x)[:20] + '...' if len(str(x)) > 20 else str(x) for x in preview_data.iloc[i][:5]]}")
            
            # Try to read with the detected header row
            clean_csv.seek(0)
            print(f"Reading CSV with header at row {header_row_index+1}...")
            
            all_data = pd.read_csv(
                clean_csv, 
                dtype=str, 
                skiprows=header_row_index,
                header=0,  # The row after skipping is the header
                encoding='utf-8'
            )
            
            print(f"Successfully read CSV with {len(all_data)} rows and {len(all_data.columns)} columns")
            
        except Exception as e:
            print(f"Error during initial CSV parsing: {type(e).__name__}: {str(e)}")
            print("Attempting to normalize the CSV format...")
            
            # Normalize the CSV to handle delimiter/format issues
            clean_csv, header_row_index = normalize_csv(csv_content)
            
            # Try parsing again with the normalized content
            try:
                print(f"Reading normalized CSV with skiprows={header_row_index}...")
                
                all_data = pd.read_csv(
                    clean_csv, 
                    dtype=str, 
                    skiprows=header_row_index,
                    header=0,
                    encoding='utf-8'
                )
                
                print(f"Successfully read normalized CSV with {len(all_data)} rows and {len(all_data.columns)} columns")
                
            except Exception as e:
                print(f"Error during normalized CSV parsing: {type(e).__name__}: {str(e)}")
                # One last attempt - try with more permissive settings
                clean_csv.seek(0)
                all_data = pd.read_csv(
                    clean_csv,
                    dtype=str,
                    skiprows=header_row_index,
                    header=0,
                    encoding='utf-8',
                    engine='python',  # Try the python engine which is more forgiving
                    on_bad_lines='skip'  # Skip problematic lines
                )
                print(f"Successfully read normalized CSV using python engine with {len(all_data)} rows")
        
        print("Column headers:", list(all_data.columns)[:10])  # Print first 10 headers
        
        total_rows = len(all_data)
        all_data = all_data.fillna("")
        print(f"Successfully processed file. Found {total_rows} total rows in CSV")
        
        # If resuming, adjust starting point
        if resume_from > 0:
            processed = resume_from
            print(f"Resuming from row {resume_from}")
        
        # Use smaller chunks - this is key for staying within time limits
        chunk_size = 2  # Smaller value than original 1000
        
        # Send an initial progress update to client
        initial_payload = {
            "total_rows": total_rows,
            "progress": processed,
            "status": "starting" if resume_from == 0 else "resuming"
        }
        yield json.dumps(initial_payload) + "\n"
        
        # Process chunks starting from resume point if specified
        for chunk_start in range(processed, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = all_data.iloc[chunk_start:chunk_end]
            num_rows = len(chunk)
            
            # Send a pre-processing notification for this chunk
            chunk_start_payload = {
                "progress": processed,
                "status": "processing_chunk",
                "chunk_start": chunk_start,
                "chunk_end": chunk_end
            }
            yield json.dumps(chunk_start_payload) + "\n"
            
            try:
                # Get company scores from OpenAI
                rows = score_batch(chunk, column_map, criteria)
                
                # Create response payload with rows
                payload = {
                    "progress": processed + num_rows,
                    "total_rows": total_rows,
                    "status": "chunk_complete",
                    "result": rows
                }
                
                processed_pct = round((processed + num_rows) / total_rows * 100)
                print(f"Progress: {processed + num_rows}/{total_rows} ({processed_pct}%)")
                
            except APIError as e:
                print(f"OpenAI API error: {type(e).__name__}: {str(e)}")
                payload = {
                    "progress": processed,
                    "status": "chunk_error",
                    "error": f"OpenAI error: {e.__class__.__name__}: {e}",
                }
            except Exception as e:
                print(f"Unexpected error in batch processing: {type(e).__name__}: {str(e)}")
                payload = {
                    "progress": processed,
                    "status": "chunk_error",
                    "error": f"Error: {type(e).__name__}: {str(e)}",
                }

            processed += num_rows
            json_payload = json.dumps(payload)
            
            yield json_payload + "\n"
            
        # Send a completion message
        final_payload = {
            "progress": processed,
            "total_rows": total_rows,
            "status": "completed"
        }
        yield json.dumps(final_payload) + "\n"
        
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
    
    # Process resume information if provided
    resume_from = int(request.form.get("resumeFrom", "0"))
    if resume_from > 0:
        print(f"Request to resume from row {resume_from}")
    
    # Read the file into memory to prevent "I/O operation on closed file" error
    print("Reading uploaded file into memory...")
    uploaded_file = request.files["file"]
    file_content = uploaded_file.read()
    
    # Create a BytesIO object that can be safely read multiple times
    csv_data = BytesIO(file_content)
    
    print("Starting analysis stream...")
    ndjson_stream = stream_analysis(csv_data, column_map, criteria, resume_from=resume_from)
    
    # Set a max execution time for Vercel's 60-second limit
    def timeout_generator(generator, max_time=50):  # 50 seconds to be safe
        start_time = time.time()
        last_progress = 0  # Track last known progress
        
        # Try to process as much as possible within the time limit
        for item in generator:
            # Extract progress from yielded items if possible
            try:
                data = json.loads(item.strip())
                if "progress" in data:
                    last_progress = data["progress"]
                if "total_rows" in data:
                    total_rows = data["total_rows"]
            except:
                pass
                
            yield item
            
            # Check if we're approaching the time limit
            if time.time() - start_time > max_time:
                # Send a special message indicating we're timing out
                timeout_msg = json.dumps({
                    "status": "timeout",
                    "message": "Processing timeout reached. You can resume from where processing stopped.",
                    "progress": last_progress,
                    "total_rows": total_rows if 'total_rows' in locals() else 0
                }) + "\n"
                yield timeout_msg
                break
    
    # Wrap the stream with timeout handling
    timeout_stream = timeout_generator(ndjson_stream)
    
    print("Returning streaming response")
    return Response(timeout_stream, mimetype="application/x-ndjson")

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
