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

MODEL_NAME = "gpt-4.1"  
MAX_TOKENS = 1500
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
    """Returns the system prompt with user‑supplied criteria embedded and Carrick investment context."""
    
    # Get today's date in a readable format
    from datetime import datetime
    today_date = datetime.now().strftime("%B %d, %Y")
    
    # Investment context derived from Carrick portfolio analysis
    investment_context = """
    Representative investments in our portfolio typically include:
    
    - Enterprise B2B technology companies with strong SaaS or recurring revenue models
    - Companies specializing in: cybersecurity, identity management, data analytics, workflow automation, compliance solutions, healthcare technology, and fintech
    - Firms with approximately 80-350 employees (STRICT REQUIREMENT - companies with >650 employees should receive scores no higher than 4 and companies with <70 employees should receive scores no higher than a 6)
    - Primarily located in the US, with some investments in UK, Israel, and Canada
    - Companies that have typically raised between $5M and $150M in total funding
    - Strong emphasis on AI/ML, cloud infrastructure, and specialized industry solutions
    - Focus on regulated industries including healthcare, financial services, and legal
    - Companies addressing critical security, compliance, operational efficiency, and financial access needs
    
    High-scoring companies typically demonstrate:
    1. Clear product-market fit with enterprise customers
    2. Recurring revenue models with strong retention metrics
    3. Technology that addresses critical business needs with measurable ROI
    4. Solutions for underserved or rapidly growing market segments
    
    IMPORTANT SCORING GUIDELINES:
    - Companies with <70 employees should receive scores no higher than 6
    - Companies with >650 employees should receive scores no higher than 4
    - Companies with >1000 employees should receive scores no higher than 2
    - Companies with >2000 employees should receive a score of 0-1
    - Companies with missing employee count data (FTEs) should receive scores no higher than 5
    - Only companies that strongly match our criteria should receive scores of 7-10
    - Companies with >$150M in total funding should receive scores no higher than 7
    - Companies with >$200M in total funding should receive scores no higher than 6
    - Only companies with Venture Capital, Private, Private Equity, or Seed ownership should score above 1. All others (e.g., Non-Profit, Gov Agency, Edu Institution, Public Sub, Private Sub, Public, Investment Company) must receive scores no higher than 1.
    """
    
    # Example companies from portfolio that represent ideal investments
    example_companies = """
    Example 1: Saviynt
    Description: Leading provider of identity security solutions, offering cloud identity management, privileged access management, and application access governance.
    Industries: Computer & Network Security
    Specialties: Data access governance, SOD monitoring, remediation, continuous controls monitoring, privilege access governance, application GRC, cloud security
    Products/Services: IT Services
    End Markets: Healthcare, Business Services, Education, Finance, Industrials, Government
    Employee Count: 180
    Score: 9
    Rationale: Strong fit with cybersecurity focus, appropriate size, enterprise SaaS model
    
    Example 2: Exiger
    Description: Software development company focused on supply chain and third-party risk management, audit and assurance, construction monitoring, and program design.
    Industries: Computer Software
    Specialties: Regulatory strategy, compliance governance, financial crime compliance, due diligence, investigations, technology, analytics, artificial intelligence, supply chain risk management
    Products/Services: Software Applications and Web-Based Platforms
    End Markets: Healthcare, Industrials, Business Services, Finance
    Employee Count: 250
    Score: 9
    Rationale: Risk management solution, appropriate size, enterprise customers
    
    Example 3: LegalSifter
    Description: Consulting firm specializing in contract management solutions, offering AI-contract review, customizable document-assembly tools, and contract management services.
    Industries: Law Practice & Legal Services
    Specialties: Contract Management, Law, Natural Language Processing, Machine Learning, Artificial Intelligence, Legal, Contract Review, Document Review
    Products/Services: Software Applications and Web-Based Platforms
    End Markets: Healthcare, Business Services, Education, Industrials, Finance
    Employee Count: 85
    Score: 8
    Rationale: AI-driven legal tech, growing company, strong vertical focus
    
    Example 4: OnPay
    Description: Payroll company specializing in employee payroll, HR automation, and benefits solutions, offering direct deposit, hiring and onboarding, PTO management, and employee self-service.
    Industries: Human Resources, Staffing, & Recruiting
    Specialties: Payroll, HR, benefits, small business, accountants, API
    Products/Services: Contracting Services
    End Markets: Business Services, Healthcare, Education, Consumer Services
    Employee Count: 186
    Score: 9
    Rationale: HR tech focus, appropriate size, scalable solution
    
    Example 5: DailyPay
    Description: Financial service company offering earned wage access enabling employees to access earned income before payday, transforming how businesses pay employees.
    Industries: Investment Banking, Banking
    Specialties: Finance as a service, on-demand finance, receivable factoring, financial wellness, fintech, technology
    Products/Services: Financial Services
    End Markets: Business Services, Healthcare
    Employee Count: 350
    Score: 8
    Rationale: Innovative fintech, appropriate size, strong growth potential
    
    """
    
    # Criteria handling instruction
    criteria_instruction = """
    IMPORTANT ADDITIONAL SCORING EMPHASIS!!:
    The following criteria should be considered as specialized focus areas that modify the overall scoring approach. DO NOT boost scores for companies that match these criteria. Instead, REDUCE scores (by 1-2 points) for companies that DO NOT match these criteria, while keeping scores unchanged for companies that DO match these criteria. This maintains appropriate scores for companies matching our current interests while downgrading those that don't align with our specialized focus areas. These criteria are NOT replacements for the core investment criteria but represent specific areas of current interest:    
    """
    
    # If criteria is provided, include it with the instruction, otherwise omit it
    criteria_section = (criteria_instruction + "\n" + criteria.upper()) if criteria.strip() else ""
    
    return (
        f"You are an expert venture analyst. Your task is to evaluate companies based on the investment criteria provided below. Today's date is {today_date}.\n\n"
        "For each company in the input, you MUST assign an investability score from 0 to 10 (integer only).\n\n"
        "Pay special attention to employee count, country, ownership, total raised, latest round, date of most recent investment, and industry fit.\n\n"
        "IMPORTANT: You must return your analysis in VALID JSON format with this exact structure:\n"
        '{"rows":[{"company_name":"Company Name 1", "investability_score":8}, {"company_name":"Company Name 2", "investability_score":5}, ...]}\n\n'
        "Each company MUST have both a company_name and investability_score field.\n\n"
        "VERY IMPORTANT: DO NOT CHANGE THE COMPANY NAMES IN ANY WAY - USE THEM EXACTLY AS PROVIDED. DO NOT EXPAND ABBREIVIATIONS, DO NOT CHANGE CAPITALIZATION, DO NOT CHANGE SPACING OR PUNCTUATION. RETURN THE COMPANY NAMES EXACTLY AS PROVIDED. \n\n"
        "VERY IMPORTANT: Only companies with Venture Capital, Private, Private Equity, or Seed ownership should score above 1. All others (e.g., Non-Profit, Gov Agency, Edu Institution, Public Sub, Private Sub, Public, Investment Company) must be scored 1 or lower."
        "VERY IMPORTANT: Follow the scoring guidelines precisely. companies with >650 employees should receive scores no higher than 4 and companies with <70 employees should receive scores no higher than a 6.\n\n"
        "ANALYSIS DATE CONTEXT: When evaluating criteria related to dates or time periods (like 'X months ago' or 'last Y years'), use the provided date calculations ('Days since...' and 'Years since...') to make precise comparisons.\n\n"
        "CARRICK INVESTMENT CONTEXT:\n" + investment_context + "\n\n"
        "EXAMPLE PORTFOLIO COMPANIES AND SCORING RATIONALE:\n" + example_companies + "\n\n"
        + criteria_section
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
    
    # Date parsing helper function
    def attempt_date_parse(date_str):
        """Try to parse a date string in multiple formats."""
        import re
        from datetime import datetime
        
        if not date_str or not isinstance(date_str, str):
            return None
            
        # Clean the date string - remove parentheses and extra text
        clean_date = re.sub(r'\([^)]*\)', '', date_str).strip()
        
        # List of common date formats
        formats = [
            "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
            "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"
        ]
        
        # Try each format
        for fmt in formats:
            try:
                return datetime.strptime(clean_date, fmt)
            except ValueError:
                continue
        
        # Handle "Month Year" format
        if re.match(r'[A-Za-z]+ \d{4}', clean_date):
            try:
                return datetime.strptime(clean_date, "%B %Y")
            except ValueError:
                try:
                    return datetime.strptime(clean_date, "%b %Y")
                except ValueError:
                    pass
        
        # Handle year-only format
        if re.match(r'^\d{4}$', clean_date):
            return datetime(int(clean_date), 1, 1)
        
        return None
    
    # Get current date for calculations
    from datetime import datetime
    current_date = datetime.now()
    current_date_str = current_date.strftime("%B %d, %Y")
    
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
        
        # Enhanced date processing for investment date
        if recent_investment_col and recent_investment_col in row and row[recent_investment_col]:
            date_str = row[recent_investment_col]
            
            # Add the original date string
            company_data.append(f"Date of Most Recent Investment: {date_str}")
            
            # Try to parse the date
            parsed_date = attempt_date_parse(date_str)
            if parsed_date:
                # Calculate time metrics
                days_since = (current_date - parsed_date).days
                years_since = days_since / 365.0
                months_since = days_since / 30.0
                
                # Add processed date information
                company_data.append(f"Today's Date: {current_date_str}")
                company_data.append(f"Days Since Last Investment: {days_since}")
                company_data.append(f"Months Since Last Investment: {months_since:.1f}")
                company_data.append(f"Years Since Last Investment: {years_since:.2f}")
        
        # Enhanced date processing for latest raised field if different from recent investment
        if latest_raised_col and latest_raised_col in row and row[latest_raised_col]:
            # Skip if it's the same column as recent_investment_col
            if latest_raised_col != recent_investment_col:
                date_str = row[latest_raised_col]
                
                # Add the original date string
                company_data.append(f"Latest Raised: {date_str}")
                
                # Try to parse the date
                parsed_date = attempt_date_parse(date_str)
                if parsed_date:
                    # Calculate time metrics
                    days_since = (current_date - parsed_date).days
                    years_since = days_since / 365.0
                    months_since = days_since / 30.0
                    
                    # Add processed date information
                    company_data.append(f"Days Since Latest Raised: {days_since}")
                    company_data.append(f"Months Since Latest Raised: {months_since:.1f}")
                    company_data.append(f"Years Since Latest Raised: {years_since:.2f}")
        
        # Safe data access with existence check for each column (for non-date fields)
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
        }
        
        # Add each non-date field if it has content and wasn't already added
        for label, value in fields.items():
            # Skip Date of Most Recent Investment and Latest Raised as they're handled separately
            if label not in ["Date of Most Recent Investment", "Latest Raised"]:
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
    
    # Create user message with explicit date context
    user_message = (
        f"Analyze these companies based on our investment criteria. "
        f"Today's date is {current_date_str}. "
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
                }
            except Exception as e:
                print(f"Unexpected error in batch processing: {type(e).__name__}: {str(e)}")
                payload = {
                    "progress": processed,
                    "status": "chunk_error",
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
