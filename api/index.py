from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from openai import OpenAI
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route('/api/analyze', methods=['POST'])
def analyze_companies():
    try:
        # Get request data
        data = request.json
        csv_data = data.get('csv_data')
        column_mappings = data.get('column_mappings')
        investing_criteria = data.get('investing_criteria')
        criteria_weights = data.get('criteria_weights', [])
        
        # Convert CSV string to DataFrame
        df = pd.read_csv(pd.StringIO(csv_data))
        
        # Process each row
        results = []
        for _, row in df.iterrows():
            score = score_company(row, column_mappings, investing_criteria, criteria_weights)
            # Add score to row data
            row_dict = row.to_dict()
            row_dict['investability_score'] = score
            results.append(row_dict)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def score_company(row, column_mappings, investing_criteria, criteria_weights):
    # Create a prompt with weighted criteria
    prompt = format_prompt(row, column_mappings, investing_criteria, criteria_weights)
    
    # Call OpenAI API
    for attempt in range(3):
        try:
            MODEL = "gpt-4o"  # Can be configurable
            
            system_msg = (
                "You are a venture‑capital analyst. "
                "Given an investing thesis and company data, reply **ONLY** with JSON "
                'like {"score": <0‑10>} and nothing else.'
            )
            
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=50
            )
            
            return int(json.loads(resp.choices[0].message.content)["score"])
        except Exception as e:
            if attempt == 2:
                print(f"Row {row.name} failed: {e}")
                return None
            time.sleep(2 * (attempt + 1))  # simple back‑off

def format_prompt(row, column_mappings, investing_criteria, criteria_weights):
    # Start with the base investing criteria
    prompt = f"Investing criteria:\n{investing_criteria.strip()}\n\n"
    
    # Add weighted emphasis if provided
    if criteria_weights:
        prompt += "Importance weightings (higher values mean more important):\n"
        for item in criteria_weights:
            prompt += f"- {item['label']}: {item['weight']}x importance\n"
        prompt += "\n"
    
    # Add company data
    prompt += "Company data:\n"
    
    # Add required fields
    for field, column in column_mappings.items():
        if column and column in row:
            value = row[column]
            # Handle NaN values
            if pd.isna(value):
                value = ""
            prompt += f"- {field.replace('_', ' ').title()}: {value}\n"
    
    prompt += "\nRespond with the JSON only."
    return prompt

# Default route to handle Vercel's health checks
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({"status": "API is running"})
