# api/analyze.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os, json, time, io
import pandas as pd
from openai import OpenAI
import csv, re

app = Flask(__name__)
CORS(app, origins=["https://company-investability-score.vercel.app"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze_companies():
    try:
        body = request.get_json(force=True)
        
        # Log what we received
        print("Request keys:", body.keys())
        
        csv_data = body.get("csv_data", "")
        column_map = body.get("column_mappings", {})
        investing_text = body.get("investing_criteria", "")
        weightings = body.get("criteria_weights", [])
        
        if not csv_data:
            return jsonify({"success": False, "error": "Missing CSV data"}), 400
            
        # Print first part of CSV data to debug
        print(f"CSV data preview: {csv_data[:200]}...")
        print(f"Column mappings: {column_map}")
        
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
        except Exception as e:
            return jsonify({"success": False, "error": f"CSV parsing error: {str(e)}"}), 400
            
        # Check if we got required columns
        missing_columns = []
        for key, col_name in column_map.items():
            if col_name and col_name not in df.columns:
                missing_columns.append(f"{key} (mapped to '{col_name}')")
                
        if missing_columns:
            return jsonify({
                "success": False, 
                "error": f"Missing mapped columns: {', '.join(missing_columns)}"
            }), 400

        # Convert employee count to numeric if it exists
        if 'employee_count' in column_map and column_map['employee_count'] in df.columns:
            df[column_map['employee_count']] = pd.to_numeric(df[column_map['employee_count']], errors='coerce')
            
        # Convert founding year to numeric if it exists
        if 'founding_year' in column_map and column_map['founding_year'] in df.columns:
            df[column_map['founding_year']] = pd.to_numeric(df[column_map['founding_year']], errors='coerce')
        
        def generate():

            total_rows = len(df)
            
            # Send initial progress update
            yield json.dumps({"type": "progress", "count": 0, "total": total_rows}) + "\n"
            
            # Process in batches for better progress feedback
            batch_size = min(10, total_rows)  # Process 10 rows at a time, or fewer if total < 10
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Create a prompt for just this batch
                batch_prompt = build_batch_prompt(batch_df, column_map, investing_text, weightings)
                
                # Call GPT for this batch
                chat = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return ONLY valid JSON in the shape "
                                '{"rows":[{<csv row fields>, "investability_score":<0‑10>}, …]}.'
                            ),
                        },
                        {"role": "user", "content": batch_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=10096,
                )
                
                # Update progress
                yield json.dumps({
                    "type": "progress", 
                    "count": end_idx, 
                    "total": total_rows
                }) + "\n"
            
            # After all batches are processed, get the final score
            final_prompt = build_batch_prompt(df, column_map, investing_text, weightings)
            
            final_chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON in the shape "
                            '{"rows":[{<csv row fields>, "investability_score":<0‑10>}, …]}.'
                        ),
                    },
                    {"role": "user", "content": final_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=10096,
            )
            
            # Send the final results
            rows = json.loads(final_chat.choices[0].message.content)["rows"]
            yield json.dumps({"type": "results", "results": rows}) + "\n"
        
        # Return a streaming response
        return Response(generate(), mimetype='application/x-ndjson')
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {str(e)}\nDetails: {error_details}")
        return jsonify({"success": False, "error": str(e)}), 500
        
# ───────────────────────── helpers ────────────────────────────
def build_batch_prompt(df, column_map, thesis, weightings):
    out = [f"Investing criteria:\n{thesis.strip()}"]
    if weightings:
        out.append("Importance weightings:")
        for w in weightings:
            out.append(f"- {w['label']}: {w['weight']}×")
    out.append("\nCSV rows:")
    for _, row in df.iterrows():
        line = {k: ("" if pd.isna(v) else v) for k, v in row.items()}
        out.append(json.dumps(line))
    out.append("\nReturn JSON only.")
    return "\n".join(out)


# health‑check
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    return jsonify({"status": "API is running"})
