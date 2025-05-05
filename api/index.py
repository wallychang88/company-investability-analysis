# api/analyze.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, time, io
import pandas as pd
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze_companies():
    try:
        body = request.get_json(force=True)
        csv_data        = body["csv_data"]
        column_map      = body["column_mappings"]
        investing_text  = body["investing_criteria"]
        weightings      = body.get("criteria_weights", [])

        # fix #1 – parse CSV
        df = pd.read_csv(io.StringIO(csv_data))

        # ⚠️  fix #2 – score the ENTIRE batch once
        prompt = build_batch_prompt(df, column_map,
                                    investing_text, weightings)

        chat = client.chat.completions.create(
            model="gpt-4o-mini",         # fix #3
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON in the shape "
                        '{"rows":[{<csv row fields>, "investability_score":<0‑10>}, …]}.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4096,
        )

        rows = json.loads(chat.choices[0].message.content)["rows"]
        return jsonify({"success": True, "results": rows})

    except Exception as e:
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
