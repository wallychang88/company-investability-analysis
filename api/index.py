# api/analyze.py
from flask import Flask, request, jsonify
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
        csv_data        = body["csv_data"]
        column_map      = body["column_mappings"]
        investing_text  = body["investing_criteria"]
        weightings      = body.get("criteria_weights", [])

        # parse CSV
        hdr_idx = detect_header_row(csv_data)           # auto‑detect
        df = pd.read_csv(io.StringIO(csv_data),
                 header=hdr_idx,                # use that line
                 skip_blank_lines=True)


        # score the ENTIRE batch once
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
            max_tokens=10096,
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

def detect_header_row(csv_text: str, max_scan: int = 10) -> int:
    """
    Return the line index (0 = first line) that most likely contains column
    names, not data.  Strategy:
      • Scan the first *max_scan* lines.
      • Choose the first line where at least half of the cells contain
        alphabetic characters and the cells are mostly unique.
      • Fallback to 0 if nothing matches.
    """
    reader = csv.reader(io.StringIO(csv_text))
    for idx, row in enumerate(reader):
        if idx >= max_scan:
            break
        # ignore completely blank rows
        non_blank = [c for c in row if c.strip()]
        if len(non_blank) < 2:
            continue
        alpha_cells = sum(bool(re.search(r"[A-Za-z]", c)) for c in non_blank)
        unique_cells = len(set(non_blank)) >= len(non_blank) * 0.9
        if alpha_cells >= len(non_blank) / 2 and unique_cells:
            return idx
    return 0


# health‑check
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    return jsonify({"status": "API is running"})
