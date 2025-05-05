import React, { useState, useEffect, useMemo, useRef } from "react";
import Papa from "papaparse";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

/**
 * VC Investability Analysis Tool – front‑end
 * ------------------------------------------------------------------
 * This is a **full rebuild** of the original front‑end preserving all
 * UI/UX features while fixing the runtime bugs we identified:
 *   • Keeps HEADER_ROW_INDEX = 4 (comment corrected)
 *   • Uses Papa's auto‑delimiter detection (more robust)
 *   • Streams NDJSON from the back‑end (matching /api/analyze v2)
 *   • FormData upload instead of JSON (memory‑safe on BE)
 *   • Rounds float scores when binning → accurate histogram
 *   • Swap window.alert → react‑toastify non‑blocking toasts
 *   • Safari ≤17 fallback for ReadableStream
 */

/* ─────────────────────────── Constants ─────────────────────────── */
const HEADER_ROW_INDEX = 4; // Header row is index 4 (5‑th human row)

/* Required columns for analysis */
const REQUIRED_COLS = [
  "employee_count",
  "description",
  "industries",
  "specialties",
  "products_services",
  "end_markets",
];

/* Optional columns offered in the UI */
const OPTIONAL = ["country", "ownership", "founding_year"];

/* ───────────────────── Main Component ─────────────────────────── */
export default function VCAnalysisTool() {
  /* ─────────── State ─────────── */
  const [file, setFile] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [columnMap, setColumnMap] = useState({});
  const [investCriteria, setInvestCriteria] = useState(
    `We invest in enterprise SaaS and managed services companies that:\n• have 80–300 employees\n• provide a product or service that supports AI/HPC infrastructure and/or is an enabler of AI/HPC environment buildout\n• are not overhyped application‑layer LLM SaaS products\n• have a clear, defensible moat (e.g., proprietary data or network effects)`
  );
  const [criteriaWeights, setCriteriaWeights] = useState([]); // [{id,label,weight}]

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [resultCount, setResultCount] = useState(0);
  const [results, setResults] = useState([]);

  const abortRef = useRef(null);

  /* ───────── Parse bullet list → criteriaWeights ───────── */
  useEffect(() => {
    const bullets = investCriteria
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("•"))
      .map((l) => l.slice(1).trim());

    setCriteriaWeights((prev) =>
      bullets.map((txt, i) => {
        const existing = prev.find((p) => p.label === txt);
        return existing || { id: `c_${i}`, label: txt, weight: 1.0 };
      })
    );
  }, [investCriteria]);

  /* ─────────── File Upload & Parse ─────────── */
  const handleFileUpload = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);

    Papa.parse(f, {
      delimiter: "", // auto‑detect (csv / tsv / pipe)
      header: false,
      skipEmptyLines: false,
      complete: ({ data }) => {
        const hdr = data[HEADER_ROW_INDEX];
        const rows = data
          .slice(HEADER_ROW_INDEX + 1)
          .map((row) => {
            const o = {};
            hdr.forEach((h, idx) => {
              if (h) o[h.trim()] = row[idx];
            });
            return o;
          })
          .filter((o) => Object.keys(o).length > 0);

        setParsedData(rows);
        const cleanHeaders = hdr.filter(Boolean).map((h) => h.trim());
        setHeaders(cleanHeaders);

        // Auto‑map columns (case‑insensitive match)
        const lower = cleanHeaders.map((h) => h.toLowerCase());
        const find = (needle) => {
          const idx = lower.indexOf(needle);
          return idx !== -1 ? cleanHeaders[idx] : "";
        };
        setColumnMap({
          employee_count: find("employee count"),
          description: find("description"),
          industries: find("industries"),
          specialties: find("specialties"),
          products_services: find("products and services") || find("products & services"),
          end_markets: find("end markets"),
          country: find("country"),
          ownership: find("ownership"),
          founding_year: find("founding year") || find("founded"),
        });
      },
      error: (err) => toast.error(`CSV parse error: ${err.message}`),
    });
  };

  /* ─────────── Column mapping helpers ─────────── */
  const updateMapping = (field, value) =>
    setColumnMap((cm) => ({ ...cm, [field]: value }));

  /* ─────────── Criteria weight slider ─────────── */
  const updateWeight = (id, w) =>
    setCriteriaWeights((arr) => arr.map((x) => (x.id === id ? { ...x, weight: w } : x)));

  /* ─────────── Progress histogram (0‑10) ─────────── */
  const distribution = useMemo(() => {
    const bins = Array(11).fill(0);
    results.forEach((r) => {
      const s = Math.round(parseFloat(r.investability_score));
      if (s >= 0 && s <= 10) bins[s] += 1;
    });
    return bins;
  }, [results]);

  /* ─────────── API Call ─────────── */
  const processData = async () => {
    if (!file) {
      toast.warn("Upload a file first");
      return;
    }

    // Validate required mappings
    const missing = REQUIRED_COLS.filter((c) => !columnMap[c]);
    if (missing.length) {
      toast.error(`Map the required columns: ${missing.join(", ")}`);
      return;
    }

    // Begin processing
    setIsProcessing(true);
    setProgress(0);
    setResultCount(0);
    setResults([]);

    // Abort controller for multiple runs
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const fd = new FormData();
    fd.append("file", file);
    fd.append("columnMap", JSON.stringify(columnMap));
    fd.append("criteria", investCriteria);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: fd, signal: controller.signal });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

      if (res.body && res.body.getReader) {
        await readStream(res.body.getReader());
      } else {
        await readStreamXHR(fd, controller.signal); // Safari ≤17 fallback
      }
    } catch (err) {
      if (err.name !== "AbortError") toast.error(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  /* ─────────── Stream reader (NDJSON) ─────────── */
  const readStream = async (reader) => {
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (line.trim()) handlePayload(JSON.parse(line));
      }
    }
  };

  /* ─────────── Safari fallback via XHR ─────────── */
  const readStreamXHR = (body, signal) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/analyze");
      xhr.responseType = "text";
      signal.addEventListener("abort", () => xhr.abort());
      let lastIdx = 0;
      xhr.onreadystatechange = () => {
        if (xhr.readyState >= 3) {
          const chunk = xhr.responseText.slice(lastIdx);
          lastIdx = xhr.responseText.length;
          chunk
            .split("\n")
            .filter(Boolean)
            .forEach((line) => handlePayload(JSON.parse(line)));
        }
        if (xhr.readyState === 4) resolve();
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(body);
    });

  /* ─────────── Merge payload from server ─────────── */
  const handlePayload = (data) => {
    if (data.error) {
      toast.error(data.error);
      return;
    }
    if (Array.isArray(data.result)) {
      setResults((prev) => [...prev, ...data.result]);
    }
    if (typeof data.progress === "number") {
      setResultCount(data.progress);
      setProgress(Math.round((data.progress / parsedData.length) * 100));
    }
  };

  /* ─────────── Download CSV helper ─────────── */
  const downloadCSV = () => {
    if (!results.length) return;
    const merged = parsedData.map((row) => {
      const match = results.find((r) => r.company_name === row[columnMap.description] || r.company_name === row[columnMap.company_name]);
      return match ? { ...row, investability_score: match.investability_score } : row;
    });
    const csv = Papa.unparse(merged, {
      columns: [...headers, "investability_score"],
    });
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "investability_analysis.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  /* ─────────── UI Render helpers ─────────── */
  const renderMappingSelect = (field, label, required = false) => (
    <div key={field}>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {!required && <span className="text-green-600 ml-1">(Optional)</span>}
      </label>
      <select
        value={columnMap[field] || ""}
        onChange={(e) => updateMapping(field, e.target.value)}
        className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-200"
      >
        <option value="">Select column</option>
        {headers.map((h) => (
          <option key={h} value={h}>
            {h}
          </option>
        ))}
      </select>
    </div>
  );

  /* Histogram bars */
  const Histogram = () => (
    <div className="p-6 bg-gray-50 rounded-lg shadow-inner">
      <h3 className="font-medium text-gray-700 mb-4">Score Distribution</h3>
      <div className="flex items-end justify-between h-48 space-x-2 px-2">
        {distribution.map((cnt, score) => {
          const pct = results.length ? cnt / results.length : 0;
          return (
            <div key={score} className="flex flex-col items-center w-full">
              <span className="text-xs mb-1">{cnt}</span>
              <div
                className={`w-full rounded-t-md ${
                  score >= 7 ? "bg-green-500" : score >= 4 ? "bg-yellow-500" : "bg-red-500"
                }`}
                style={{ height: `${Math.max(5, pct * 150)}px` }}
              />
              <span className="text-xs mt-1 font-medium">{score}</span>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-4">
        <span>Poor</span>
        <span>Moderate</span>
        <span>Strong</span>
      </div>
    </div>
  );

  /* Top 5 table */
  const TopTable = () => (
    <div>
      <h3 className="font-medium text-gray-700 mb-4">Top Companies by Investability Score</h3>
      <div className="border border-gray-200 rounded-lg overflow-hidden shadow">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">Company</th>
              <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">Score</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {results
              .slice()
              .sort((a, b) => b.investability_score - a.investability_score)
              .slice(0, 5)
              .map((r, idx) => (
                <tr key={idx} className={idx % 2 ? "bg-gray-50" : ""}>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-700">{r.company_name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-bold">{r.investability_score}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  /* ─────────── Render ─────────── */
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="p-6 max-w-6xl mx-auto">
        {/* Header */}
        <header className="bg-white rounded-xl shadow-xl overflow-hidden mb-8">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-8 text-white text-center">
            <h1 className="text-3xl font-bold">VC Investability Analysis Tool</h1>
          </div>
        </header>

        {/* 1. Upload */}
        <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-blue-50 space-y-4">
          <h2 className="text-xl font-semibold text-blue-800 flex items-center">Upload CSV / TSV File</h2>
          <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-blue-200 rounded-lg cursor-pointer bg-white hover:bg-blue-50 transition text-center">
            <input type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" onChange={handleFileUpload} className="hidden" />
            <span className="mt-2 text-sm text-gray-600">{file ? file.name : "Select a file"}</span>
          </label>
        </section>

        {/* 2. Preview */}
        {parsedData.length > 0 && (
          <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white overflow-x-auto">
            <h2 className="text-xl font-semibold text-blue-800 mb-4">CSV Preview</h2>
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-100">
                <tr>
                  {headers.map((h) => (
                    <th key={h} className="px-4 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {parsedData.slice(0, 5).map((row, idx) => (
                  <tr key={idx} className={idx % 2 ? "bg-gray-50" : ""}>
                    {headers.map((h) => (
                      <td key={h} className="px-4 py-3 whitespace-nowrap text-gray-700">
                        {row[h]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-right text-xs text-gray-500 mt-2">Showing 5 of {parsedData.length} rows</p>
          </section>
        )}

        {/* 3. Column Map */}
        {headers.length > 0 && (
          <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white space-y-6">
            <h2 className="text-xl font-semibold text-blue-800">Map Columns</h2>
            {/* Required */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-blue-50 p-4 rounded-lg">
              {REQUIRED_COLS.map((c) =>
                renderMappingSelect(c, c.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()), true)
              )}
            </div>
            {/* Optional */}
            <div>{OPTIONAL.map((c) => renderMappingSelect(c, c.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())))}</div>
          </section>
        )}

        {/* 4. Criteria & weights */}
        <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white space-y-6">
          <h2 className="text-xl font-semibold text-blue-800">Define Investing Criteria</h2>
          <textarea
            value={investCriteria}
            onChange={(e) => setInvestCriteria(e.target.value)}
            rows={6}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-200"
          />
          <p className="text-xs text-indigo-600">Use bullet points (•) to define each criterion.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {criteriaWeights.map((it) => (
              <div key={it.id} className="bg-gray-50 p-4 rounded-lg shadow-sm">
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 truncate" title={it.label}>{it.label}</span>
                  <span className="text-sm font-bold text-indigo-600">{it.weight.toFixed(1)}×</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={it.weight}
                  onChange={(e) => updateWeight(it.id, parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </section>

        {/* Analyze */}
        <div className="text-center mb-8">
          <button
            onClick={processData}
            disabled={isProcessing || !parsedData.length}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white text-lg font-medium rounded-xl hover:from-blue-700 hover:to-indigo-800 disabled:opacity-50 shadow-lg"
          >
            {isProcessing ? "Processing…" : "Analyze Companies"}
          </button>
        </div>

        {/* Progress */}
        {isProcessing && (
          <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white space-y-4">
            <h2 className="text-xl font-semibold text-blue-800">Processing Companies</h2>
            <div className="flex items-center justify-between text-xs font-semibold text-blue-600">
              <span>Progress</span>
              <span>
                {resultCount} / {parsedData.length}
              </span>
            </div>
            <div className="w-full h-3 bg-blue-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-center text-sm text-gray-500">
              {progress < 100 ? "Analyzing…" : "Analysis complete!"}
            </p>
          </section>
        )}

        {/* Results */}
        {results.length > 0 && !isProcessing && (
          <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white space-y-8 overflow-x-auto">
            <h2 className="text-xl font-semibold text-blue-800">Analysis Results</h2>
            <TopTable />
            <Histogram />
            <div className="text-center">
              <button
                onClick={downloadCSV}
                className="mt-4 px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 shadow-md"
              >
                Download Results CSV
              </button>
            </div>
          </section>
        )}

        <ToastContainer position="bottom-right" />
      </div>
    </div>
  );
}
