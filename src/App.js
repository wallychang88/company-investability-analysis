// src/App.js
import React, { useCallback, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { FixedSizeList as List } from "react-window";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";

/**
 * CSV metadata
 * --------------------------------------------------------------------------
 * Header row is the 5‑th HUMAN row (index 4, zero‑based).  The previous code
 * comment was wrong; the constant has always been correct.
 */
const HEADER_ROW_INDEX = 4;

/**
 * Utility – derive column map heuristically from header names.
 */
const defaultColumnMap = (headers) => ({
  company_name:
    headers.find((h) => /company.?name/i.test(h)) ?? headers[0] ?? "",
  description:
    headers.find((h) => /description|summary/i.test(h)) ?? headers[1] ?? "",
});

/**
 * Virtualised cell renderer to keep the DOM light even for 10k‑row previews.
 */
const TableRow = ({ data, index, style }) => {
  const row = data[index];
  return (
    <div className="table-row grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-2 p-1 border-b" style={style}>
      {row.map((cell, i) => (
        <div key={i} className="truncate">
          {cell}
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const [file, setFile] = useState(null);
  const [headers, setHeaders] = useState([]);
  const [rows, setRows] = useState([]);
  const [criteria, setCriteria] = useState("");
  const [columnMap, setColumnMap] = useState({});
  const [distribution, setDistribution] = useState({});
  const [progress, setProgress] = useState(0);
  const abortControllerRef = useRef(null);

  /**
   * File‑chooser → parse only a small preview so the UI is instant.
   */
  const handleFile = useCallback((e) => {
    const newFile = e.target.files?.[0];
    if (!newFile) return;

    setFile(newFile);
    Papa.parse(newFile, {
      delimiter: "", // auto‑detect
      skipEmptyLines: true,
      preview: HEADER_ROW_INDEX + 20, // header + 20 rows
      complete: ({ data }) => {
        const hdr = data[HEADER_ROW_INDEX];
        const rowsPreview = data.slice(HEADER_ROW_INDEX + 1);
        setHeaders(hdr);
        setRows(rowsPreview);
        setColumnMap(defaultColumnMap(hdr));
      },
      error: (err) => toast.error(`Parse error: ${err.message}`),
    });
  }, []);

  /**
   * Live distribution for the histogram.
   */
  const histogram = useMemo(() => {
    const dist = {};
    Object.values(distribution).forEach((score) => {
      const bucket = Math.round(score); // 7 and 7.0 align
      dist[bucket] = (dist[bucket] || 0) + 1;
    });
    return dist;
  }, [distribution]);

  /**
   * Build FormData & stream results.
   */
  const handleAnalyze = useCallback(async () => {
    if (!file) {
      toast.warning("Please choose a CSV / TSV file first");
      return;
    }
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const fd = new FormData();
    fd.append("file", file);
    fd.append("columnMap", JSON.stringify(columnMap));
    fd.append("criteria", criteria);

    try {
      // Prefer fetch streaming, fall back for Safari ≤17.
      if (ReadableStream && "getReader" in ReadableStream.prototype) {
        await fetchStream(fd, controller.signal);
      } else {
        await fetchXHR(fd, controller.signal);
      }
    } catch (err) {
      if (err.name !== "AbortError") toast.error(err.message);
    }
  }, [file, columnMap, criteria]);

  /**
   * Fetch with real streaming (Chrome, FF, modern Safari).
   */
  const fetchStream = async (body, signal) => {
    setProgress(0);
    setDistribution({});
    const res = await fetch("/api/analyze", { method: "POST", body, signal });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, newline);
        buf = buf.slice(newline + 1);
        if (!line.trim()) continue;
        handlePartial(JSON.parse(line));
      }
    }
  };

  /**
   * XHR fallback – emits `progress` events so we can still update %.
   */
  const fetchXHR = (body, signal) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/analyze");
      xhr.responseType = "text";
      xhr.upload.onprogress = (e) =>
        setProgress(Math.round((e.loaded / e.total) * 100));
      xhr.onprogress = () => {
        const lines = xhr.responseText.split(/\r?\n/).filter(Boolean);
        lines.slice(progress).forEach((line) => handlePartial(JSON.parse(line)));
      };
      xhr.onloadend = () =>
        xhr.status === 200 ? resolve() : reject(new Error(xhr.statusText));
      xhr.onerror = () => reject(new Error("Network error"));
      signal.addEventListener("abort", () => xhr.abort());
      xhr.send(body);
    });

  /**
   * Merge incremental JSON payload from the server.
   */
  const handlePartial = ({ progress: p, result, error }) => {
    setProgress(p);
    if (error) {
      toast.error(error);
      return;
    }
    // result is an array of { company_name, investability_score, rationale }
    setDistribution((prev) => {
      const next = { ...prev };
      result.forEach(({ company_name, investability_score }) => {
        next[company_name] = investability_score;
      });
      return next;
    });
  };

  /**
   * React‑window list height = viewport‑aware
   */
  const tableHeight = Math.min(400, rows.length * 35);

  return (
    <main className="px-6 py-4 max-w-6xl mx-auto space-y-6">
      <h1 className="text-3xl font-semibold">Company Investability Score</h1>

      <section className="space-y-3">
        <input
          type="file"
          accept=".csv,.tsv,text/csv,text/tab-separated-values"
          onChange={handleFile}
          className="file:mr-4"
        />

        {headers.length > 0 && (
          <>
            <div className="border rounded shadow-sm">
              <List
                height={tableHeight}
                itemCount={rows.length}
                itemSize={35}
                width="100%"
                itemData={rows}
              >
                {TableRow}
              </List>
            </div>

            <textarea
              value={criteria}
              onChange={(e) => setCriteria(e.target.value)}
              placeholder="Scoring criteria (optional)…"
              className="w-full p-2 border rounded h-24"
            />

            <button
              onClick={handleAnalyze}
              className="px-4 py-2 bg-blue-600 text-white rounded"
            >
              Analyze
            </button>

            <progress
              max={100}
              value={progress}
              className="w-full h-3 rounded"
            />

            {Object.keys(histogram).length > 0 && (
              <div className="flex gap-1 items-end mt-4">
                {Array.from({ length: 10 }, (_, i) => (
                  <div
                    key={i + 1}
                    className="flex-1 bg-blue-500"
                    style={{ height: `${(histogram[i + 1] || 0) * 10}px` }}
                    title={`${i + 1}: ${histogram[i + 1] || 0}`}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </section>

      <ToastContainer position="bottom-right" />
    </main>
  );
}
