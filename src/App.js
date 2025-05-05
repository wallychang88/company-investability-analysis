import React, { useState, useEffect } from "react";
import Papa from "papaparse";

/* ───────────────────── Component ───────────────────────────── */
export default function VCAnalysisTool() {
  /* ─────────────────────────── State ─────────────────────────── */
  const [file, setFile] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [processedData, setProcessedData] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [resultCount, setResultCount] = useState(0);

  const [columnMappings, setColumnMappings] = useState({
    employee_count: "",
    description: "",
    industries: "",
    specialties: "",
    products_services: "",
    end_markets: "",
    country: "",
    ownership: "",
    founding_year: "",
  });

  const DEFAULT_THESIS = `We invest in enterprise SaaS and managed services companies that:
• have 80–300 employees
• provide a product or service that supports AI/HPC infrastructure and/or is an enabler of AI/HPC environment buildout
• are not overhyped application‑layer LLM SaaS products
• have a clear, defensible moat (e.g., proprietary data or network effects)`;

  const [investingCriteria, setInvestingCriteria] = useState(DEFAULT_THESIS);
  const [criteriaItems, setCriteriaItems] = useState([]);   // derived below

  /* ───────── Parse bullet points → criteriaItems ───────── */
  useEffect(() => {
    const bullets = investingCriteria
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("•"))
      .map((l) => l.slice(1).trim());

    setCriteriaItems(
      bullets.map((txt, i) => ({ id: `criteria_${i}`, label: txt, weight: 1.0 }))
    );
  }, [investingCriteria]);

  /* ─────────── File upload → parse CSV with header at row 3 ─────────── */
  const handleFileUpload = (e) => {
    const uploaded = e.target.files[0];
    if (!uploaded) return;
    setFile(uploaded);

    // First, let's auto-detect the delimiter
    const fileReader = new FileReader();
    fileReader.onload = (event) => {
      const sample = event.target.result.slice(0, 2000); // Get a sample of the file
      
      // Auto-detect the delimiter by checking for common ones
      let detectedDelimiter = ','; // Default to comma (CSV)
      
      // Check if it's more likely a TSV
      const tabCount = (sample.match(/\t/g) || []).length;
      const commaCount = (sample.match(/,/g) || []).length;
      
      // If there are more tabs than commas, assume it's a TSV
      if (tabCount > commaCount) {
        detectedDelimiter = '\t';
      }
      
      // Parse the entire file, maintaining empty lines
      Papa.parse(uploaded, {
        delimiter: detectedDelimiter,
        header: false,
        skipEmptyLines: false, // Keep empty lines to maintain row count
        complete: ({ data: full }) => {
          // The header is at row 3 (index 2 since arrays are 0‑indexed)
          const HEADER_ROW_INDEX = 4;
          
          // Extract headers from row 3
          const headers = full[HEADER_ROW_INDEX];
          
          // Skip header row and process all data rows
          const rows = full.slice(HEADER_ROW_INDEX + 1).map((row) => {
            const o = {};
            headers.forEach((h, i) => {
              if (h) {
                o[h.trim()] = row[i];
              }
            });
            return o;
          }).filter(row => Object.keys(row).length > 0); // Filter out empty rows

          setParsedData(rows);
          setHeaders(headers.filter(Boolean).map((h) => h.trim()));

          /* Auto‑map column names */
          const lower = headers.map((h) => h?.toLowerCase().trim() || "");
          const findHdr = (needle) => {
            const idx = lower.indexOf(needle);
            return idx !== -1 ? headers[idx].trim() : "";
          };

          const autoMap = {
            employee_count:    findHdr("employee count"),
            description:       findHdr("description"),
            industries:        findHdr("industries"),
            specialties:       findHdr("specialties"),
            products_services: findHdr("products and services") || findHdr("products & services"),
            end_markets:       findHdr("end markets"),
            country:           findHdr("country"),
            ownership:         findHdr("ownership"),
            founding_year:     findHdr("founding year") || findHdr("founded"),
          };
          setColumnMappings((cm) => ({ ...cm, ...autoMap }));
        },
      });
    };
    
    fileReader.readAsText(uploaded);
  };

  const handleColumnMappingChange = (field, value) =>
    setColumnMappings((cm) => ({ ...cm, [field]: value }));

  /**
   * Updates the weight of a specific criteria item
   * @param {string} id    – criteria id
   * @param {number} weight – value between 0‑2
   */
  const updateCriteriaWeight = (id, weight) => {
    setCriteriaItems((items) =>
      items.map((it) => (it.id === id ? { ...it, weight } : it))
    );
  };

  /* ───────────────────────── API call ───────────────────────── */
  const processData = async () => {
  if (!parsedData.length) {
    window.alert("Please upload a file before processing");
    return;
  }

  /** Ensure required columns are mapped */
  const required = [
    "employee_count",
    "description",
    "industries",
    "specialties",
    "products_services",
    "end_markets",
  ];
  const missing = required.filter((c) => !columnMappings[c]);
  if (missing.length) {
    window.alert(`Please map the required columns: ${missing.join(", ")}`);
    return;
  }

  setIsProcessing(true);
  setProgress(0);
  setResultCount(0);

  try {
    const csvString = Papa.unparse(parsedData, {
      quotes: true, // Ensure strings with commas are quoted
      header: true  // Include headers
    });
    const API_URL = "/api/analyze"; // relative path for Vercel

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        csv_data: csvString,
        column_mappings: columnMappings,
        investing_criteria: investingCriteria,
        criteria_weights: criteriaItems,
      }),
    });

    if (!response.ok) throw new Error("API request failed");
    
    // Get the reader for the stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    // Read the stream
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      // Decode and add to buffer
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete lines
      let lines = buffer.split('\n');
      buffer = lines.pop(); // Keep the last incomplete line in the buffer
      
      for (const line of lines) {
        if (line.trim() === '') continue;
        
        try {
          const data = JSON.parse(line);
          console.log("Received data:", data); // Add debugging
          
          if (data.type === 'progress') {
            // Update progress
            setResultCount(data.count);
            setProgress(Math.round((data.count / data.total) * 100));
          } else if (data.type === 'results') {
            // Final results
            console.log("Setting processed data:", data.results);
            setProcessedData(data.results);
          }
        } catch (error) {
          console.error("Error parsing JSON:", error, "Line:", line);
        }
      }
    }
    
    // Make sure to process any remaining data at the end
    if (buffer.trim() !== '') {
      try {
        const data = JSON.parse(buffer);
        console.log("Processing final buffer:", data);
        if (data.type === 'results') {
          console.log("Setting final processed data:", data.results);
          setProcessedData(data.results);
        }
      } catch (error) {
        console.error("Error parsing final buffer:", error);
      }
    }
    
    setProgress(100);
    } catch (err) {
      console.error(err);
      window.alert("An error occurred while processing the data. Please try again.");
    } finally {
      setIsProcessing(false);
    };

  const downloadResults = () => {
    if (!processedData.length) return;
    
    // Create a new array with all the original columns and data,
    // with the investability score added as a new column
    const outputData = processedData.map(row => {
      // Start with a copy of the original row
      const outputRow = { ...row };
      
      // Make sure the investability score is the last column
      // by temporarily removing it
      const score = outputRow.investability_score;
      delete outputRow.investability_score;
      
      // Add it back as the final column
      outputRow.investability_score = score;
      
      return outputRow;
    });
    
    // Generate CSV with all columns preserved
    const csv = Papa.unparse(outputData, {
      columns: [...headers, 'investability_score'] // Ensure all original headers + score
    });
    
    // Create and trigger download
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "investability_analysis.csv";
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  /* ─────────────────────────── UI ─────────────────────────── */
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="p-6 max-w-6xl mx-auto">
        {/* ───── Header Card ───── */}
        <div className="bg-white rounded-xl shadow-xl overflow-hidden mb-8">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-8 text-white text-center">
            <h1 className="text-3xl font-bold">VC Investability Analysis Tool</h1>
          </div>

          {/* ░░░░░ Main content ░░░░░ */}
          <div className="p-8 space-y-10">
            {/* 1. File Upload */}
            <section className="p-6 border border-blue-100 rounded-lg bg-blue-50 space-y-4">
              <h2 className="text-xl font-semibold text-blue-800 flex items-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                1. Upload CSV File
              </h2>
              <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-blue-200 rounded-lg cursor-pointer bg-white hover:bg-blue-50 transition text-center">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-12 h-12 text-blue-400"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
                    clipRule="evenodd"
                  />
                </svg>
                <span className="mt-2 text-sm text-gray-600">
                  {file ? file.name : "Select a CSV file"}
                </span>
              </label>
            </section>

            {/* 2. CSV Preview */}
            {parsedData.length > 0 && (
              <section className="p-6 border border-blue-100 rounded-lg bg-white overflow-x-auto">
                <h2 className="text-xl font-semibold text-blue-800 mb-4 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 mr-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  2. CSV Preview <span className="ml-2 text-sm text-blue-500">(First 5 rows)</span>
                </h2>

                <div className="border rounded-lg overflow-hidden">
                  <table className="min-w-full divide-y divide-gray-200 text-sm">
                    <thead className="bg-gray-100">
                      <tr>
                        {headers.map((h) => (
                          <th
                            key={h}
                            className="px-4 py-3 text-left font-medium text-gray-500 uppercase tracking-wider"
                          >
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
                </div>
                <p className="text-right text-xs text-gray-500 mt-2">
                  Showing 5 of {parsedData.length} rows
                </p>
              </section>
            )}

            {/* 3. Column Mapping */}
            {headers.length > 0 && (
              <section className="p-6 border border-blue-100 rounded-lg bg-white space-y-6">
                <h2 className="text-xl font-semibold text-blue-800 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 mr-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 11.5V14m0-2.5v-6a2.5 2.5 0 015 0v6a2.5 2.5 0 01-5 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M11 17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 00-1-1h-2a1 1 0 00-1 1v2z"
                    />
                  </svg>
                  3. Map Columns
                </h2>

                {/** Required columns */}
                <div className="p-4 bg-blue-50 rounded-lg space-y-4">
                  <h3 className="font-medium text-blue-700 flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-1"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                    Required Columns
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {["employee_count", "description", "industries", "specialties", "products_services", "end_markets"].map((field) => (
                      <div key={field}>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          {field.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                          <span className="text-red-500 ml-1">*</span>
                        </label>
                        <select
                          value={columnMappings[field]}
                          onChange={(e) => handleColumnMappingChange(field, e.target.value)}
                          className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-200"
                        >
                          <option value="">Select column</option>
                          {headers.map((h) => (
                            <option key={h} value={h}>
                              {h}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>

                {/** Optional columns */}
                <div className="p-4 bg-green-50 rounded-lg space-y-4">
                  <h3 className="font-medium text-green-700 flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-1"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                    Optional Columns
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {["country", "ownership", "founding_year"].map((field) => (
                      <div key={field}>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          {field.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                          <span className="text-green-600 ml-1">(Optional)</span>
                        </label>
                        <select
                          value={columnMappings[field]}
                          onChange={(e) => handleColumnMappingChange(field, e.target.value)}
                          className="block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-200"
                        >
                          <option value="">Select column</option>
                          {headers.map((h) => (
                            <option key={h} value={h}>
                              {h}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}

            {/* 4. Investing Criteria & Weights */}
            <section className="p-6 border border-blue-100 rounded-lg bg-white space-y-6">
              <h2 className="text-xl font-semibold text-blue-800 flex items-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                4. Define Investing Criteria
              </h2>

              <div className="bg-indigo-50 p-4 rounded-lg">
                <textarea
                  value={investingCriteria}
                  onChange={(e) => setInvestingCriteria(e.target.value)}
                  rows={6}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-200"
                  placeholder="Enter bullet‑point criteria using the • symbol..."
                />
                <p className="mt-2 text-xs text-indigo-600">
                  Use bullet points (•) to define each criterion. The system will automatically detect and weight them.
                </p>
              </div>

              {/* Weight sliders */}
              <div className="space-y-4">
                <h3 className="font-medium text-indigo-700 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 mr-1"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z"
                      clipRule="evenodd"
                    />
                  </svg>
                  Criteria Weighting
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {criteriaItems.map((item) => (
                    <div key={item.id} className="bg-gray-50 p-4 rounded-lg shadow-sm">
                      <label className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700 truncate" title={item.label}>
                          {item.label}
                        </span>
                        <span className="text-sm font-bold text-indigo-600">{item.weight.toFixed(1)}×</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={item.weight}
                        onChange={(e) => updateCriteriaWeight(item.id, parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer focus:outline-none"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0</span><span>1</span><span>2</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* Process Button */}
            <div className="text-center">
              <button
                onClick={processData}
                disabled={isProcessing || !parsedData.length}
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white text-lg font-medium rounded-xl hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50 shadow-lg transform transition hover:-translate-y-1 flex items-center justify-center"
              >
                {isProcessing ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Processing...
                  </>
                ) : (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-2"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M6.672 1.911a1 1 0 10-1.932.518l.259.966a1 1 0 001.932-.518l-.26-.966zM2.429 4.74a1 1 0 10-.517 1.932l.966.259a1 1 0 00.517-1.932l-.966-.26zm8.814-.569a1 1 0 00-1.415-1.414l-.707.707a1 1 0 101.415 1.415l.707-.708zm-7.071 7.072l.707-.707A1 1 0 003.465 9.12l-.708.707a1 1 0 001.415 1.415zm3.2-5.171a1 1 0 00-1.3 1.3l4 10a1 1 0 001.823.075l1.38-2.759 3.018 3.02a1 1 0 001.414-1.415l-3.019-3.02 2.76-1.379a1 1 0 00-.076-1.822l-10-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                    Analyze Companies
                  </>
                )}
              </button>
            </div>

            {/* Progress Bar */}
            {isProcessing && (
              <section className="p-6 border border-blue-100 rounded-lg bg-white">
                <h2 className="text-xl font-semibold text-blue-800 mb-4 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 mr-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4"
                    />
                  </svg>
                  Processing Companies
                </h2>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold text-blue-600">
                    <span>Progress</span>
                    <span>
                      {resultCount} / {parsedData.length} companies
                    </span>
                  </div>

                  <div className="w-full h-3 bg-blue-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>

                  <p className="text-center text-sm text-gray-500">
                    {progress < 100 ? "Analyzing..." : "Analysis complete!"}
                  </p>
                </div>
              </section>
            )}

            {/* 5. Analysis Results */}
            {processedData.length > 0 && !isProcessing && (
              <section className="p-6 border border-blue-100 rounded-lg bg-white space-y-8 overflow-x-auto">
                <h2 className="text-xl font-semibold text-blue-800 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 mr-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  5. Analysis Results
                </h2>

               {/* Top 5 table */}
              <div>
                <h3 className="font-medium text-gray-700 mb-4">Top Companies by Investability Score</h3>
                <div className="border border-gray-200 rounded-lg overflow-hidden shadow">
                  <table className="min-w-full divide-y divide-gray-200 text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">
                          Company Name
                        </th>
                        <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">
                          Founding Year
                        </th>
                        <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">
                          Employee Count
                        </th>
                        <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">
                          Investability Score
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {processedData
                        .slice()
                        .sort((a, b) => b.investability_score - a.investability_score)
                        .slice(0, 5)
                        .map((row, idx) => {
                          // Find the appropriate column names from mappings
                          const companyNameColumn = headers.find(h => h.toLowerCase().includes('company') || h.toLowerCase().includes('name'));
                          const foundingYearColumn = columnMappings.founding_year;
                          const employeeCountColumn = columnMappings.employee_count;
                          
                          return (
                            <tr key={idx} className={idx % 2 ? "bg-gray-50" : ""}>
                              <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                                {companyNameColumn ? row[companyNameColumn] : "N/A"}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                                {foundingYearColumn ? row[foundingYearColumn] : "N/A"}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                                {employeeCountColumn ? row[employeeCountColumn] : "N/A"}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span
                                  className={`px-3 py-1 inline-flex text-sm font-bold rounded-full ${
                                    row.investability_score >= 7
                                      ? "bg-green-100 text-green-800"
                                      : row.investability_score >= 4
                                      ? "bg-yellow-100 text-yellow-800"
                                      : "bg-red-100 text-red-800"
                                  }`}
                                >
                                  {row.investability_score}/10
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </div>

                {/* Score distribution */}
                <div className="p-6 bg-gray-50 rounded-lg shadow-inner">
                  <h3 className="font-medium text-gray-700 mb-4">Score Distribution</h3>
                  <div className="flex items-end justify-between h-48 space-x-2 px-2">
                    {[...Array(11).keys()].map((score) => {
                      const count = processedData.filter((r) => r.investability_score === score).length;
                      const pct = (count / processedData.length) * 100;
                      return (
                        <div key={score} className="flex flex-col items-center w-full">
                          <span className="text-xs mb-1">{count}</span>
                          <div
                            className={`w-full rounded-t-md ${
                              score >= 7
                                ? "bg-green-500"
                                : score >= 4
                                ? "bg-yellow-500"
                                : "bg-red-500"
                            }`}
                            style={{ height: `${Math.max(5, pct * 2)}px` }}
                          />
                          <span className="text-xs mt-1 font-medium">{score}</span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-4">
                    <span>Poor Match</span>
                    <span>Moderate Match</span>
                    <span>Strong Match</span>
                  </div>
                </div>

                <div className="text-center">
                  <button
                    onClick={downloadResults}
                    className="mt-8 px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 shadow-md inline-flex items-center"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-2"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                    Download Results CSV
                  </button>
                </div>
              </section>
            )}

            {/* Footer */}
          </div>
        </div>
      </div>
    </div>
  );
};
