import React, { useState, useEffect, useMemo, useRef } from "react";
import Papa from "papaparse";

/**
 * VC Investability Analysis Tool – front‑end

/* ─────────────────────────── Constants ─────────────────────────── */
const HEADER_ROW_INDEX = 4; // Header row is index 4 (5‑th human row)

/* Required columns for analysis */
const REQUIRED_COLS = [
  "employee_count",
  "description",
  "industries",
  "specialties",
  "products_services",
  "end_markets",
  "country",
  "ownership",
  "total_raised",
  "latest_raised",
  "date_of_most_recent_investment"
];

/* Optional columns offered in the UI */
const OPTIONAL = ["field_1", "field_2", "field_3"];

/* ───────────────────── Main Component ─────────────────────────── */
export default function VCAnalysisTool() {
  /* ─────────── State ─────────── */
  const [file, setFile] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [columnMap, setColumnMap] = useState({});
  const [investCriteria, setInvestCriteria] = useState(
  `We invest in: 
  • Enterprise SaaS, software enabled services and managed service/BPO companies
  • Have between 80 and 300 employees
  • Are based in the US, UK, Canada, or Israel
  • Have raised in total less than $150 million
  • Ownership is private, venture capital, private equity, or seed
  • provide a product or service that supports AI/HPC infrastructure and/or is an enabler of AI/HPC environment buildout
  • have a clear, defensible moat (e.g., proprietary data or network effects)`
  );
  const [criteriaWeights, setCriteriaWeights] = useState([]); // [{id,label,weight}]
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [resultCount, setResultCount] = useState(0);
  const [results, setResults] = useState([]);
  const [lastError, setLastError] = useState(null); // For error state
  const abortRef = useRef(null);
  const [canResume, setCanResume] = useState(false);
  const [resumeState, setResumeState] = useState({
  progress: 0,
  totalRows: 0
});

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
  
  // Reset all analysis-related state when a new file is uploaded
  setFile(f);
  setParsedData([]);
  setHeaders([]);
  setColumnMap({});
  setResults([]);
  setProgress(0);
  setResultCount(0);
  setLastError(null);
  setCanResume(false);
  setResumeState({
    progress: 0,
    totalRows: 0
  });
  
  // If there's an active abort controller, use it to cancel any ongoing processing
  if (abortRef.current) {
    try {
      abortRef.current.abort();
      abortRef.current = null;
    } catch (e) {
      console.warn("Error aborting previous request:", e);
    }
  }

  // Now parse the new file
  Papa.parse(f, {
    delimiter: "", // auto‑detect (csv / tsv / pipe)
    header: false,
    skipEmptyLines: false,
    complete: ({ data }) => {
      // Detect the header row by looking for "Company Name" and "Informal Name"
      let headerRowIndex = -1;
      
      for (let i = 0; i < Math.min(10, data.length); i++) {
        const row = data[i];
        if (Array.isArray(row) && row.length >= 2) {
          // Check if first column contains "Company Name" and second contains "Informal Name"
          if (
            row[0] && row[0].toString().includes("Company Name") && 
            row[1] && row[1].toString().includes("Informal Name")
          ) {
            headerRowIndex = i;
            console.log(`Header row detected at line ${i+1}`);
            break;
          }
        }
      }
      
      // If header row not found, fallback to default (row 5 or row 4)
      if (headerRowIndex < 0) {
        if (data.length > 4) {
          headerRowIndex = 4;  // Row 5 (index 4)
          console.log("Header row not detected, using default row 5");
        } else if (data.length > 3) {
          headerRowIndex = 3;  // Row 4 (index 3)
          console.log("Header row not detected, using shorter default row 4");
        } else if (data.length > 0) {
          headerRowIndex = 0;
          console.log("Header row not detected, using first row");
        } else {
          setLastError("File appears to be empty");
          return;
        }
      }
      
      const hdr = data[headerRowIndex];
      if (!hdr || !Array.isArray(hdr) || hdr.filter(Boolean).length === 0) {
        setLastError("Could not detect headers in file");
        return;
      }
      
      // Process data rows after the header
      const rows = data
        .slice(headerRowIndex + 1)
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
        const idx = lower.indexOf(needle.toLowerCase());
        return idx !== -1 ? cleanHeaders[idx] : "";
      };
      setColumnMap({
        employee_count: find("employee count") || find("employees"),
        description: find("description") || find("company description"),
        industries: find("industries") || find("industry"),
        specialties: find("specialties") || find("specialty"),
        products_services: find("products and services") || find("products & services") || find("products"),
        end_markets: find("end markets") || find("markets"),
        country: find("country") || find("location") || find("hq country"),
        ownership: find("ownership") || find("company type") || find("company status"),
        total_raised: find("total raised") || find("total funding") || find("funding"),
        latest_raised: find("latest raised") || find("latest round") || find("last fundraise"),
        date_of_most_recent_investment: find("date of most recent investment") || find("last investment date") || find("most recent funding date"),
        field_1: "",
        field_2: "",
        field_3: ""
      });
      
      setLastError(null);
    },
    error: (err) => setLastError(`CSV parse error: ${err.message}`),
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
const processData = async (resumeFrom = 0) => {
  setLastError(null);
  
  if (!file) {
    setLastError("Upload a file first");
    return;
  }

    // Validate required mappings
    const missing = REQUIRED_COLS.filter((c) => !columnMap[c]);
    if (missing.length) {
      const missingNames = missing.map(c => c.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()));
      setLastError(`Please map these required columns: ${missingNames.join(", ")}`);
      return;
    }

    // Check if we have any rows to process
    if (parsedData.length === 0) {
      setLastError("No valid data rows found to process");
      return;
    }

    // Validate criteria bullets
    const bullets = investCriteria
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("•"));
    
    if (bullets.length === 0) {
      setLastError("Please add at least one bullet point (•) criterion");
      return;
    }

    // Begin processing
setIsProcessing(true);
  if (resumeFrom === 0) {
    // Only reset progress if starting from beginning
    setProgress(0);
    setResultCount(0);
    setResults([]);
  } else {
    console.log(`Resuming processing from row ${resumeFrom}`);
  }

    // Abort controller for multiple runs
    if (abortRef.current) {
      try {
        abortRef.current.abort();
      } catch (e) {
        console.warn("Error aborting previous request:", e);
      }
    }
    const controller = new AbortController();
    abortRef.current = controller;

const fd = new FormData();
  fd.append("file", file);
  fd.append("columnMap", JSON.stringify(columnMap));
  fd.append("criteria", investCriteria);
  
  // Add resume information if resuming
  if (resumeFrom > 0) {
    fd.append("resumeFrom", resumeFrom.toString());
  }
    
    // Add weights if available
    if (criteriaWeights.length > 0) {
      fd.append("weights", JSON.stringify(
        criteriaWeights.reduce((obj, item) => {
          obj[item.label] = item.weight;
          return obj;
        }, {})
      ));
    }

    try {
    const res = await fetch("/api/analyze", { 
      method: "POST", 
      body: fd, 
      signal: controller.signal 
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`API error (${res.status}): ${errorText || res.statusText}`);
    }

    // Reset resume state since we're processing now
    setCanResume(false);

    if (res.body && typeof res.body.getReader === 'function') {
      await readStream(res.body.getReader());
    } else {
      // Safari ≤17 fallback
      await readStreamXHR(fd, controller.signal);
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      console.error("Analysis error:", err);
      setLastError(err.message || "Unknown error");
    }
  } finally {
    setIsProcessing(false);
  }
};

  /* ─────────── Stream reader (NDJSON) ─────────── */
  const readStream = async (reader) => {
    const decoder = new TextDecoder();
    let buf = "";
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buf += decoder.decode(value, { stream: true });
        let nl;
        
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          
          if (line.trim()) {
            try {
              const payload = JSON.parse(line);
              handlePayload(payload);
            } catch (e) {
              console.error("Failed to parse JSON line:", line, e);
            }
          }
        }
      }
    } catch (e) {
      console.error("Stream reading error:", e);
      throw e;
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
            .forEach((line) => {
              try {
                handlePayload(JSON.parse(line));
              } catch (e) {
                console.error("Failed to parse JSON line:", line, e);
              }
            });
        }
        
        if (xhr.readyState === 4) {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve();
          } else {
            reject(new Error(`XHR Error: ${xhr.status} ${xhr.statusText}`));
          }
        }
      };
      
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(body);
    });

/* ─────────── Merge payload from server ─────────── */
const handlePayload = (data) => {
  if (!data) return;
  
  if (data.error) {
    setLastError(data.error);
    return;
  }
  
  // Handle timeout status from the server
  if (data.status === 'timeout') {
    console.log("Server timeout detected, enabling resume capability");
    setCanResume(true);
    setResumeState({
      progress: data.progress || 0,
      totalRows: data.total_rows || parsedData.length
    });
    setLastError("Processing timed out but can be resumed");
    return;
  }
  
  // Handle status updates
  if (data.status === 'starting' || data.status === 'resuming') {
    console.log(`Analysis ${data.status} with ${data.total_rows} total rows`);
    // Update total rows if provided
    if (data.total_rows) {
      setResumeState(prev => ({...prev, totalRows: data.total_rows}));
    }
  }
  
  // Handle chunk status
  if (data.status === 'chunk_complete' && Array.isArray(data.result)) {
    if (data.result.length > 0) {
      // Check for valid data before updating
      const validResults = data.result.filter(r => 
        r.company_name && typeof r.investability_score !== 'undefined'
      );
      
      if (validResults.length > 0) {
        // Update results state
        setResults(prev => {
          // Filter out duplicates
          const newResults = [...prev];
          validResults.forEach(result => {
            // Check if this company is already in results
            const existingIndex = newResults.findIndex(r => r.company_name === result.company_name);
            if (existingIndex >= 0) {
              // Update existing entry
              newResults[existingIndex] = result;
            } else {
              // Add new entry
              newResults.push(result);
            }
          });
          
          return newResults;
        });
      }
    }
  }
  
  // Regular array results (backward compatibility)
  if (Array.isArray(data.result) && !data.status) {
    // ... existing array handling code ...
  }
  
  if (typeof data.progress === "number") {
    // Always update progress counter for UX feedback
    setResultCount(data.progress);
    
    // Update resume state
    setResumeState(prev => ({...prev, progress: data.progress}));
    
    // Calculate progress percentage
    const totalForCalc = data.total_rows || resumeState.totalRows || parsedData.length;
    const progressPercent = Math.min(100, Math.round((data.progress / totalForCalc) * 100));
    setProgress(progressPercent);
  }
};

     /* ─────────── Download CSV helper ─────────── */
const downloadCSV = () => {
  if (!results.length) return;
  
  // Find name field (either description or company_name)
  const companyField = columnMap.company_name || columnMap.description;
  
  if (!companyField) {
    setLastError("Could not determine company name field");
    return;
  }
  
  // Create the reorganized data
  const reorganizedData = parsedData.map((row) => {
    // Find the corresponding result with the score
    let match = null;
    
    // Try multiple matching strategies to find the corresponding result
    if (companyField) {
      match = results.find((r) => r.company_name === row[companyField]);
    }
    
    if (!match && columnMap.description && companyField !== columnMap.description) {
      match = results.find((r) => r.company_name === row[columnMap.description]);
    }
    
    if (!match && companyField) {
      const companyName = row[companyField];
      if (companyName) {
        match = results.find((r) => 
          r.company_name.toLowerCase().includes(companyName.toLowerCase()) ||
          companyName.toLowerCase().includes(r.company_name.toLowerCase())
        );
      }
    }
    
    // Create new row with reorganized columns
    const newRow = {};
    
    // Define and populate primary columns directly accessing the original row data
    newRow["Company"] = row["Informal Name"] || "";
    newRow["Founded"] = row["Founding Year"] || "";
    newRow["Country"] = row["Country"] || "";
    newRow["FTEs"] = row["Employee Count"] || "";
    newRow["LTM FTE Growth"] = row["12 Months Growth Rate %"] || "";
    newRow["Ownership"] = row["Ownership"] || "";
    newRow["Total Raised"] = row["Total Raised"] || "";
    newRow["Latest Raised"] = row["Latest Raised"] || "";
    newRow["Last Round"] = row["Date of Most Recent Investment"] || "";
    newRow["Investors"] = row["Investors"] || "";
    newRow["Parent Company"] = row["Parent Company"] || "";
    newRow["Investability Score"] = match ? match.investability_score : "N/A";
    newRow["Website"] = row["Website"] || "";
    newRow["Description"] = row["Description"] || "";
    
    // Add remaining columns after the primary ones
    const primaryColumns = [
      "Informal Name", "Founding Year", "Country", "Employee Count", "12 Months Growth Rate %",
      "Ownership", "Total Raised", "Latest Raised", "Date of Most Recent Investment",
      "Investors", "Parent Company", "Website", "Description"
    ];
    
    headers.forEach(originalHeader => {
      // Skip if this column was already included in our primary columns
      if (!primaryColumns.includes(originalHeader)) {
        newRow[originalHeader] = row[originalHeader] || "";
      }
    });
    
    return newRow;
  });
  
  // Define the column order for the output
  const orderedColumns = [
    "Company", "Founded", "Country", "FTEs", "LTM FTE Growth", "Ownership", 
    "Total Raised", "Latest Raised", "Last Round", "Investors", 
    "Parent Company", "Investability Score", "Website", "Description",
    ...headers.filter(h => !([
      "Informal Name", "Founding Year", "Country", "Employee Count", "12 Months Growth Rate %",
      "Ownership", "Total Raised", "Latest Raised", "Date of Most Recent Investment",
      "Investors", "Parent Company", "Website", "Description"
    ].includes(h)))
  ];
  
  // Generate CSV with the new structure
  const csv = Papa.unparse(reorganizedData, {
    columns: orderedColumns
  });
  
  // Create download
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
        {required && <span className="text-red-600 ml-1">*</span>}
      </label>
      <select
        value={columnMap[field] || ""}
        onChange={(e) => updateMapping(field, e.target.value)}
        className={`block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-200 ${
          required && !columnMap[field] ? "border-red-300 bg-red-50" : ""
        }`}
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

/* Top 5 table with enhanced display */
const TopTable = () => {
  // Get matched company data (combines results with original data)
  const topCompanies = useMemo(() => {
    // Early return if no results
    if (results.length === 0) return [];
    
    // Get all results sorted by score
    return results
      .slice()
      .sort((a, b) => b.investability_score - a.investability_score)
      .map((result, index) => {
        // Get the company name from the result
        let companyName = result.company_name;
        
        // Check if company name is blank/empty (marked as "blank" by the backend)
        const isNameBlank = !companyName || companyName.trim() === '' || companyName === "blank";
        
        // Find the original company data using multiple matching strategies
        let originalData = null;
        
        // First try exact match on description field
        if (columnMap.description) {
          originalData = parsedData.find(row => row[columnMap.description] === companyName);
        }
        
        // If no match, try exact match on company_name field
        if (!originalData && columnMap.company_name) {
          originalData = parsedData.find(row => row[columnMap.company_name] === companyName);
        }
        
        // If still no match, try case-insensitive contains match
        if (!originalData && columnMap.description) {
          originalData = parsedData.find(row => {
            const desc = row[columnMap.description];
            return desc && desc.toLowerCase().includes(companyName.toLowerCase());
          });
        }
        
        // If still no match, try case-insensitive match on any column
        if (!originalData) {
          originalData = parsedData.find(row => {
            return Object.values(row).some(value => 
              value && typeof value === 'string' && 
              value.toLowerCase().includes(companyName.toLowerCase())
            );
          });
        }
        
        // Default to empty object if still no match
        originalData = originalData || {};
        
        // Find website URL if it exists
        const websiteCol = headers.find(h => 
          h.toLowerCase() === 'website' || 
          h.toLowerCase() === 'url' || 
          h.toLowerCase() === 'web' || 
          h.toLowerCase() === 'company url'
        );
        
        const websiteUrl = websiteCol && originalData[websiteCol] ? originalData[websiteCol] : '';
        
        // Format website URL - ensure it has http/https
        let formattedUrl = websiteUrl;
        if (formattedUrl && !formattedUrl.startsWith('http')) {
          formattedUrl = 'https://' + formattedUrl;
        }
        
        // Look specifically for "Founding Year" in the CSV
        let foundingYear = 'N/A';
        if (originalData["Founding Year"] && originalData["Founding Year"].trim()) {
          foundingYear = originalData["Founding Year"];
        }
          
        // Get employee count
        const employeeCount = columnMap.employee_count && originalData[columnMap.employee_count]
          ? originalData[columnMap.employee_count]
          : 'N/A';
        
        // For blank company names, use "NAME FIELD BLANK (Row #)" format
        const displayName = isNameBlank 
          ? `NAME FIELD BLANK (Row ${index + 1})` 
          : companyName;
        
        return {
          name: displayName,
          foundingYear,
          employeeCount,
          website: formattedUrl,
          score: result.investability_score
        };
      });
  }, [results, parsedData, columnMap, headers]);

  return (
    <div>
      <h3 className="font-medium text-gray-700 mb-4">Companies by Investability Score</h3>
      <div className="border border-gray-200 rounded-lg overflow-hidden shadow">
        <div className="overflow-auto" style={{ maxHeight: '500px' }}>
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-6 py-3 text-left font-medium text-gray-500 uppercase tracking-wider">Company</th>
                <th className="px-6 py-3 text-center font-medium text-gray-500 uppercase tracking-wider">Founded</th>
                <th className="px-6 py-3 text-center font-medium text-gray-500 uppercase tracking-wider">Employees</th>
                <th className="px-6 py-3 text-center font-medium text-gray-500 uppercase tracking-wider">Website</th>
                <th className="px-6 py-3 text-center font-medium text-gray-500 uppercase tracking-wider">Score</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {topCompanies.length === 0 ? (
                <tr>
                  <td colSpan="5" className="px-6 py-4 text-center text-gray-500">No results yet</td>
                </tr>
              ) : (
                topCompanies.map((company, idx) => (
                  <tr key={idx} className={idx % 2 ? "bg-gray-50" : ""}>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-medium">{company.name}</td>
                    <td className="px-6 py-4 text-center whitespace-nowrap text-gray-700">{company.foundingYear}</td>
                    <td className="px-6 py-4 text-center whitespace-nowrap text-gray-700">{company.employeeCount}</td>
                    <td className="px-6 py-4 text-center whitespace-nowrap text-gray-700">
                      {company.website ? (
                        <a 
                          href={company.website} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 hover:underline"
                        >
                          Visit Site
                        </a>
                      ) : 'N/A'}
                    </td>
                    <td className="px-6 py-4 text-center whitespace-nowrap text-gray-700 font-bold">
                      <span className={`inline-block px-3 py-1 rounded-full ${
                        company.score >= 7 ? "bg-green-100 text-green-800" : 
                        company.score >= 4 ? "bg-yellow-100 text-yellow-800" : 
                        "bg-red-100 text-red-800"
                      }`}>
                        {company.score}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      {topCompanies.length > 0 && (
        <p className="mt-2 text-xs text-gray-500 text-right">
          Showing all {topCompanies.length} results. Scroll to see more.
        </p>
      )}
    </div>
  );
};

  /* ─────────── Render ─────────── */
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="p-6 max-w-6xl mx-auto">
        {/* Header */}
        <header className="bg-white rounded-xl shadow-xl overflow-hidden mb-8">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-8 text-white text-center">
            <h1 className="text-3xl font-bold">SourceScrub Investability Analysis Tool</h1>
            <p className="text-blue-100 mt-2">Analyze companies against your investment criteria. Upload a raw SourceScrub CSV file.</p>
          </div>
        </header>

        {/* Error Display */}
        {lastError && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <p className="font-medium">Error:</p>
            <p>{lastError}</p>
          </div>
        )}

        {/* 1. Upload */}
        <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-blue-50 space-y-4">
          <h2 className="text-xl font-semibold text-blue-800 flex items-center">Upload CSV / TSV File</h2>
          <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-blue-200 rounded-lg cursor-pointer bg-white hover:bg-blue-50 transition text-center">
            <input type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" onChange={handleFileUpload} className="hidden" />
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span className="mt-2 text-base text-gray-600">{file ? file.name : "Click to select a file or drop it here"}</span>
            <span className="text-xs text-gray-500 mt-1">CSV/TSV format with company data</span>
          </label>
        </section>

        {/* 2. Preview */}
        {parsedData.length > 0 && (
          <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white">
            <h2 className="text-xl font-semibold text-blue-800 mb-4">CSV Preview</h2>
            <div className="overflow-x-auto">
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
                        <td key={h} className="px-4 py-3 text-gray-700">
                          <div className="max-w-xs overflow-hidden text-ellipsis whitespace-nowrap">
                            {row[h]}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
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
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {renderMappingSelect("field_1", "Field 1")}
              {renderMappingSelect("field_2", "Field 2")}
              {renderMappingSelect("field_3", "Field 3")}
            </div>
          </section>
        )}

        {/* 4. Criteria & weights */}
        <section className="p-6 mb-6 border border-blue-100 rounded-lg bg-white space-y-6">
          <h2 className="text-xl font-semibold text-blue-800">Define Investing Criteria</h2>
          <textarea
            value={investCriteria}
            onChange={(e) => setInvestCriteria(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                const cursorPosition = e.target.selectionStart;
                const textBeforeCursor = investCriteria.substring(0, cursorPosition);
                const textAfterCursor = investCriteria.substring(cursorPosition);
                
                // Analyze the indentation of existing bullet points
                const lines = textBeforeCursor.split('\n');
                let indent = '';
                
                // Look for bullet points in previous lines to determine indentation
                for (let i = lines.length - 1; i >= 0; i--) {
                  const line = lines[i];
                  if (line.includes('•')) {
                    // Get the indentation (spaces before the bullet)
                    const match = line.match(/^(\s*)[•]/);
                    if (match && match[1]) {
                      indent = match[1];
                      break;
                    }
                  }
                }
                
                // If no indentation found, use 1 space to match original
                if (!indent) {
                  indent = ' ';
                }
                
                // Add a new line with properly indented bullet point
                setInvestCriteria(textBeforeCursor + '\n' + indent + '• ' + textAfterCursor);
                
                // Set cursor position after the bullet point (delayed to ensure state is updated)
                setTimeout(() => {
                  e.target.selectionStart = cursorPosition + 3 + indent.length; // Position after indented "• "
                  e.target.selectionEnd = cursorPosition + 3 + indent.length;
                }, 0);
              }
            }}
            rows={10}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-200"
            placeholder="Enter your investment criteria. Press Enter to add a new bullet point."
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
          {!parsedData.length && file && (
            <p className="text-sm text-red-600 mt-2">No data rows were found in your file</p>
          )}
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
            <button
              onClick={() => {
                if (abortRef.current) {
                  abortRef.current.abort();
                }
              }}
              className="mx-auto block px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700"
            >
              Cancel
            </button>
          </section>
        )}

{/* Add this to your UI, perhaps under the progress */}
{canResume && !isProcessing && (
  <div className="text-center mt-4">
    <p className="text-amber-600 mb-2">Processing timed out due to Vercel's 60-second limit.</p>
    <button
      onClick={() => processData(resumeState.progress)}
      className="px-6 py-3 bg-amber-500 text-white font-medium rounded-lg hover:bg-amber-600 focus:outline-none focus:ring-2 focus:ring-amber-300 shadow-md"
    >
      Resume Processing from Row {resumeState.progress}
    </button>
  </div>
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
      </div>
    </div>
  );
}
