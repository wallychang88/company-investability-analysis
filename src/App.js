import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import _ from 'lodash';

const VCAnalysisTool = () => {
  const [file, setFile] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [columnMappings, setColumnMappings] = useState({
    employee_count: '',
    description: '',
    industries: '',
    specialties: '',
    products_services: '',
    end_markets: '',
    country: '',
    ownership: '',
    founding_year: ''
  });
  const [investingCriteria, setInvestingCriteria] = useState(`We invest in enterprise SaaS and managed services companies that:
• have 80–300 employees
• provide a product or service that supports AI/HPC infrastructure and/or is an enabler of AI/HPC environment buildout
• are not overhyped application layer LLM SaaS products
• and have a clear, defensible moat (e.g., proprietary data or network effects)`);

  // Parse criteria text to extract criteria items
  const [criteriaItems, setCriteriaItems] = useState([
    { id: 'employee_size', label: 'Employee Size (80-300)', weight: 1.0 },
    { id: 'ai_hpc_focus', label: 'AI/HPC Infrastructure Focus', weight: 1.0 },
    { id: 'not_overhyped', label: 'Not Overhyped LLM Products', weight: 1.0 },
    { id: 'defensible_moat', label: 'Defensible Moat', weight: 1.0 }
  ]);

  // Effect to parse criteria text
  useEffect(() => {
    const parseCriteriaText = () => {
      const lines = investingCriteria.split('\n');
      const bulletItems = lines
        .filter(line => line.trim().startsWith('•'))
        .map(line => line.trim().substring(1).trim());
      
      if (bulletItems.length === 0) return;
      
      const newCriteriaItems = [];
      
      // Try to extract criteria items
      bulletItems.forEach((item, index) => {
        let id, label;
        
        if (item.toLowerCase().includes('employee')) {
          id = 'employee_size';
          label = item;
        } else if (item.toLowerCase().includes('ai') || item.toLowerCase().includes('hpc') || item.toLowerCase().includes('infrastructure')) {
          id = 'ai_hpc_focus';
          label = item;
        } else if (item.toLowerCase().includes('overhyped') || item.toLowerCase().includes('llm')) {
          id = 'not_overhyped';
          label = item;
        } else if (item.toLowerCase().includes('moat') || item.toLowerCase().includes('defensible')) {
          id = 'defensible_moat';
          label = item;
        } else {
          // Create a new criteria
          id = `criteria_${index}`;
          label = item;
        }
        
        // Find existing weight or use default
        const existingItem = criteriaItems.find(c => c.id === id);
        const weight = existingItem ? existingItem.weight : 1.0;
        
        newCriteriaItems.push({ id, label, weight });
      });
      
      // Maintain weights from existing criteria when updating
      if (newCriteriaItems.length > 0) {
        setCriteriaItems(newCriteriaItems.map(item => {
          const existingItem = criteriaItems.find(c => c.id === item.id);
          return { 
            ...item, 
            weight: existingItem ? existingItem.weight : item.weight 
          };
        }));
      }
    };
    
    parseCriteriaText();
  }, [investingCriteria]);
  
  // Update weights function
  const updateCriteriaWeight = (id, weight) => {
    setCriteriaItems(
      criteriaItems.map(item => 
        item.id === id ? { ...item, weight } : item
      )
    );
  };
  const [processedData, setProcessedData] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [resultCount, setResultCount] = useState(0);

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
    
    if (uploadedFile) {
      // First read raw file to detect header row (row 3 - index 2 in 0-based indexing)
      Papa.parse(uploadedFile, {
        header: false,
        preview: 10, // Read first 10 rows to ensure we get the header
        skipEmptyLines: true,
        complete: function(initialResults) {
          // Header is at row 3 (index 2 in 0-based indexing)
          const headerRow = initialResults.data[2];
          
          // Now parse again with the proper header row
          Papa.parse(uploadedFile, {
            header: false,
            skipEmptyLines: true,
            complete: function(fullResults) {
              const headers = fullResults.data[2]; // Get headers from row 3
              const data = fullResults.data.slice(3).map(row => {
                const rowObj = {};
                headers.forEach((header, i) => {
                  if (header) rowObj[header] = row[i];
                });
                return rowObj;
              });
              
              setParsedData(data);
              setHeaders(headers.filter(h => h)); // Filter out empty headers
              
              // Try to auto-map column headers
              const headerMap = {};
              const headerLowerCase = headers.map(h => h ? h.toLowerCase() : '');
              
              if (headerLowerCase.includes('employee count')) {
                headerMap.employee_count = headers[headerLowerCase.indexOf('employee count')];
              }
              if (headerLowerCase.includes('description')) {
                headerMap.description = headers[headerLowerCase.indexOf('description')];
              }
              if (headerLowerCase.includes('industries')) {
                headerMap.industries = headers[headerLowerCase.indexOf('industries')];
              }
              if (headerLowerCase.includes('specialties')) {
                headerMap.specialties = headers[headerLowerCase.indexOf('specialties')];
              }
              if (headerLowerCase.includes('products and services') || headerLowerCase.includes('products & services')) {
                const index = headerLowerCase.includes('products and services') 
                  ? headerLowerCase.indexOf('products and services')
                  : headerLowerCase.indexOf('products & services');
                headerMap.products_services = headers[index];
              }
              if (headerLowerCase.includes('end markets')) {
                headerMap.end_markets = headers[headerLowerCase.indexOf('end markets')];
              }
              if (headerLowerCase.includes('country')) {
                headerMap.country = headers[headerLowerCase.indexOf('country')];
              }
              if (headerLowerCase.includes('ownership')) {
                headerMap.ownership = headers[headerLowerCase.indexOf('ownership')];
              }
              if (headerLowerCase.includes('founding year') || headerLowerCase.includes('founded')) {
                const index = headerLowerCase.includes('founding year') 
                  ? headerLowerCase.indexOf('founding year')
                  : headerLowerCase.indexOf('founded');
                headerMap.founding_year = headers[index];
              }
              
              setColumnMappings(prevMappings => ({
                ...prevMappings,
                ...headerMap
              }));
            }
          });
        }
      });
    }
  };

  const handleColumnMappingChange = (field, value) => {
    setColumnMappings({
      ...columnMappings,
      [field]: value
    });
  };

  // Remove the simulateAPIScoring function since we're using the real API now
  // const simulateAPIScoring = (row) => { ... };

  const processData = async () => {
    if (!parsedData.length) {
      alert('Please upload a file before processing');
      return;
    }
    
    // Check if required columns are mapped
    const requiredColumns = ['employee_count', 'description', 'industries', 'specialties', 'products_services', 'end_markets'];
    const missingColumns = requiredColumns.filter(col => !columnMappings[col]);
    
    if (missingColumns.length > 0) {
      alert(`Please map the required columns: ${missingColumns.join(', ')}`);
      return;
    }
    
    setIsProcessing(true);
    setProgress(0);
    setResultCount(0);
    
    try {
      // Convert data to CSV string to send to backend
      const csvString = Papa.unparse(parsedData);
      
      // Use relative path for API URL when deployed on Vercel
      const API_URL = '/api/analyze';
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          csv_data: csvString,
          column_mappings: columnMappings,
          investing_criteria: investingCriteria,
          criteria_weights: criteriaItems
        }),
      });
      
      if (!response.ok) {
        throw new Error('API request failed');
      }
      
      const responseData = await response.json();
      
      if (responseData.success) {
        setProcessedData(responseData.results);
        // Set progress to 100% when complete
        setResultCount(parsedData.length);
        setProgress(100);
      } else {
        throw new Error(responseData.error || 'Unknown error occurred');
      }
      
      setIsProcessing(false);
    } catch (error) {
      console.error('Error processing data:', error);
      alert('An error occurred while processing the data. Please try again.');
      setIsProcessing(false);
    }
  };
  };

    setIsProcessing(true);
    setProgress(0);
    setResultCount(0);
    
    const results = [];
    const total = parsedData.length;
    
    for (let i = 0; i < parsedData.length; i++) {
      const row = parsedData[i];
      const score = await simulateAPIScoring(row);
      
      const processedRow = {
        ...row,
        investability_score: score
      };
      
      results.push(processedRow);
      setResultCount(i + 1);
      setProgress(Math.round(((i + 1) / total) * 100));
    }
    
    setProcessedData(results);
    setIsProcessing(false);
  } catch (error) {
    console.error('Error processing data:', error);
    alert('An error occurred while processing the data. Please try again.');
    setIsProcessing(false);
  }
  };

  const downloadResults = () => {
    if (!processedData.length) return;
    
    const csv = Papa.unparse(processedData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'investability_analysis.csv');
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto bg-white rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-6 text-center">VC Investability Analysis Tool</h1>
      
      {/* File Upload Section */}
      <div className="mb-8 p-4 border rounded-lg bg-gray-50">
        <h2 className="text-xl font-semibold mb-4">1. Upload CSV File</h2>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          className="block w-full text-sm text-gray-500 
            file:mr-4 file:py-2 file:px-4
            file:rounded-md file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>
      
      {/* Preview Section */}
      {parsedData.length > 0 && (
        <div className="mb-8 p-4 border rounded-lg bg-gray-50 overflow-x-auto">
          <h2 className="text-xl font-semibold mb-4">2. CSV Preview (First 5 Rows)</h2>
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                {headers.map((header, index) => (
                  <th key={index} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {parsedData.slice(0, 5).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.map((header, colIndex) => (
                    <td key={colIndex} className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">
                      {row[header]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {/* Column Mapping Section */}
      {headers.length > 0 && (
        <div className="mb-8 p-4 border rounded-lg bg-gray-50">
          <h2 className="text-xl font-semibold mb-4">3. Map Columns</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Required Columns */}
            <div className="col-span-2">
              <h3 className="font-medium mb-3 text-blue-700">Required Columns:</h3>
            </div>
            {['employee_count', 'description', 'industries', 'specialties', 'products_services', 'end_markets'].map((field) => (
              <div key={field} className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}*
                </label>
                <select
                  value={columnMappings[field]}
                  onChange={(e) => handleColumnMappingChange(field, e.target.value)}
                  className="block w-full mt-1 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                >
                  <option value="">Select column</option>
                  {headers.map((header) => (
                    <option key={header} value={header}>
                      {header}
                    </option>
                  ))}
                </select>
              </div>
            ))}
            
            {/* Optional Columns */}
            <div className="col-span-2 mt-4">
              <h3 className="font-medium mb-3 text-green-700">Optional Columns:</h3>
            </div>
            {['country', 'ownership', 'founding_year'].map((field) => (
              <div key={field} className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} (Optional)
                </label>
                <select
                  value={columnMappings[field]}
                  onChange={(e) => handleColumnMappingChange(field, e.target.value)}
                  className="block w-full mt-1 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                >
                  <option value="">Select column</option>
                  {headers.map((header) => (
                    <option key={header} value={header}>
                      {header}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Investing Criteria Section */}
      <div className="mb-8 p-4 border rounded-lg bg-gray-50">
        <h2 className="text-xl font-semibold mb-4">4. Define Investing Criteria</h2>
        <textarea
          value={investingCriteria}
          onChange={(e) => setInvestingCriteria(e.target.value)}
          rows={6}
          className="block w-full mt-1 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
          placeholder="Enter your investing criteria with bullet points (• symbol)..."
        />
        
        {/* Criteria Weighting Toggles - Dynamic based on parsed criteria */}
        <div className="mt-6">
          <h3 className="font-medium mb-3">Criteria Weighting:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {criteriaItems.map(item => (
              <div key={item.id}>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700" title={item.label}>
                    {item.label.length > 40 ? item.label.substring(0, 40) + '...' : item.label}
                  </span>
                  <span className="text-sm font-medium">{item.weight.toFixed(1)}x</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={item.weight}
                  onChange={(e) => updateCriteriaWeight(item.id, parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Process Button */}
      <div className="mb-8 flex justify-center">
        <button
          onClick={processData}
          disabled={isProcessing || !parsedData.length}
          className="px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
        >
          {isProcessing ? 'Processing...' : 'Analyze Companies'}
        </button>
      </div>
      
      {/* Progress Bar */}
      {isProcessing && (
        <div className="mb-8 p-4 border rounded-lg bg-gray-50">
          <h2 className="text-xl font-semibold mb-4">Processing Companies</h2>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                  Progress
                </span>
              </div>
              <div className="text-right">
                <span className="text-xs font-semibold inline-block text-blue-600">
                  {resultCount} / {parsedData.length} companies
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
              <div style={{ width: `${progress}%` }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"></div>
            </div>
          </div>
        </div>
      )}
      
      {/* Results Section */}
      {processedData.length > 0 && !isProcessing && (
        <div className="mb-8 p-4 border rounded-lg bg-gray-50 overflow-x-auto">
          <h2 className="text-xl font-semibold mb-4">5. Results Preview (First 5 Companies)</h2>
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                {headers.slice(0, 3).map((header, index) => (
                  <th key={index} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {header}
                  </th>
                ))}
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Investability Score
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {processedData.slice(0, 5).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.slice(0, 3).map((header, colIndex) => (
                    <td key={colIndex} className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">
                      {row[header]}
                    </td>
                  ))}
                  <td className="px-4 py-2 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      row.investability_score >= 7 ? 'bg-green-100 text-green-800' : 
                      row.investability_score >= 4 ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-red-100 text-red-800'
                    }`}>
                      {row.investability_score}/10
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {/* Score Distribution */}
          <div className="mt-6">
            <h3 className="font-medium mb-2">Score Distribution:</h3>
            <div className="flex items-center space-x-1">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(score => {
                const count = processedData.filter(row => row.investability_score === score).length;
                const percentage = (count / processedData.length) * 100;
                return (
                  <div key={score} className="flex flex-col items-center">
                    <div className="text-xs mb-1">{count}</div>
                    <div 
                      className={`w-6 ${
                        score >= 7 ? 'bg-green-500' : 
                        score >= 4 ? 'bg-yellow-500' : 
                        'bg-red-500'
                      }`} 
                      style={{ height: `${Math.max(4, percentage * 2)}px` }}
                    ></div>
                    <div className="text-xs mt-1">{score}</div>
                  </div>
                );
              })}
            </div>
          </div>
          
          <div className="mt-6 flex justify-center">
            <button
              onClick={downloadResults}
              className="px-6 py-2 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
            >
              Download Results CSV
            </button>
          </div>
        </div>
      )}
      
      <div className="text-center text-sm text-gray-500 mt-8">
        <p>Note: This is a frontend simulation. In production, this would connect to your actual API for scoring companies using GPT-4o.</p>
        <p className="mt-2">The backend would transform the weighted criteria into parameters for the LLM scoring system.</p>
      </div>
    </div>
  );
};

export default VCAnalysisTool;
