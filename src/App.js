{/* Process Button */}
          <div className="mb-8 flex justify-center">
            <button
              onClick={processData}
              disabled={isProcessing || !parsedData.length}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white text-lg font-medium rounded-xl hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50 shadow-lg transform transition hover:-translate-y-1 flex items-center"
            >
              {isProcessing ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M6.672 1.911a1 1 0 10-1.932.518l.259.966a1 1 0 001.932-.518l-.26-.966zM2.429 4.74a1 1 0 10-.517 1.932l.966.259a1 1 0 00.517-1.932l-.966-.26zm8.814-.569a1 1 0 00-1.415-1.414l-.707.707a1 1 0 101.415 1.415l.707-.708zm-7.071 7.072l.707-.707A1 1 0 003.465 9.12l-.708.707a1 1 0 001.415 1.415zm3.2-5.171a1 1 0 00-1.3 1.3l4 10a1 1 0 001.823.075l1.38-2.759 3.018 3.02a1 1 0 001.414-1.415l-3.019-3.02 2.76-1.379a1 1 0 00-.076-1.822l-10-4z" clipRule="evenodd" />
                  </svg>
                  Analyze Companies
                </>
              )}
            </button>
          </div>
          
          {/* Progress Bar */}
          {isProcessing && (
            <div className="mb-8 p-6 border border-blue-100 rounded-lg bg-white">
              <h2 className="text-xl font-semibold mb-4 text-blue-800 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                </svg>
                Processing Companies
              </h2>
              <div className="relative pt-1">
                <div className="flex mb-2 items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold inline-block py-1 px-2 uppercaseimport React, { useState, useEffect } from 'react';
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="p-6 max-w-6xl mx-auto">
        <div className="bg-white rounded-xl shadow-xl overflow-hidden mb-8">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-8 text-white">
            <h1 className="text-3xl font-bold text-center">VC Investability Analysis Tool</h1>
            <p className="text-center mt-2 text-blue-100 max-w-3xl mx-auto">
              Upload company data, define your investment criteria, and leverage AI to identify the most promising opportunities.
            </p>
          </div>
          
          <div className="p-8">{/* Main content will go here */}
          {/* File Upload Section */}
          <div className="mb-8 p-6 border border-blue-100 rounded-lg bg-blue-50">
            <h2 className="text-xl font-semibold mb-4 text-blue-800 flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              1. Upload CSV File
            </h2>
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col w-full h-32 border-2 border-blue-200 border-dashed hover:bg-blue-50 hover:border-blue-300 rounded-lg cursor-pointer transition-all">
                <div className="flex flex-col items-center justify-center pt-7">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-12 h-12 text-blue-400 group-hover:text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                  </svg>
                  <p className="pt-1 text-sm tracking-wider text-gray-600 group-hover:text-gray-700">
                    {file ? file.name : 'Select a CSV file'}
                  </p>
                </div>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="opacity-0"
                />
              </label>
            </div>
          </div>
      
          {/* Preview Section */}
          {parsedData.length > 0 && (
            <div className="mb-8 p-6 border border-blue-100 rounded-lg bg-white overflow-x-auto">
              <h2 className="text-xl font-semibold mb-4 text-blue-800 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                2. CSV Preview
                <span className="ml-2 text-sm font-normal text-blue-500">(First 5 Rows)</span>
              </h2>
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-100">
                    <tr>
                      {headers.map((header, index) => (
                        <th key={index} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {parsedData.slice(0, 5).map((row, rowIndex) => (
                      <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        {headers.map((header, colIndex) => (
                          <td key={colIndex} className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                            {row[header]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-2 text-xs text-gray-500 text-right">
                Showing 5 of {parsedData.length} rows
              </div>
            </div>
          )}
      
          {/* Column Mapping Section */}
          {headers.length > 0 && (
            <div className="mb-8 p-6 border border-blue-100 rounded-lg bg-white">
              <h2 className="text-xl font-semibold mb-4 text-blue-800 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a2.5 2.5 0 015 0v6a2.5 2.5 0 01-5 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 00-1-1h-2a1 1 0 00-1 1v2z" />
                </svg>
                3. Map Columns
              </h2>
              
              {/* Required Columns */}
              <div className="p-4 bg-blue-50 rounded-lg mb-6">
                <h3 className="font-medium mb-4 text-blue-700 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Required Columns:
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {['employee_count', 'description', 'industries', 'specialties', 'products_services', 'end_markets'].map((field) => (
                    <div key={field} className="mb-2">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        <span className="ml-1 text-red-500">*</span>
                      </label>
                      <select
                        value={columnMappings[field]}
                        onChange={(e) => handleColumnMappingChange(field, e.target.value)}
                        className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
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
              
              {/* Optional Columns */}
              <div className="p-4 bg-green-50 rounded-lg">
                <h3 className="font-medium mb-4 text-green-700 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Optional Columns:
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {['country', 'ownership', 'founding_year'].map((field) => (
                    <div key={field} className="mb-2">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        <span className="ml-1 text-green-500">(Optional)</span>
                      </label>
                      <select
                        value={columnMappings[field]}
                        onChange={(e) => handleColumnMappingChange(field, e.target.value)}
                        className="block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50"
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
            </div>
          )}
      
          {/* Investing Criteria Section */}
          <div className="mb-8 p-6 border border-blue-100 rounded-lg bg-white">
            <h2 className="text-xl font-semibold mb-4 text-blue-800 flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              4. Define Investing Criteria
            </h2>
            <div className="bg-indigo-50 p-4 mb-6 rounded-lg">
              <textarea
                value={investingCriteria}
                onChange={(e) => setInvestingCriteria(e.target.value)}
                rows={6}
                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                placeholder="Enter your investing criteria with bullet points (• symbol)..."
              />
              <p className="mt-2 text-xs text-indigo-600">
                Use bullet points (•) to define each criterion. The system will automatically identify and weight them.
              </p>
            </div>
            
            {/* Criteria Weighting Toggles - Dynamic based on parsed criteria */}
            <div className="mt-6">
              <h3 className="font-medium mb-4 text-indigo-700 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" clipRule="evenodd" />
                </svg>
                Criteria Weighting:
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-white p-4 rounded-lg border border-indigo-100">
                {criteriaItems.map(item => (
                  <div key={item.id} className="bg-white p-4 rounded-lg shadow-sm">
                    <label className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700" title={item.label}>
                        {item.label.length > 40 ? item.label.substring(0, 40) + '...' : item.label}
                      </span>
                      <span className="text-sm font-bold text-indigo-600">{item.weight.toFixed(1)}x</span>
                    </label>
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={item.weight}
                        onChange={(e) => updateCriteriaWeight(item.id, parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer focus:outline-none"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1 px-1">
                        <span>0</span>
                        <span>1</span>
                        <span>2</span>
                      </div>
                    </div>
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
                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase                     rounded-full text-blue-600 bg-blue-200">
                      Progress
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-xs font-semibold inline-block text-blue-600">
                      {resultCount} / {parsedData.length} companies
                    </span>
                  </div>
                </div>
                <div className="overflow-hidden h-3 mb-4 text-xs flex rounded-full bg-blue-100">
                  <div 
                    style={{ width: `${progress}%` }} 
                    className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-500"
                  >
                  </div>
                </div>
                <p className="text-center text-sm text-gray-500">
                  {progress < 100 ? 'Analyzing company data using advanced AI...' : 'Analysis complete!'}
                </p>
              </div>
            </div>
          )}
      
                {/* Results Section */}
          {processedData.length > 0 && !isProcessing && (
            <div className="mb-8 p-6 border border-blue-100 rounded-lg bg-white overflow-x-auto">
              <h2 className="text-xl font-semibold mb-6 text-blue-800 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                5. Analysis Results
              </h2>
              
              {/* Company Score Table */}
              <div className="mb-8">
                <h3 className="font-medium mb-4 text-gray-700">Top Companies by Investability Score:</h3>
                <div className="border border-gray-200 rounded-lg overflow-hidden shadow">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {headers.slice(0, 3).map((header, index) => (
                          <th key={index} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            {header}
                          </th>
                        ))}
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Investability Score
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {processedData
                        .slice()
                        .sort((a, b) => b.investability_score - a.investability_score)
                        .slice(0, 5)
                        .map((row, rowIndex) => (
                          <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                            {headers.slice(0, 3).map((header, colIndex) => (
                              <td key={colIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {row[header]}
                              </td>
                            ))}
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`px-3 py-1 inline-flex text-sm leading-5 font-bold rounded-full ${
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
                </div>
              </div>
              
              {/* Score Distribution */}
              <div className="p-6 bg-gray-50 rounded-lg shadow-inner">
                <h3 className="font-medium mb-4 text-gray-700">Score Distribution:</h3>
                <div className="flex items-end justify-between h-48 space-x-2 px-2">
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(score => {
                    const count = processedData.filter(row => row.investability_score === score).length;
                    const percentage = (count / processedData.length) * 100;
                    return (
                      <div key={score} className="flex flex-col items-center w-full">
                        <div className="text-xs mb-1">{count}</div>
                        <div 
                          className={`w-full rounded-t-md ${
                            score >= 7 ? 'bg-green-500' : 
                            score >= 4 ? 'bg-yellow-500' : 
                            'bg-red-500'
                          }`}} 
                          style={{ height: `${Math.max(5, percentage * 2)}px` }}
                        ></div>
                        <div className="text-xs mt-1 font-medium">{score}</div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-4 flex justify-between text-xs text-gray-500">
                  <span>Poor Match</span>
                  <span>Moderate Match</span>
                  <span>Strong Match</span>
                </div>
              </div>
              
              <div className="mt-8 flex justify-center">
                <button
                  onClick={downloadResults}
                  className="px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 shadow-md flex items-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                  Download Results CSV
                </button>
              </div>
            </div>
          )}
          
          <div className="text-center text-sm text-gray-500 mt-8">
            <p>© {new Date().getFullYear()} VC Investability Analysis Tool - Powered by GPT-4o</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VCAnalysisTool;
