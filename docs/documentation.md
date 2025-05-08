# Company Investability Score - Developer Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Core Functionality](#core-functionality)
6. [API Reference](#api-reference)
7. [Development Workflow](#development-workflow)
8. [Troubleshooting](#troubleshooting)
9. [Deployment](#deployment)
10. [Future Improvements](#future-improvements)

## Project Overview

The Company Investability Score tool is a web application designed to analyze companies from a CSV/TSV file (typically exported from SourceScrub) against investment criteria to determine their investment potential. The application:

1. Uploads and parses a CSV file containing company data
2. Maps the CSV columns to required fields
3. Allows users to customize investment criteria with weight adjustments
4. Analyzes companies against the criteria using the OpenAI API
5. Provides a downloadable report with investment scores

The application contains two main components:
- **Frontend**: A React application that handles the user interface and data visualization
- **Backend**: A Python Flask API that processes the CSV data and communicates with OpenAI for analysis

## Architecture

### Project Structure

```
/
├── api/                    # Python backend files
│   ├── analyze.py          # Main API logic for analyzing companies
│   └── requirements.txt    # Python dependencies
├── public/                 # Static frontend files
│   └── index.html          # Main HTML file
├── src/                    # React frontend code
│   ├── App.js              # Main React component
│   ├── App.css             # Main stylesheet
│   ├── index.js            # React entry point
│   └── index.css           # Global styles (Tailwind imports)
├── package.json            # Frontend dependencies
├── tailwind.config.js      # Tailwind CSS config
├── postcss.config.js       # PostCSS config
└── vercel.json             # Vercel deployment config
```

### Technology Stack

- **Frontend**:
  - React 18.2.0
  - Tailwind CSS 3.4.1
  - Recharts 2.5.0 (for data visualization)
  - PapaParse 5.3.2 (for CSV parsing)

- **Backend**:
  - Python 3.12+
  - Flask 3.1.0
  - Flask-CORS 4.0.0
  - OpenAI 1.23.0
  - Pandas 2.2.3
  - NumPy 2.2.5

- **Deployment**:
  - Vercel (configured via vercel.json)

## Backend Setup

### Prerequisites

- Python 3.12 or newer
- OpenAI API key

### Installation

1. Navigate to the project directory
2. Install the Python dependencies:

```bash
pip install -r api/requirements.txt
```

3. Set up the OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Configuration

The backend uses the GPT-4o model from OpenAI with the following configuration:

```python
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 1500
TEMPERATURE = 0.2
BATCH_SIZE = 2  # rows per OpenAI request
TIMEOUT = 30  # seconds
```

These settings can be adjusted in the `api/analyze.py` file.

## Frontend Setup

### Prerequisites

- Node.js 14 or newer
- npm or yarn

### Installation

1. Navigate to the project directory
2. Install the JavaScript dependencies:

```bash
npm install
# or
yarn install
```

### Development Server

Start the development server:

```bash
npm start
# or
yarn start
```

The application will be available at http://localhost:3000.

## Core Functionality

### 1. CSV Upload and Parsing

The application accepts CSV/TSV files (typically from SourceScrub) and parses them using PapaParse. It automatically detects the header row (usually at row 5) and maps columns to required fields. The frontend handles the initial parsing and preview.

Key functions:
- `handleFileUpload` in `App.js` - Processes the uploaded file
- `normalizeCSV` in `analyze.py` - Ensures consistent CSV formatting on the backend

### 2. Column Mapping

The user must map CSV columns to required fields:

**Required Fields:**
- Employee count
- Company description
- Industries
- Specialties
- Products/services
- End markets
- Country
- Ownership type
- Total raised
- Latest raised
- Date of most recent investment

**Optional Fields:**
- Three customizable additional fields

The application attempts to automatically map fields based on common column names but allows manual adjustment.

### 3. Investment Criteria

The application has built-in investment criteria based on Carrick Capital's investment thesis:
- Enterprise B2B technology companies
- SaaS or recurring revenue models
- Specific industries (cybersecurity, data analytics, etc.)
- Employee count (80-350 recommended)
- Geographic focus (primarily US)
- Funding range ($5M to $150M)
- Technology focus (AI/ML, cloud, etc.)

Users can define additional criteria with bullet points and adjust weights for each criterion.

### 4. Analysis Process

When the user clicks "Analyze Companies," the application:

1. Sends the CSV data, column mappings, and criteria to the backend
2. The backend processes the data in batches (2 companies per batch)
3. Each batch is analyzed using the OpenAI API with the company details and investment criteria
4. The results are streamed back to the frontend as they become available
5. The frontend updates the UI in real-time showing progress and results

The backend uses the GPT-4o model to score each company on a scale of 0-10, with specific guidelines influencing the scores (e.g., companies with <70 employees capped at 6, >650 employees capped at 4).

### 5. Results and Export

The results are displayed in two formats:
- A sortable table showing companies with their scores
- A histogram showing the distribution of scores

Users can download a reorganized CSV file containing:
- Original company data
- Investability scores
- Additional formatted columns for better analysis

## API Reference

### `/api/analyze` (POST)

Analyzes companies against investment criteria.

**Request Body (FormData):**
- `file`: CSV/TSV file containing company data
- `columnMap`: JSON string mapping column names to required fields
- `criteria`: String containing additional investment criteria
- `weights`: (Optional) JSON string with weights for criteria
- `resumeFrom`: (Optional) Row number to resume analysis from

**Response:**
NDJSON stream with the following types of updates:
- Progress updates: `{ "progress": number, "total_rows": number }`
- Result batches: `{ "result": [{ "company_name": string, "investability_score": number }] }`
- Status updates: `{ "status": string }` (starting, processing_chunk, chunk_complete, timeout, etc.)
- Error messages: `{ "error": string }`

### `/api/health` (GET)

Simple health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "time": 1683123456.789,
  "openai_api_key_present": true
}
```

## Development Workflow

### Modifying the Backend

1. Modify the code in `api/analyze.py`
2. Update the dependencies in `api/requirements.txt` if needed
3. Test locally by running:
   ```bash
   python api/analyze.py
   ```

### Modifying the Frontend

1. Modify the code in `src/App.js` or other frontend files
2. Update UI components, styles, or logic as needed
3. Test locally using:
   ```bash
   npm start
   ```

### Key Files for Common Changes

- **Column Mapping Logic**: `src/App.js` (handleFileUpload function)
- **Investment Criteria**: `api/analyze.py` (build_system_prompt function)
- **OpenAI Configuration**: `api/analyze.py` (near the beginning)
- **CSV Processing**: `api/analyze.py` (normalize_csv function)
- **Result Formatting**: `src/App.js` (downloadCSV function)

## Troubleshooting

### Common Issues

#### CSV Parsing Issues

If users experience issues with CSV parsing:

1. Check that the file is properly formatted
2. Look for unusual delimiters or encoding issues
3. Adjust the `normalize_csv` function in `analyze.py` if needed
4. Check the header row detection logic in both frontend and backend

#### OpenAI API Errors

If OpenAI API calls fail:

1. Verify the API key is set correctly
2. Check for rate limiting or quota issues
3. Try reducing `BATCH_SIZE` in `analyze.py`
4. Increase `TIMEOUT` if requests are timing out

#### Processing Timeout

The application has a built-in timeout mechanism for Vercel's 60-second limit. If processing times out:

1. The application will store progress and allow resuming
2. Reduce `BATCH_SIZE` in `analyze.py`
3. Consider processing fewer companies at once

### Debugging

- Frontend logs: Check the browser console
- Backend logs: Check Vercel function logs or local terminal output
- API responses: Monitor network requests in browser dev tools

## Deployment

The application is configured for deployment on Vercel with the following setup:

### Vercel Configuration

The `vercel.json` file configures:
- API routes forwarded to the Python backend
- Frontend static file serving

### Environment Variables

The following environment variables need to be set in Vercel:
- `OPENAI_API_KEY`: Your OpenAI API key

### Deployment Steps

1. Connect the GitHub repository to Vercel
2. Configure the environment variables
3. Deploy the application
4. Verify the API health check endpoint is working

## Future Improvements

Potential areas for enhancement:

1. **Performance Optimization**:
   - Implement caching for repeated analysis
   - Optimize CSV processing for larger files
   - Add worker threads for parallel processing

2. **User Experience**:
   - Add user accounts and saved criteria
   - Implement result filtering and sorting
   - Add data visualization dashboards

3. **Analysis Enhancements**:
   - Support for custom scoring algorithms
   - Integration with additional data sources
   - Comparative analysis against portfolios

4. **Technical Improvements**:
   - Add comprehensive test suite
   - Implement type checking with TypeScript
   - Add database storage for results

---

This documentation provides a comprehensive overview of the Company Investability Score tool. For any questions or issues, please contact the project maintainer.
