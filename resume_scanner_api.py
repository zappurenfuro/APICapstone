import os
import io
import json
import tempfile
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import uuid

# Import your ResumeScanner class
# Assuming the file is named resume_scanner.py or Backup.py
from Backup import ResumeScanner  # Adjust the import based on your actual file name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Scanner API",
    description="API for scanning and matching resumes against a pre-trained model",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to only allow your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class ScanResponse(BaseModel):
    success: bool
    message: str
    job_matches: Optional[List[Dict[str, Any]]] = None
    resume_matches: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    filename: Optional[str] = None
    scan_id: Optional[str] = None

# Global scanner instance
scanner = None

@app.on_event("startup")
async def startup_event():
    """Initialize the scanner on startup."""
    global scanner
    
    # Define folders - adjust these paths for your environment
    input_folder = os.environ.get("INPUT_FOLDER", "input")
    output_folder = os.environ.get("OUTPUT_FOLDER", "output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize scanner
    logging.info("Initializing ResumeScanner...")
    scanner = ResumeScanner(input_folder, output_folder)
    
    # Load data and embeddings
    try:
        # Try to load pre-processed data
        logging.info("Loading pre-processed data...")
        scanner.df = pd.read_csv(os.path.join(output_folder, 'processed_resumes.csv'))
        scanner.load_embeddings()
        logging.info("Successfully loaded pre-processed data and embeddings")
    except Exception as e:
        logging.error(f"Error loading pre-processed data: {str(e)}")
        logging.info("Processing data from scratch...")
        scanner.load_data()
        scanner.create_embeddings()
        logging.info("Successfully processed data and created embeddings")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global scanner
    if scanner:
        scanner.cleanup()
        logging.info("Cleaned up scanner resources")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global scanner
    if scanner and scanner.df is not None and scanner.embeddings is not None:
        return {"status": "healthy", "data_loaded": True, "records": len(scanner.df)}
    else:
        return {"status": "unhealthy", "data_loaded": False}

# Create a modified version of match_text that doesn't save files
def match_text_no_save(scanner, text, top_n=5, file_name=None, match_type="resume"):
    """
    A wrapper around scanner.match_text that prevents saving CSV files.
    """
    # Store the original results_saved state
    original_results_saved = scanner.results_saved.copy() if hasattr(scanner, 'results_saved') and scanner.results_saved else {}
    
    # Set results as already saved to prevent saving files
    scanner.results_saved = {match_type: True}
    
    try:
        # Call the original match_text method
        result = scanner.match_text(text, top_n, file_name, match_type)
        return result
    finally:
        # Restore the original results_saved state
        scanner.results_saved = original_results_saved

@app.post("/scan", response_model=ScanResponse)
async def scan_resume(
    file: UploadFile = File(...),
    top_n: int = Form(5)
):
    """
    Scan a resume and find matching jobs.
    
    Args:
        file: The resume file (PDF, DOCX, DOC)
        top_n: Number of top matches to return
        
    Returns:
        JSON response with matching results
    """
    global scanner
    
    # Check if scanner is initialized
    if not scanner:
        raise HTTPException(status_code=500, detail="Scanner not initialized")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Unsupported file format", "error": f"Only PDF, DOCX, and DOC files are supported. Got {file_ext}"}
        )
    
    try:
        # Create a temporary file to store the uploaded resume
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract text from the resume
        resume_text = scanner.extract_text_from_file(temp_file_path)
        
        # Use our custom match_text function that doesn't save files
        job_matches = match_text_no_save(scanner, resume_text, top_n, file.filename, "job")
        resume_matches = match_text_no_save(scanner, resume_text, top_n, file.filename, "resume")
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Generate a unique scan ID
        scan_id = str(uuid.uuid4())
        
        # Convert DataFrames to lists of dictionaries
        job_matches_list = job_matches.to_dict(orient='records')
        resume_matches_list = resume_matches.to_dict(orient='records')
        
        return {
            "success": True,
            "message": f"Successfully processed resume and found {len(job_matches_list)} job matches and {len(resume_matches_list)} resume matches",
            "job_matches": job_matches_list,
            "resume_matches": resume_matches_list,
            "filename": file.filename,
            "scan_id": scan_id
        }
        
    except Exception as e:
        logging.error(f"Error processing resume: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error processing resume", "error": str(e)}
        )

@app.post("/scan-text", response_model=ScanResponse)
async def scan_resume_text(
    resume_text: str = Form(...),
    top_n: int = Form(5)
):
    """
    Scan resume text directly and find matching jobs.
    
    Args:
        resume_text: The text content of the resume
        top_n: Number of top matches to return
        
    Returns:
        JSON response with matching results
    """
    global scanner
    
    # Check if scanner is initialized
    if not scanner:
        raise HTTPException(status_code=500, detail="Scanner not initialized")
    
    try:
        # Use our custom match_text function that doesn't save files
        job_matches = match_text_no_save(scanner, resume_text, top_n, None, "job")
        
        # Generate a unique scan ID
        scan_id = str(uuid.uuid4())
        
        # Convert job matches to list of dictionaries
        job_matches_list = job_matches.to_dict(orient='records')
        
        return {
            "success": True,
            "message": f"Successfully processed resume text and found {len(job_matches_list)} matches",
            "job_matches": job_matches_list,
            "resume_matches": [],
            "scan_id": scan_id
        }
        
    except Exception as e:
        logging.error(f"Error processing resume text: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error processing resume text", "error": str(e)}
        )

# Run the API server
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Run the server
    uvicorn.run("resume_scanner_api:app", host="0.0.0.0", port=port, reload=True)