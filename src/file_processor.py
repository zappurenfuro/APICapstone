# -*- coding: utf-8 -*-

import os
import logging
import hashlib
import traceback
import PyPDF2
import docx2txt
import textract

from .text_processor import TextProcessor

class FileProcessor:
    """Class for file processing operations."""
    
    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        """
        Extract text from various document formats (doc, docx, pdf).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as string
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                text = FileProcessor._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc', '.docs']:
                text = FileProcessor._extract_from_doc(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Clean the extracted text
            text = TextProcessor.clean_text(text)
                
            # Debug: Print a hash of the extracted text to verify it's different
            text_hash = hashlib.md5(text.encode()).hexdigest()
            logging.info(f"Extracted text from {file_path} (hash: {text_hash[:8]}...)")
            
            # Debug: Print the first 200 characters of the text
            preview = text[:200].replace('\n', ' ').strip()
            logging.info(f"Text preview: {preview}...")
            
            return text
            
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If PyPDF2 fails to extract meaningful text, try textract as backup
            if not text.strip():
                logging.info(f"PyPDF2 failed to extract text from {file_path}, trying textract...")
                text = textract.process(file_path).decode('utf-8')
                
        except Exception as e:
            logging.error(f"Error in PDF extraction: {str(e)}")
            logging.error(traceback.format_exc())
            # Try textract as a fallback
            try:
                text = textract.process(file_path).decode('utf-8')
            except Exception as e2:
                logging.error(f"Textract also failed: {str(e2)}")
                raise
                
        return text
    
    @staticmethod
    def _extract_from_doc(file_path: str) -> str:
        """Extract text from DOC/DOCX file."""
        try:
            # Try docx2txt first (for .docx)
            text = docx2txt.process(file_path)
        except Exception as e:
            logging.error(f"docx2txt failed: {str(e)}")
            # Fall back to textract (handles .doc and other formats)
            try:
                text = textract.process(file_path).decode('utf-8')
            except Exception as e2:
                logging.error(f"Textract also failed: {str(e2)}")
                raise
        
        return text