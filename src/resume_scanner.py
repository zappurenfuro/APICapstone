# -*- coding: utf-8 -*-

import os
from typing import Dict
import pandas as pd

from .system_manager import SystemManager
from .data_processor import DataProcessor
from .embedding_manager import EmbeddingManager
from .matching_engine import MatchingEngine

class ResumeScanner:
    """Main class for resume scanning and matching functionality (Facade pattern)."""
    
    def __init__(self, input_folder: str, output_folder: str):
        """Initialize the ResumeScanner with input and output folders."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Set up system resources
        SystemManager.log_system_resources()
        self.device, self.use_mixed_precision = SystemManager.setup_device()
        self.temp_dir, self.temp_path = SystemManager.setup_ram_disk()
        
        # Initialize components
        self.data_processor = DataProcessor(input_folder, output_folder)
        self.embedding_manager = EmbeddingManager('BAAI/bge-large-en-v1.5', self.device, self.use_mixed_precision, output_folder)
        self.matching_engine = MatchingEngine(self.embedding_manager, self.data_processor, output_folder)
    
    def load_data(self):
        """Load and process the resume datasets."""
        return self.data_processor.load_data()
    
    def create_embeddings(self, batch_size=32):
        """Create embeddings for the loaded data."""
        return self.embedding_manager.create_embeddings(self.data_processor.df, batch_size)
    
    def load_embeddings(self, file_path=None):
        """Load pre-computed embeddings from file."""
        return self.embedding_manager.load_embeddings(file_path)
    
    def process_resume_file(self, file_path: str, top_n: int = 5) -> Dict:
        """Process a resume file and match it against the database."""
        return self.matching_engine.process_resume_file(file_path, top_n)
    
    def match_resume_to_jobs(self, resume_text: str, top_n: int = 5, file_name: str = None) -> pd.DataFrame:
        """Match a resume against job titles in the dataset."""
        return self.matching_engine.match_resume_to_jobs(resume_text, top_n, file_name)
    
    def cleanup(self):
        """Clean up resources."""
        SystemManager.cleanup_resources(self.temp_dir)