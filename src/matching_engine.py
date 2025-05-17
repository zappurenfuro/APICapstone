# -*- coding: utf-8 -*-

import os
import time
import logging
import traceback
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict

from .text_processor import TextProcessor
from .file_processor import FileProcessor

class MatchingEngine:
    """Class for matching operations."""
    
    def __init__(self, embedding_manager, data_processor, output_folder):
        """Initialize the MatchingEngine."""
        self.embedding_manager = embedding_manager
        self.data_processor = data_processor
        self.output_folder = output_folder
    
    def match_text(self, text: str, top_n: int = 5, file_name: str = None, match_type: str = "resume") -> pd.DataFrame:
        """
        Unified matching function for all matching operations with memory optimization.
        
        Args:
            text: Text to match
            top_n: Number of top matches to return
            file_name: Optional name of the file for output naming
            match_type: Type of match (for logging and output naming)
            
        Returns:
            DataFrame with top matches and similarity scores
        """
        logging.info(f"Matching text against dataset ({match_type})...")
        
        # Clean the input text
        text = TextProcessor.clean_text(text)
        
        # Debug: Print a hash of the input text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        logging.info(f"Matching text (hash: {text_hash[:8]}...)")
        
        # Check if embeddings exist
        if self.embedding_manager.embeddings is None:
            logging.warning("Embeddings not found. Loading or creating embeddings...")
            loaded_embeddings = self.embedding_manager.load_embeddings()
            if loaded_embeddings is None:
                if self.data_processor.df is not None:
                    self.embedding_manager.create_embeddings(self.data_processor.df)
                else:
                    raise ValueError("No data available for creating embeddings.")
        
        # Encode the text
        user_embedding = self.embedding_manager.encode_text(text)
        
        # If we're using memory mapping, process in chunks
        if hasattr(self.embedding_manager, 'embeddings_file') and self.embedding_manager.embeddings_file:
            # Load embeddings in chunks to avoid loading everything into RAM
            embeddings_mmap = np.load(self.embedding_manager.embeddings_file, mmap_mode='r')
            
            # Process in chunks
            chunk_size = 10000  # Adjust based on your RAM
            total_samples = embeddings_mmap.shape[0]
            
            # Initialize array to store all similarities
            all_similarities = np.zeros(total_samples)
            
            for i in tqdm(range(0, total_samples, chunk_size), desc="Computing similarities"):
                end_idx = min(i + chunk_size, total_samples)
                
                # Load chunk into RAM
                embeddings_chunk = embeddings_mmap[i:end_idx]
                
                # Calculate similarities for this chunk
                chunk_similarities = cosine_similarity(user_embedding, embeddings_chunk)[0]
                
                # Store in the full array
                all_similarities[i:end_idx] = chunk_similarities
            
            # Get top matches
            top_indices = all_similarities.argsort()[-top_n:][::-1]
            top_similarities = all_similarities[top_indices]
            
        else:
            # If embeddings are already in memory (GPU or CPU), use them directly
            # Move embeddings to CPU if they're on GPU
            if self.embedding_manager.embeddings.is_cuda:
                embeddings_cpu = self.embedding_manager.embeddings.detach().cpu().numpy()
            else:
                embeddings_cpu = self.embedding_manager.embeddings.detach().numpy()
            
            # Calculate cosine similarity
            all_similarities = cosine_similarity(user_embedding, embeddings_cpu)[0]
            
            # Get top matches
            top_indices = all_similarities.argsort()[-top_n:][::-1]
            top_similarities = all_similarities[top_indices]
        
        # Get the corresponding rows from the dataframe
        top_matches = self.data_processor.df.iloc[top_indices].copy()
        top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
        
        # Apply title case formatting to ensure consistent display
        top_matches['title'] = top_matches['title'].apply(lambda x: TextProcessor.format_title(x))
        
        # Save results with unique filename if provided
        if file_name:
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            output_file = f"{match_type}_matches_{base_name}_{int(time.time())}.csv"
            top_matches.to_csv(os.path.join(self.output_folder, output_file), index=False)
            logging.info(f"Found {len(top_matches)} {match_type} matches and saved to {output_file}")
        
        return top_matches
    
    def process_resume_file(self, file_path: str, top_n: int = 5) -> Dict:
        """
        Process a resume file and match it against the database.
        
        Args:
            file_path: Path to the resume file
            top_n: Number of top matches to return
            
        Returns:
            Dictionary with results
        """
        try:
            # Extract text from file
            resume_text = FileProcessor.extract_text_from_file(file_path)
            logging.info(f"Extracted {len(resume_text)} characters from {file_path}")
            
            # Match against database
            top_matches = self.match_text(resume_text, top_n, file_path, "resume")
            
            return {
                'resume_text': resume_text,
                'top_matches': top_matches
            }
        except Exception as e:
            logging.error(f"Error processing resume file {file_path}: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                'error': str(e),
                'file_path': file_path
            }
    
    def match_resume_to_jobs(self, resume_text: str, top_n: int = 5, file_name: str = None) -> pd.DataFrame:
        """
        Match a resume against job titles in the dataset.
        This uses the same matching process as regular resume matching,
        but filters for entries with job-related titles.
        
        Args:
            resume_text: Text of the resume to match
            top_n: Number of top matches to return
            file_name: Optional name of the resume file for output naming
            
        Returns:
            DataFrame with top job matches and similarity scores
        """
        try:
            if self.data_processor.df is None:
                raise ValueError("Dataset not loaded. Call load_data() first.")
            
            # Clean the resume text
            resume_text = TextProcessor.clean_text(resume_text)
            
            # Get all matches
            job_matches = self.match_text(resume_text, top_n, file_name, "job")
            
            # Print results
            print(f"\n🏆 Top Job Matches for {file_name if file_name else 'resume'}:")
            for idx, row in job_matches.iterrows():
                print(f"\n🔹 Title: {row['title']}")
                print(f"   Similarity: {row['similarity_percentage']}%")
                # Use embedding_text instead of text_representation
                print(f"   Description: {row['embedding_text'][:100]}...")
            
            return job_matches
        except Exception as e:
            logging.error(f"Error matching resume to jobs: {str(e)}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()