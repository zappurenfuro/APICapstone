# -*- coding: utf-8 -*-

import os
import gc
import logging
import traceback
import pandas as pd
import psutil

from .text_processor import TextProcessor

class DataProcessor:
    """Class for data processing operations."""
    
    def __init__(self, input_folder: str, output_folder: str):
        """Initialize the DataProcessor with input and output folders."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.df = None
    
    def load_data(self):
        """Load and process the resume datasets with memory optimization."""
        logging.info("Loading datasets with memory optimization...")
        
        # Define optimized dtypes to reduce memory usage
        dtypes = {
            'person_id': 'int32',
            'ability': 'category',
            'title': 'category',
            'skill': 'category'
        }
        
        # Determine optimal chunk size based on available RAM
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        chunk_size = min(100000, max(10000, int(available_ram_gb * 20000)))  # Heuristic
        logging.info(f"Using chunk size of {chunk_size} based on {available_ram_gb:.2f} GB available RAM")
        
        try:
            # Load CSV files with chunking to reduce memory usage
            logging.info("Loading CSV files in chunks...")
            
            # Process each file in chunks
            df1_chunks = pd.read_csv(os.path.join(self.input_folder, '01_people.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df1 = pd.concat(list(df1_chunks))
            
            df2_chunks = pd.read_csv(os.path.join(self.input_folder, '02_abilities.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df2 = pd.concat(list(df2_chunks))
            
            df3_chunks = pd.read_csv(os.path.join(self.input_folder, '03_education.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df3 = pd.concat(list(df3_chunks))
            
            df4_chunks = pd.read_csv(os.path.join(self.input_folder, '04_experience.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df4 = pd.concat(list(df4_chunks))
            
            df5_chunks = pd.read_csv(os.path.join(self.input_folder, '05_person_skills.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df5 = pd.concat(list(df5_chunks))
            
            # Clean text in all dataframes
            logging.info("Cleaning text in all dataframes...")
            for df in [df1, df2, df3, df4, df5]:
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    if col != 'person_id':  # Skip ID columns
                        df[col] = df[col].apply(TextProcessor.clean_text)
            
            # Filter and clean data
            logging.info("Filtering and cleaning data...")
            df1 = self._filter_person(df1).drop(columns=['name', 'email', 'phone', 'linkedin'], errors='ignore')
            df2 = self._filter_person(df2)
            df3 = self._filter_person(df3).drop(columns=['institution', 'start_date', 'location'], errors='ignore')
            df4 = self._filter_person(df4).drop(columns=['firm', 'start_date', 'end_date', 'location'], errors='ignore')
            df5 = self._filter_person(df5)
            
            # Process title column to keep only the first title if multiple exist
            logging.info("Processing title column to keep only the first title...")
            if 'title' in df4.columns:
                df4['title'] = df4['title'].apply(lambda x: str(x).split(';')[0].strip() if pd.notna(x) else x)
            
            # Aggregate text by person
            logging.info("Aggregating text by person...")
            df2_agg = self._aggregate_text(df2)
            df3_agg = self._aggregate_text(df3)
            df4_agg = self._aggregate_text(df4)
            df5_agg = self._aggregate_text(df5)
            
            # Free memory after each step
            del df2
            gc.collect()
            
            # Merge dataframes incrementally to save memory
            logging.info("Merging dataframes...")
            self.df = df1.merge(df2_agg, on='person_id', how='left')
            del df1, df2_agg
            gc.collect()
            
            self.df = self.df.merge(df3_agg, on='person_id', how='left')
            del df3_agg
            gc.collect()
            
            self.df = self.df.merge(df4_agg, on='person_id', how='left')
            del df4_agg
            gc.collect()
            
            self.df = self.df.merge(df5_agg, on='person_id', how='left')
            del df5_agg, df3, df4, df5
            gc.collect()
            
            # Remove duplicate rows after merging, excluding person_id from the check
            logging.info("Removing duplicate rows based on content (excluding person_id)...")
            initial_rows = len(self.df)
            
            # Get all columns except person_id for duplicate checking
            content_columns = [col for col in self.df.columns if col != 'person_id']
            
            # Keep first occurrence of each unique content combination
            self.df = self.df.drop_duplicates(subset=content_columns, keep='first')
            
            removed_rows = initial_rows - len(self.df)
            logging.info(f"Removed {removed_rows} duplicate content rows ({removed_rows/initial_rows*100:.2f}% of data)")
            
            # Fill missing values
            logging.info("Creating text representation...")
            self.df['ability'] = self.df['ability'].fillna('Unknown ability')
            self.df['skill'] = self.df['skill'].fillna('Unknown skill')
            
            # Process title column to keep only the first title if multiple exist
            if 'title' in self.df.columns:
                self.df['title'] = self.df['title'].apply(lambda x: str(x).split(';')[0].strip() if pd.notna(x) else 'Unknown title')
                # Apply title case formatting (capitalize first letter of each word)
                self.df['title'] = self.df['title'].apply(lambda x: TextProcessor.format_title(x))
            else:
                self.df['title'] = 'Unknown title'
            
            # Create text representation for embeddings (without program and text_representation columns)
            # Use ability, title, and skill for embeddings
            self.df['embedding_text'] = self.df.apply(lambda row: " | ".join([
                TextProcessor.clean_text(str(row.get('ability', ''))),
                TextProcessor.clean_text(str(row.get('title', ''))),
                TextProcessor.clean_text(str(row.get('skill', '')))
            ]), axis=1)
            
            # Optimize memory usage
            self.optimize_dataframe_memory()
            
            # Save processed data
            processed_file = os.path.join(self.output_folder, 'processed_resumes.csv')
            self.df.to_csv(processed_file, index=False)
            logging.info(f"Processed {len(self.df)} resumes and saved to CSV.")
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def optimize_dataframe_memory(self):
        """Optimize DataFrame memory usage."""
        start_mem = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        logging.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
        
        # Convert object types to categories where appropriate
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].nunique() < 0.5 * len(self.df):
                self.df[col] = self.df[col].astype('category')
        
        # Downcast numeric columns
        for col in self.df.select_dtypes(include=['int']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        
        for col in self.df.select_dtypes(include=['float']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        # Print memory usage
        end_mem = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        logging.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
        logging.info(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
        
        return self.df
    
    def _filter_person(self, df):
        """Filter persons by ID."""
        if 'person_id' in df.columns:
            df['person_id'] = df['person_id'].astype('int32')
            return df[df['person_id'] <= 54928]
        return df
    
    def _aggregate_text(self, df, group_col='person_id'):
        """Aggregate text data by group column."""
        if group_col in df.columns:
            return df.groupby(group_col).agg(lambda x: '; '.join(x.dropna().unique())).reset_index()
        return df