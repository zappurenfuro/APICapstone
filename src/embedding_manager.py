# -*- coding: utf-8 -*-

import os
import gc
import time
import logging
import traceback
import numpy as np
import torch
import psutil
import multiprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from .utils.dataset import ResumeDataset
from .text_processor import TextProcessor

class EmbeddingManager:
    """Class for managing embeddings."""
    
    def __init__(self, model_name: str, device, use_mixed_precision: bool, output_folder: str):
        """Initialize the EmbeddingManager."""
        self.model_name = model_name
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.output_folder = output_folder
        self.model = None
        self.embeddings = None
        self.embeddings_file = None
        self.embeddings_shape = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        logging.info(f"Loading embedding model ({self.model_name})...")
        self.model = SentenceTransformer(self.model_name)
        self.model = self.model.to(self.device)
        
        # Optimize model for inference
        self.model.eval()
    
    def create_embeddings(self, df, batch_size=32):
        """Create embeddings using PyTorch DataLoader for efficient batching."""
        if df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logging.info("Creating embeddings with DataLoader for efficient batching...")
        
        # Determine optimal batch size based on system resources
        if torch.cuda.is_available():
            # If GPU is available, adjust batch size based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            optimal_batch_size = min(batch_size, max(4, int(gpu_memory_gb * 4)))  # Heuristic
        else:
            # If CPU only, adjust based on available RAM
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            optimal_batch_size = min(batch_size, max(4, int(available_ram_gb * 2)))  # Heuristic
        
        # Determine optimal number of workers
        optimal_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 workers
        
        logging.info(f"Using batch size: {optimal_batch_size}, workers: {optimal_workers}")
        
        # Create dataset and dataloader
        # Use embedding_text instead of text_representation
        resume_texts = df['embedding_text'].tolist()
        dataset = ResumeDataset(resume_texts)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=optimal_batch_size, 
            shuffle=False, 
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()  # Only use pin_memory if GPU is available
        )
        
        # Pre-allocate output array in RAM
        embedding_dim = 1024  # For BAAI/bge-large-en-v1.5
        all_embeddings = np.zeros((len(resume_texts), embedding_dim), dtype=np.float32)
        
        # Process batches
        start_time = time.time()
        start_idx = 0
        
        for i, batch in enumerate(tqdm(dataloader, desc="Creating embeddings")):
            # Use mixed precision if available
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            else:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            
            # Store in pre-allocated array
            end_idx = start_idx + len(batch)
            all_embeddings[start_idx:end_idx] = batch_embeddings
            start_idx = end_idx
            
            # Log progress and clear memory periodically
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Processed {end_idx}/{len(resume_texts)} samples ({end_idx/len(resume_texts)*100:.1f}%) in {elapsed:.1f}s")
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Convert to tensor and move to device
        self.embeddings = torch.tensor(all_embeddings).to(self.device)
        
        # Save embeddings to file
        np.save(os.path.join(self.output_folder, 'resume_embeddings.npy'), all_embeddings)
        logging.info(f"Created and saved embeddings for {len(resume_texts)} resumes in {time.time() - start_time:.1f}s")
        
        return self.embeddings
    
    def load_embeddings(self, file_path=None):
        """Load pre-computed embeddings from file."""
        if file_path is None:
            file_path = os.path.join(self.output_folder, 'resume_embeddings.npy')
        
        if not os.path.exists(file_path):
            logging.warning(f"Embeddings file {file_path} not found. Creating embeddings...")
            return None
        
        logging.info(f"Loading embeddings from {file_path}...")
        
        try:
            # Use memory mapping for large files
            if os.path.getsize(file_path) > 1e9:  # If file is larger than 1GB
                embeddings_cpu = np.load(file_path, mmap_mode='r')
                logging.info(f"Using memory mapping for large embeddings file")
                
                # For operations, we'll load chunks into GPU as needed
                self.embeddings_file = file_path
                self.embeddings_shape = embeddings_cpu.shape
                
                # Load a small batch into GPU for immediate use
                batch_size = min(1000, embeddings_cpu.shape[0])
                self.embeddings = torch.tensor(embeddings_cpu[:batch_size]).to(self.device)
            else:
                # For smaller files, load everything into memory
                embeddings_cpu = np.load(file_path)
                self.embeddings = torch.tensor(embeddings_cpu).to(self.device)
            
            logging.info(f"Loaded embeddings with shape {embeddings_cpu.shape}")
            return self.embeddings
            
        except Exception as e:
            logging.error(f"Error loading embeddings: {str(e)}")
            logging.error(traceback.format_exc())
            logging.warning("Failed to load embeddings.")
            return None
    
    def encode_text(self, text):
        """Encode a single text."""
        # Clean the input text
        text = TextProcessor.clean_text(text)
        
        # Encode the text
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    embedding = self.model.encode([text], normalize_embeddings=True)
            else:
                embedding = self.model.encode([text], normalize_embeddings=True)
        
        return embedding