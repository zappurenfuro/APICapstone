# %%
import os
import sys
import numpy as np
import pandas as pd
import logging
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union
import tempfile
import hashlib
import time
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
import traceback
import re
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define ResumeDataset outside methods to make it picklable
class ResumeDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# Auto-install required packages
def install_required_packages():
    required_packages = {
        'psutil': 'psutil',
        'umap-learn': 'umap',
        'docx2txt': 'docx2txt',
        'PyPDF2': 'PyPDF2',
        'textract': 'textract',
        'tqdm': 'tqdm',
        'colorama': 'colorama'
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            logging.info(f"Installing {package_name}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install required packages
install_required_packages()

# Now import the packages
import psutil
import docx2txt
import PyPDF2
import textract
from tqdm import tqdm
try:
    from umap import UMAP
except ImportError:
    logging.warning("UMAP import failed even after installation attempt")

class ResumeScanner:
    """Main class for resume scanning and matching functionality."""
    
    def __init__(self, input_folder: str, output_folder: str, cv_folder: str = None):
        """Initialize the ResumeScanner with input and output folders."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cv_folder = cv_folder
        self.model = None
        self.df = None
        self.embeddings = None
        self.temp_dir = None
        self.results_saved = {}  # Track saved files to avoid duplicates
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Set up RAM disk for temporary files
        self.setup_ram_disk()
        
        # Check for GPU availability and log system resources
        self.setup_device()
        
        # Load the embedding model
        self._load_model()
    
    def clean_text(self, text):
        """
        Clean text by removing quotes and exclamation marks.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove single quotes, double quotes, and exclamation marks
        text = text.replace("'", "").replace('"', "").replace('!', "")
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_title_text(self, title):
        """Special cleaning for title text to improve matching."""
        if not title or pd.isna(title):
            return ""
        
        # Convert to string and lowercase
        title = str(title).lower()
        
        # Remove seniority indicators that don't affect core role
        title = re.sub(r'\b(senior|sr|junior|jr|lead|principal|chief|head)\b', '', title)
        
        # Normalize common title variations
        title = title.replace('dev', 'developer').replace('eng', 'engineer').replace('sw', 'software')
        
        # Remove multiple spaces and trim
        text = re.sub(r'\s+', ' ', title).strip()
        
        return text
    
    def setup_device(self):
        """Set up device (CPU/GPU) and log system resources."""
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Log system resources
        self.log_system_resources()
        
        # Set up mixed precision if available
        self.use_mixed_precision = False
        if self.device.type == 'cuda' and torch.cuda.is_available():
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                self.use_mixed_precision = True
                logging.info("Mixed precision is available and will be used")
                
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logging.info("CUDA optimizations enabled")
    
    def log_system_resources(self):
        """Log available system resources."""
        # CPU info
        cpu_count = multiprocessing.cpu_count()
        
        # RAM info
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 ** 3)  # GB
        ram_available = ram.available / (1024 ** 3)  # GB
        
        # GPU info
        gpu_info = "Not available"
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        
        logging.info(f"System resources:")
        logging.info(f"  CPU: {cpu_count} cores")
        logging.info(f"  RAM: {ram_total:.2f} GB total, {ram_available:.2f} GB available")
        if torch.cuda.is_available():
            logging.info(f"  GPU: {gpu_info} with {gpu_memory:.2f} GB memory")
        else:
            logging.info(f"  GPU: {gpu_info}")
    
    def setup_ram_disk(self, size_mb=1024):
        """Set up a RAM disk for temporary files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = self.temp_dir.name
        logging.info(f"Created RAM-based temporary directory at {temp_path}")
        return temp_path
    
    def _load_model(self):
        """Load the sentence transformer model."""
        logging.info("Loading embedding model (BAAI/bge-large-en-v1.5)...")
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.model = self.model.to(self.device)
        
        # Optimize model for inference
        self.model.eval()
    
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
                        df[col] = df[col].apply(self.clean_text)
            
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
                self.df['title'] = self.df['title'].apply(lambda x: self._format_title(x))
            else:
                self.df['title'] = 'Unknown title'
            
            # Create text representation for embeddings (without program and text_representation columns)
            # Use ability, title, and skill for embeddings
            self.df['embedding_text'] = self.df.apply(lambda row: " | ".join([
                self.clean_text(str(row.get('ability', ''))),
                self.clean_title_text(str(row.get('title', ''))) * 3,  # Repeat title 3 times for higher weight
                self.clean_text(str(row.get('skill', '')))
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
    
    def _format_title(self, title):
        """Format title with proper capitalization (title case)."""
        if not title or pd.isna(title):
            return "Unknown Title"
        
        # Clean the title first
        title = self.clean_text(title)
        
        # Convert to title case (first letter of each word capitalized)
        words = title.lower().split()
        return ' '.join(word.capitalize() for word in words)
    
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
    
    def create_embeddings(self, batch_size=32):
        """Create embeddings using PyTorch DataLoader for efficient batching."""
        if self.df is None:
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
        resume_texts = self.df['embedding_text'].tolist()
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
                with torch.amp.autocast('cuda'):
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
            return self.create_embeddings()
        
        logging.info(f"Loading embeddings from {file_path}...")
        
        try:
            # Use memory mapping for large files
            if os.path.getsize(file_path) > 1e9:  # If file is larger than 1GB
                embeddings_cpu = np.load(file_path, mmap_mode='r', allow_pickle=True)
                logging.info(f"Using memory mapping for large embeddings file")
                
                # For operations, we'll load chunks into GPU as needed
                self.embeddings_file = file_path
                self.embeddings_shape = embeddings_cpu.shape
                
                # Load a small batch into GPU for immediate use
                batch_size = min(1000, embeddings_cpu.shape[0])
                self.embeddings = torch.tensor(embeddings_cpu[:batch_size]).to(self.device)
            else:
                # For smaller files, load everything into memory
                embeddings_cpu = np.load(file_path, allow_pickle=True)
                self.embeddings = torch.tensor(embeddings_cpu).to(self.device)
            
            logging.info(f"Loaded embeddings with shape {embeddings_cpu.shape}")
            return self.embeddings
            
        except Exception as e:
            logging.error(f"Error loading embeddings: {str(e)}")
            logging.error(traceback.format_exc())
            logging.warning("Creating new embeddings...")
            return self.create_embeddings()
    
    def extract_text_from_file(self, file_path: str) -> str:
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
                text = self._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc', '.docs']:
                text = self._extract_from_doc(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Clean the extracted text
            text = self.clean_text(text)
                
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
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + " "  # Use space instead of newline
            
            # If PyPDF2 fails to extract meaningful text, try textract as backup
            if not text.strip():
                logging.info(f"PyPDF2 failed to extract text from {file_path}, trying textract...")
                text = textract.process(file_path).decode('utf-8')
                text = text.replace('\n', ' ')  # Replace newlines with spaces
                
        except Exception as e:
            logging.error(f"Error in PDF extraction: {str(e)}")
            logging.error(traceback.format_exc())
            # Try textract as a fallback
            try:
                text = textract.process(file_path).decode('utf-8')
                text = text.replace('\n', ' ')  # Replace newlines with spaces
            except Exception as e2:
                logging.error(f"Textract also failed: {str(e2)}")
                raise
                
        return text
    
    def _extract_from_doc(self, file_path: str) -> str:
        """Extract text from DOC/DOCX file."""
        try:
            # Try docx2txt first (for .docx)
            text = docx2txt.process(file_path)
            text = text.replace('\n', ' ')  # Replace newlines with spaces
        except Exception as e:
            logging.error(f"docx2txt failed: {str(e)}")
            # Fall back to textract (handles .doc and other formats)
            try:
                text = textract.process(file_path).decode('utf-8')
                text = text.replace('\n', ' ')  # Replace newlines with spaces
            except Exception as e2:
                logging.error(f"Textract also failed: {str(e2)}")
                raise
        
        return text
    
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
        text = self.clean_text(text)
        
        # Debug: Print a hash of the input text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        logging.info(f"Matching text (hash: {text_hash[:8]}...)")
        
        # Check if embeddings exist
        if self.embeddings is None:
            logging.warning("Embeddings not found. Loading or creating embeddings...")
            self.load_embeddings()
        
        # Encode the text
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    user_embedding = self.model.encode([text], normalize_embeddings=True)
            else:
                user_embedding = self.model.encode([text], normalize_embeddings=True)
        
        # If we're using memory mapping, process in chunks
        if hasattr(self, 'embeddings_file') and self.embeddings_file:
            # Load embeddings in chunks to avoid loading everything into RAM
            embeddings_mmap = np.load(self.embeddings_file, mmap_mode='r')
            
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
            if self.embeddings.is_cuda:
                embeddings_cpu = self.embeddings.detach().cpu().numpy()
            else:
                embeddings_cpu = self.embeddings.detach().numpy()
            
            # Calculate cosine similarity
            all_similarities = cosine_similarity(user_embedding, embeddings_cpu)[0]
            
            # Get top matches
            top_indices = all_similarities.argsort()[-top_n:][::-1]
            top_similarities = all_similarities[top_indices]
        
        # Get the corresponding rows from the dataframe
        top_matches = self.df.iloc[top_indices].copy()
        top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
        
        # Apply title case formatting to ensure consistent display
        top_matches['title'] = top_matches['title'].apply(lambda x: self._format_title(x))
        
        # Save results with unique filename if provided
        
        logging.info(f"Found {len(top_matches)} {match_type} matches (not saved to file)")
        
        return top_matches
    
    def process_resume_file(self, file_path: str, top_n: int = 5) -> Dict:
        """
        Process a resume file and match it against the database.
        Domain-specific matching is disabled.
        
        Args:
            file_path: Path to the resume file
            top_n: Number of top matches to return
            
        Returns:
            Dictionary with results
        """
        try:
            # Reset saved results tracking for this file
            self.results_saved = {}
            
            # Extract text from file
            resume_text = self.extract_text_from_file(file_path)
            logging.info(f"Extracted {len(resume_text)} characters from {file_path}")
            
            # Get base filename for output naming
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Print file processing header
            print(f"\n{Fore.MAGENTA}{'='*80}")
            print(f"{Fore.MAGENTA}📄 PROCESSING RESUME: {file_path}")
            print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
            
            # Match against database
            top_matches = self.match_text(resume_text, top_n, file_path, "resume")
            
            # Match against job titles
            job_matches = self.match_resume_to_jobs(resume_text, top_n, file_path)
            
            # Print completion message
            print(f"\n{Fore.GREEN}✅ Processed {file_path} and found {len(top_matches)} matches{Style.RESET_ALL}")
            
            return {
                'resume_text': resume_text,
                'top_matches': top_matches,
                'job_matches': job_matches
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
        This uses only general matching without domain-specific matching.
        
        Args:
            resume_text: Text of the resume to match
            top_n: Number of top matches to return
            file_name: Optional name of the resume file for output naming
            
        Returns:
            DataFrame with top job matches and similarity scores
        """
        try:
            if self.df is None:
                raise ValueError("Dataset not loaded. Call load_data() first.")
            
            # Clean the resume text
            resume_text = self.clean_text(resume_text)
            
            # Get all matches using general matching only
            job_matches = self.match_text(resume_text, top_n, file_name, "job")
            
            # Print results in a formatted way
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}🏆 TOP JOB MATCHES FOR {file_name if file_name else 'resume'}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            
            for idx, row in job_matches.iterrows():
                title = row['title']
                similarity = row['similarity_percentage']
                description = row.get('embedding_text', '')
                if description:
                    description = description[:100] + "..." if len(description) > 100 else description
                    print(f"{Fore.YELLOW}🔹 {title} - {similarity:.2f}%{Style.RESET_ALL}")
                    print(f"   {description}")
                    print()
                else:
                    print(f"{Fore.YELLOW}🔹 {title} - {similarity:.2f}%{Style.RESET_ALL}")
                    print()
            
            return job_matches
        except Exception as e:
            logging.error(f"Error matching resume to jobs: {str(e)}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    def scan_cv_folder(self, folder_path=None, top_n=5):
        """
        Scan a folder for CV files and process each one.
        
        Args:
            folder_path: Path to the folder containing CV files
            top_n: Number of top matches to return for each CV
            
        Returns:
            Dictionary with results for each CV
        """
        if folder_path is None:
            folder_path = self.cv_folder
        
        if folder_path is None:
            raise ValueError("CV folder path not specified")
        
        if not os.path.exists(folder_path):
            raise ValueError(f"CV folder path does not exist: {folder_path}")
        
        print(f"\n{Fore.BLUE}{'='*80}")
        print(f"{Fore.BLUE}📂 SCANNING CV FOLDER: {folder_path}")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        
        # Get all CV files in the folder
        cv_files = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.pdf', '.docx', '.doc')):
                cv_files.append(file_path)
        
        if not cv_files:
            print(f"{Fore.YELLOW}⚠️ No CV files found in folder: {folder_path}{Style.RESET_ALL}")
            return {}
        
        print(f"{Fore.GREEN}Found {len(cv_files)} CV files in folder{Style.RESET_ALL}")
        
        # Process each CV file
        results = {}
        for cv_file in cv_files:
            try:
                print(f"\n{Fore.BLUE}Processing CV file: {os.path.basename(cv_file)}{Style.RESET_ALL}")
                result = self.process_resume_file(cv_file, top_n)
                results[cv_file] = result
            except Exception as e:
                print(f"{Fore.RED}❌ Error processing {cv_file}: {str(e)}{Style.RESET_ALL}")
                results[cv_file] = {'error': str(e), 'file_path': cv_file}
        
        print(f"\n{Fore.GREEN}🎉 Completed scanning {len(cv_files)} CV files{Style.RESET_ALL}")
        return results
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove temporary directory
        if self.temp_dir:
            self.temp_dir.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        logging.info("Cleanup complete")


# Main execution
if __name__ == "__main__":
    # Define folders
    input_folder = "input"  # Change to your input folder
    output_folder = "output"  # Change to your output folder
    cv_folder = r"C:\Users\USER\Downloads\UpdateModelAI\cv_dummy"  # Specified CV folder
    
    # Initialize scanner with the CV folder
    scanner = ResumeScanner(input_folder, output_folder, cv_folder)
    
    try:
        # Load and process data with optimization
        scanner.load_data()
        
        # Create embeddings
        scanner.create_embeddings()
        
        print(f"{Fore.GREEN}✅ Successfully processed CSV data and created embeddings{Style.RESET_ALL}")
        
        # Scan the CV folder
        results = scanner.scan_cv_folder()
        
        # Print a summary of the results
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}📊 SUMMARY OF RESULTS")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        for cv_file, result in results.items():
            if 'error' in result:
                print(f"{Fore.RED}❌ {os.path.basename(cv_file)}: Error - {result['error']}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}✅ {os.path.basename(cv_file)}: Found {len(result['top_matches'])} matches{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}🎉 All processing completed!{Style.RESET_ALL}")
        print(f"Check the output folder ({output_folder}) for results and CSV files.")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error during processing: {str(e)}{Style.RESET_ALL}")
        print(traceback.format_exc())
    
    finally:
        # Clean up resources
        scanner.cleanup()
# %%
