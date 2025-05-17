import numpy as np
import os
import logging
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_embeddings(output_file="output/valid_embeddings.npy", num_samples=100):
    """
    Create valid embeddings file for testing.
    
    Args:
        output_file: Path to save the embeddings
        num_samples: Number of sample embeddings to create
    """
    logging.info(f"Creating {num_samples} valid embeddings...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Option 1: Create random embeddings (fast but not realistic)
        # embeddings = np.random.rand(num_samples, 1024).astype(np.float32)
        
        # Option 2: Create realistic embeddings using the same model
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Generate some sample texts
        sample_texts = [
            "Software Engineer with 5 years of experience in Python and JavaScript",
            "Data Scientist specializing in machine learning and AI",
            "Frontend Developer with React and Angular experience",
            "DevOps Engineer familiar with AWS and Docker",
            "Product Manager with agile methodology expertise"
        ]
        
        # Repeat the samples to reach num_samples
        texts = []
        while len(texts) < num_samples:
            texts.extend(sample_texts)
        texts = texts[:num_samples]
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Save embeddings
        np.save(output_file, embeddings)
        
        # Verify the file
        loaded = np.load(output_file)
        
        logging.info(f"Successfully created embeddings file at {output_file}")
        logging.info(f"Embeddings shape: {loaded.shape}, dtype: {loaded.dtype}")
        logging.info(f"File size: {os.path.getsize(output_file)} bytes")
        
        # Check file header
        with open(output_file, 'rb') as f:
            header = f.read(16)
            logging.info(f"File header (hex): {header.hex()}")
        
        return True
    except Exception as e:
        logging.error(f"Error creating embeddings: {str(e)}")
        return False

if __name__ == "__main__":
    create_embeddings()