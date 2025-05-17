# -*- coding: utf-8 -*-

import gc
import logging
import multiprocessing
import tempfile
import torch
import psutil

class SystemManager:
    """Class for managing system resources and device setup."""
    
    @staticmethod
    def setup_device():
        """Set up device (CPU/GPU) and return device info."""
        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Set up mixed precision if available
        use_mixed_precision = False
        if device.type == 'cuda' and torch.cuda.is_available():
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                use_mixed_precision = True
                logging.info("Mixed precision is available and will be used")
                
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logging.info("CUDA optimizations enabled")
        
        return device, use_mixed_precision
    
    @staticmethod
    def log_system_resources():
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
    
    @staticmethod
    def setup_ram_disk(size_mb=1024):
        """Set up a RAM disk for temporary files."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = temp_dir.name
        logging.info(f"Created RAM-based temporary directory at {temp_path}")
        return temp_dir, temp_path
    
    @staticmethod
    def cleanup_resources(temp_dir=None):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove temporary directory
        if temp_dir:
            temp_dir.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        logging.info("Cleanup complete")