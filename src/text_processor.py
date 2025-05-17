# -*- coding: utf-8 -*-

import re
import pandas as pd

class TextProcessor:
    """Class for text processing operations."""
    
    @staticmethod
    def clean_text(text):
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
    
    @staticmethod
    def format_title(title):
        """Format title with proper capitalization (title case)."""
        if not title or pd.isna(title):
            return "Unknown Title"
        
        # Clean the title first
        title = TextProcessor.clean_text(title)
        
        # Convert to title case (first letter of each word capitalized)
        words = title.lower().split()
        return ' '.join(word.capitalize() for word in words)