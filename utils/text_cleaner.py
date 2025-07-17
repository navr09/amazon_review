import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def clean_review_text(text: Optional[str]) -> str:
    """Robust text cleaning for review content"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Basic HTML tag removal (without BeautifulSoup)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove common HTML entities and artifacts
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'\b(br|div|span|class)\b', ' ', text, flags=re.IGNORECASE)
        
        # Remove special characters except basic punctuation
        text = re.sub(r"[^\w\s'.,!?-]", ' ', text)
        
        # Remove short/meaningless words (1-2 characters)
        text = ' '.join([word for word in text.split() if len(word) > 2])
        
        # Normalize whitespace and case
        return re.sub(r'\s+', ' ', text).strip().lower()
    except Exception as e:
        logger.warning(f"Text cleaning error: {str(e)}")
        return ""