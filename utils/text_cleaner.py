import re
from typing import Optional
import logging
from bs4 import BeautifulSoup
import contractions

logger = logging.getLogger(__name__)

def clean_review_text(text: Optional[str]) -> str:
    """
    Enhanced text cleaning for reviews with:
    - Better HTML cleaning
    - Contraction expansion
    - Misspelling correction (basic)
    - Punctuation normalization
    - Special character handling
    """
    if not isinstance(text, str):
        return ""
    
    try:
        # 1. Remove HTML tags and entities
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        
        # 2. Fix HTML artifacts and encoding issues
        text = re.sub(r'&[a-z]+;', '', text)
        text = re.sub(r'\\[a-z]+', '', text)
        
        # 3. Expand contractions (I'm -> I am, etc.)
        text = contractions.fix(text)
        
        # 4. Correct common misspellings (basic examples)
        common_misspellings = {
            r'\bu\b': 'you',
            r'\byuor\b': 'your',
            r'\byuo\b': 'you',
            r'\breccommend\b': 'recommend',
            r'\bimponrtance\b': 'importance',
            r'\bdiffernt\b': 'different',
            r'\bslihgtest\b': 'slightest',
            r'\bthrouh\b': 'through',
            r'\bboook\b': 'book',
            r'\bimmediatley\b': 'immediately'
        }
        for pattern, replacement in common_misspellings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 5. Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?\'"]', '', text)
        
        # 6. Normalize punctuation and spacing
        text = re.sub(r'\s+([.,!?])\s*', r'\1 ', text)  # Fix spacing around punctuation
        text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
        text = re.sub(r'([.,!?])\1+', r'\1', text)  # Remove duplicate punctuation
        
        # 7. Remove quote artifacts
        text = re.sub(r'\\"', '"', text)
        
        # 8. Final cleanup
        text = text.strip()
        text = text.lower()  # Optional: lowercase everything
        
        return text
    except Exception as e:
        logger.warning(f"Text cleaning error: {str(e)}")
        return ""