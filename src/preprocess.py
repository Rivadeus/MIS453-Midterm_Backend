import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove special characters.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = text.strip()
    return text