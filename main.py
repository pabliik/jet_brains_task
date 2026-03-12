import re
import urllib.request
import zipfile
import numpy as np
from collections import Counter

def get_text_data():
    """Downloads and reads a chunk of the text8 dataset."""
    try:
        urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", "text8.zip")
    except Exception as e:
        print(f"Error downloading: {e}.")
        return ""
    
    with zipfile.ZipFile("text8.zip") as f:
        text = f.read("text8").decode("utf-8")
    
    # Using only first 5 million characters
    return text[:5_000_000]

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b[a-z]+\b', text) # Keeps only text, no punctuation or numbers 