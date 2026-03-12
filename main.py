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

def freq_count(tokens, min_count=5):
    """Counts frequencies and removes very rare words."""
    freq = Counter(tokens)
    return {w: c for w, c in freq.items() if c >= min_count}

def subsample_tokens(tokens, freq, threshold=1e-5):
    """
    Randomly drops highly frequent words such as [the, a, an, etc] to speed up training 
    and improve representations of less frequent words.
    """
    total_tokens = sum(freq.values())
    subsampled = []
    
    for word in tokens:
        if word not in freq:
            continue
        
        word_fraction = freq[word] / total_tokens # calculate how frequent the word is in fraction exp: 0,05 for 5 %
        p_discard = max(0, 1 - np.sqrt(threshold / word_fraction)) # get probability between 0 and 1
        
        if np.random.rand() > p_discard:
            subsampled.append(word)
            
    return subsampled

def vocab(freq):
    word2ind = {w: i for i, w in enumerate(freq.keys())} # word to ind exp: "fox" -> 0
    ind2word = {i: w for w, i in word2ind.items()} # ind to word exp: 0 -> "fox"
    return word2ind, ind2word

def get_pairs(tokens, word2ind, window_size=5):
    """Generates (center_word, context_word) pairs with window size 5."""
    pairs = []
    indices = [word2ind[w] for w in tokens if w in word2ind] # takes only indices
    
    for i, center in enumerate(indices):
        left = max(0, i - window_size) # calculate how far it will go to the left
        right = min(len(indices), i + window_size + 1) # how far it will go to the right
        
        for j in range(left, right):
            if i != j:
                pairs.append((center, indices[j]))
    return pairs

def build_sampling_table(freq, word2ind, table_size=1_000_000):
    """Builds a table for negative sampling based on U(w)^0.75."""
    table = []
    total = sum([c**0.75 for c in freq.values()])
    
    for word, count in freq.items():
        n = int((count**0.75 / total) * table_size)
        table.extend([word2ind[word]] * n)
        
    return np.array(table)

def sigmoid(x):
    # Clip values to prevent overflow in exp
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))