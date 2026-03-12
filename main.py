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

def update_batch(center_batch, context_batch, negatives_batch, W_in, W_out, lr):
    """
    Forward pass, loss calculation and backpropagation.
    """
    
    u_c = W_in[center_batch]
    v_pos = W_out[context_batch]
    v_neg = W_out[negatives_batch]         
    
    # Forward Pass
    # Positive contexts: dot product of each center with its context
    score_pos = np.sum(u_c * v_pos, axis=1) 
    prob_pos = sigmoid(score_pos)           
    
    # Negative contexts: dot products using Einstein summation
    score_neg = np.einsum('bd,bkd->bk', u_c, v_neg) 
    prob_neg = sigmoid(score_neg)                   
    
    # Compute Loss
    loss_pos = -np.log(prob_pos + 1e-10)
    loss_neg = -np.sum(np.log(1 - prob_neg + 1e-10), axis=1)
    batch_loss = np.sum(loss_pos + loss_neg)
    
    # Backward Pass
    error_pos = prob_pos - 1
    error_neg = prob_neg - 0
    
    # Gradients for W_out (context and negative matrices)
    grad_v_pos = error_pos[:, None] * u_c
    grad_v_neg = error_neg[:, :, None] * u_c[:, None, :]
    
    # Gradients for W_in (center matrix)
    grad_u_c = (error_pos[:, None] * v_pos) + np.sum(error_neg[:, :, None] * v_neg, axis=1)
    
    # Apply Updates (Using np.add.at for duplicate indices in a batch)
    np.add.at(W_in, center_batch, -lr * grad_u_c)
    np.add.at(W_out, context_batch, -lr * grad_v_pos)
    
    flat_neg_indices = negatives_batch.flatten()
    flat_grad_v_neg = grad_v_neg.reshape(-1, W_out.shape[1])
    np.add.at(W_out, flat_neg_indices, -lr * flat_grad_v_neg)
    
    return batch_loss

def train(pairs, table, W_in, W_out, epochs=5, K=5, initial_lr=0.025, batch_size=1024):
    """Batched training loop with learning rate decay."""
    total_pairs = len(pairs)
    pairs_arr = np.array(pairs) # Convert to numpy array for faster slicing
    
    for epoch in range(epochs):
        np.random.shuffle(pairs_arr)
        total_loss = 0
        
        # Linear Learning Rate Decay
        lr = max(initial_lr * (1 - epoch / epochs), 0.0001) 
        
        for i in range(0, total_pairs, batch_size):
            batch_pairs = pairs_arr[i:i + batch_size] 
            actual_batch_size = len(batch_pairs) # not always the batch size will be 1024
            
            centers = batch_pairs[:, 0]
            contexts = batch_pairs[:, 1]
            
            # Sample negatives for the entire batch at once
            negatives = np.random.choice(table, size=(actual_batch_size, K))

            loss = update_batch(centers, contexts, negatives, W_in, W_out, lr)
            total_loss += loss
            
        avg_loss = total_loss / total_pairs
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr:.4f}")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def nearest_neighbors(word, word2ind, ind2word, W_in, n=5):
    if word not in word2ind:
        return f"'{word}' not in vocabulary."
    
    idx = word2ind[word]
    vec = W_in[idx]
    
    # Vectorized similarity computation
    norms = np.linalg.norm(W_in, axis=1)
    sims = np.dot(W_in, vec) / (norms * np.linalg.norm(vec) + 1e-10)
    
    # Get top N indices (excluding the word itself)
    top_indices = np.argsort(-sims)[1:n+1]
    return [(ind2word[i], sims[i]) for i in top_indices]

if __name__ == "__main__":
    print("Loading data...")
    text = get_text_data()
    tokens = tokenize(text)
    
    print("Building vocabulary and subsampling...")
    freq = freq_count(tokens, min_count=5) # Increased min_count for better quality
    subsampled_tokens = subsample_tokens(tokens, freq)
    word2ind, ind2word = vocab(freq)
    
    print(f"Vocabulary size: {len(word2ind)}")
    
    print("Generating pairs and sampling table...")
    pairs = get_pairs(subsampled_tokens, word2ind)
    table = build_sampling_table(freq, word2ind)
    
    print(f"Total training pairs: {len(pairs)}")
    
    # Initialize weights
    vocab_size = len(word2ind)
    embed_dim = 100 
    
    # It is common practice to initialize W_in with small random numbers 
    # and W_out with zeros in Skip-Gram
    W_in  = np.random.uniform(-0.05, 0.05, (vocab_size, embed_dim))
    W_out = np.zeros((vocab_size, embed_dim))
    
    print("Starting training...")
    train(pairs, table, W_in, W_out, epochs=5, K=5, initial_lr=0.025, batch_size=1024)
    
    print("\nEvaluating 'king' nearest neighbors:")
    print(nearest_neighbors("king", word2ind, ind2word, W_in))