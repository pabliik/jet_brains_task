Core training loop of Word2Vec in pure NumPy

This repository contains an implementation of the Word2Vec Skip-gram model with Negative Sampling (SGNS), built using Python and NumPy.

Key Features

- Subsampling: Automatically skips very common words like "the" and "and." This makes training faster and helps the model focus on more meaningful words.
- Negative Sampling: Instead of checking every word in the dictionary for every step, the model picks a few "fake" examples to compare against.
- Learning Rate Decay: The model starts with big steps to learn quickly and gradually takes smaller.
