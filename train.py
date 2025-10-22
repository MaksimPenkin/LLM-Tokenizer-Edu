"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import time
from minbpe import BasicTokenizer, RegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("../datasets/taylorswift.txt", "r", encoding="utf-8").read()

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
