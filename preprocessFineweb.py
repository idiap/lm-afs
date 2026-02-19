# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Damien Teney damien.teney@idiap.ch
# SPDX-License-Identifier: MIT

'''
Standalone function to download and tokenize the Fineweb dataset, saved as minimalist .bin files.
https://huggingface.co/datasets/HuggingFaceFW/fineweb
'''

import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np
from dataBin import writeBinFile

# Parse aguments
assert __name__ == "__main__", "This file is only meant as a standalone function"
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Data directory")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-n", "--n_shards", type=int, default=3, help="Max number of shards to save (allow preprocessing only a fraction of the dataset)") # Not implemented
args = parser.parse_args()
print(f"Data directory: {args.data_dir}")
os.makedirs(args.data_dir, exist_ok=True)

# Download the dataset
data = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", cache_dir=args.data_dir)

# Init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc): # Tokenize single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

# Tokenize all documents and write output shards, each of 'shard_size' tokens
shardId = 0
allTokens = np.empty((args.shard_size,), dtype=np.uint16) # Preallocate buffer to hold current shard
tokenCount = 0
progressBar = None
for doc in data:
    tokens = tokenize(doc)
    if tokenCount + len(tokens) < args.shard_size: # Enough space in the current shard for the new tokens?
        allTokens[tokenCount:tokenCount+len(tokens)] = tokens # Append tokens to current shard
        tokenCount += len(tokens)
        if progressBar is None: # Create progress bar
            progressBar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shardId}")
        progressBar.update(len(tokens)) # Update progress bar
    else: # Write the current shard and start a new one
        split = "val" if shardId == 0 else "train"
        filename = os.path.join(args.data_dir, f"fineweb_{split}_{shardId:06d}.bin")
        # Split the document into whatever fits in this shard; the remainder goes to next one
        remainder = args.shard_size - tokenCount
        progressBar.update(remainder)
        allTokens[tokenCount:tokenCount+remainder] = tokens[:remainder]
        writeBinFile(filename, allTokens)
        shardId += 1
        if shardId > args.n_shards: return
        progressBar = None
        # Populate the next shard with the leftovers of the current doc
        allTokens[0:len(tokens) - remainder] = tokens[remainder:]
        tokenCount = len(tokens) - remainder

# Write remaining tokens as the last shard
if tokenCount != 0:
    assert shardId > 0
    split = "train"
    filename = os.path.join(args.data_dir, f"fineweb_{split}_{shardId:06d}.bin")
    writeBinFile(filename, allTokens[:tokenCount])
