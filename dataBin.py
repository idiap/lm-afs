# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Damien Teney damien.teney@idiap.ch
# SPDX-License-Identifier: MIT

'''
Functions to write and load tokens using minimalist binary files.
'''

import glob
import random
import math
from collections import Counter
import numpy as np
import torch

# Save tokens in .bin file
def writeBinFile(filename, tokens):
    print(f"Writing {len(tokens):,} tokens to {filename}")
    assert len(tokens) < 2**31, "token count too large" # ~2.1B tokens
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # Magic number
    header[1] = 1 # Version
    header[2] = len(tokens) # Number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    if not isinstance(tokens, np.ndarray) or (tokens.dtype != np.uint16): # Convert to numpy array, if not already one
        assert all(0 <= t < 2**16 for t in tokens), "token dictionary too large for uint16" # Validate that no token exceeds a uint16
        tokens = np.array(tokens, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes()) # Header with 256 int32s
        f.write(tokens.tobytes()) # Tokens as a uint16

def peekBinFile(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32) # Read header, which is 256 int32 integers (4 bytes each)
    assert header[0] == 20240520, f"Magic number mismatch in data file: {fileName}"
    assert header[1] == 1, f"Unsupported version in data file: {fileName}"
    nTokens = header[2] # Number of tokens (claimed)
    return nTokens

def loadBinFile(filename): # Return vector of tokens as np.array uint16
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32) # Read header, which is 256 int32 integers (4 bytes each)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        nTokens = header[2] # Number of tokens (claimed)
        tokens = np.frombuffer(f.read(), dtype=np.uint16) # Read the rest, tokens stored as uint16
    assert len(tokens) == nTokens, "Number of tokens read does not match header"
    return tokens

# DataLoader for .bin files
class DataLoaderBin:
    def __init__(self, fileNamePattern, seqLength, initFileId=0, shuffle=False, trainingData=True, shuffleTokens=False):
        self.seqLength = seqLength
        self.shuffle = shuffle
        self.fileNames = sorted(glob.glob(fileNamePattern))
        assert len(self.fileNames) > 0, f"Did not find any files that match {fileNamePattern}"
        self.currentFileId = -1
        self.vocabMapping = None # Lookup table; stores (new token IDs + 1) as uint16; 0 means undefined (tokens to be discarded)
        # Count tokens in all the files
        self.nTokens = 0
        for fileName in self.fileNames:
            nTokensPerFile = peekBinFile(fileName)
            assert nTokensPerFile >= self.seqLength + 1
            self.nTokens += int(nTokensPerFile)
        self.initFileId = initFileId
        self.shuffleTokens = shuffleTokens
        self.vocabOffset = 0
        self.reset()

    def reset(self, forceReloadFile=False):
        fileId = self.initFileId % len(self.fileNames)
        if (self.currentFileId == fileId) and (not forceReloadFile): # No need to reset file
            if self.shuffle: random.shuffle(self.seqStarts) # Shuffle in place
            self.idInFile = 0 # Reset position in file
        else: # Reset file
            self.currentFileId = fileId - 1
            self.loadNextFile() # Load file 'fileId'

    def loadNextFile(self): # Load the next file
        if (len(self.fileNames) > 1) or (self.currentFileId < 0):
            self.currentFileId = (self.currentFileId + 1) % len(self.fileNames)
            self.tokens = loadBinFile(self.fileNames[self.currentFileId]) # np.array uint16
            self.tokens = self.applyVocabMapping(self.tokens)
            if self.vocabOffset:
                self.tokens = (self.tokens + self.vocabOffset).copy() # self.tokens is read-only
        self.seqStarts = list(range(0, len(self.tokens) - self.seqLength - 1, self.seqLength)) # List of positions in self.tokens; at the last position, there is still room for a sequence of length (self.seqLength + 1)
        if self.shuffle: random.shuffle(self.seqStarts) # Shuffle in place
        self.idInFile = 0 # Position in self.seqStarts

    def getNextBatch(self):
        startPos = self.seqStarts[self.idInFile]
        buf = self.tokens[startPos : startPos + self.seqLength + 1] # np.uint16
        if self.shuffleTokens:
            buf = np.random.permutation(buf) # Shuffle tokens within the sequence
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1] # Inputs
        y = buf[1:] # Targets (inputs shifted by 1 token)
        self.idInFile += 1 # Advance position in current file
        if self.idInFile == len(self.seqStarts): # End of file
            self.loadNextFile()
        return x.cuda(), y.cuda()

    def skipBatches(self, nBatches):
        assert nBatches >= 0, f"nBatches ({nBatches}) must be positive"
        for _ in range(nBatches):
            self.idInFile += 1 # Advance position in current file
            if self.idInFile == len(self.seqStarts): # End of file
                self.loadNextFile()

    def rewind(self, nBatches):
        assert nBatches >= 0, f"nBatches ({nBatches}) must be positive"
        self.idInFile = max(0, self.idInFile - nSeqs)

    @property
    def vocabSize(self):
        vocabSize = max(self.tokens) + 1
        if (vocabSize > 128) and (vocabSize % 128): # Not a multiple of 128
            vocabSize = math.ceil(vocabSize / 128) * 128 # Round up to a multiple of 128 for efficiency and to account for the fact that the current loaded file may not use all the tokens; e.g. 50,257 GPT-2 tokens -> rounded to 50,304
        return vocabSize

    def getUniqueTokensAndTopCounts(self, nTokensToRead=math.inf, topK=0): # Get set of unique and most-frequent tokens appearing in any of the data files
        tokenCounts = Counter()
        nTokensRead = 0
        for fileName in self.fileNames:
            tokensTmp = loadBinFile(fileName) # np.array uint16
            tokensTmp = self.applyVocabMapping(tokensTmp)
            tokenCounts.update(tokensTmp) # Accumulate counts
            nTokensRead += len(tokensTmp)
            if nTokensRead > nTokensToRead: break
        tokensUnique = list(tokenCounts.keys()) # Unique tokens = keys of counter
        if topK and (self.vocabMapping is None): # self.vocabMapping not yet defined
            if topK > 0:
                keptTokens = [tok for tok, _ in tokenCounts.most_common(topK)] # Find most frequent tokens
            elif topK < 0:
                minCount = abs(topK) # If the argument is < 0, handle it as the min number of occurrences
                keptTokens = [tok for tok, cnt in tokenCounts.items() if cnt >= minCount] # Keep tokens with count > minCount
            vocabMapping = {oldId: newId for newId, oldId in enumerate(keptTokens)} # Create mapping to use small vocab: original->new token IDs (e.g. 0-99 if keeping top 100)
        else:
            keptTokens, vocabMapping = None, None
        return tokensUnique, keptTokens, vocabMapping

    def applyVocabMapping(self, tokens):
        assert isinstance(tokens, np.ndarray), f"tokens must be a NumPy array, got {type(tokens)}"
        assert tokens.dtype == np.uint16, f"tokens elements must be uint16, got {tokens.dtype}"
        if self.vocabMapping is not None:
            #print(min(tokens), max(tokens), type(tokens), tokens.dtype, tokens.shape) # For debug
            tokens = self.vocabMapping[tokens] # Use lookup table
            tokens = tokens[tokens > 0] # Remove undefined tokens (not in the lookup table)
            tokens = tokens - 1 # The lookup table stores (new token IDs + 1) so that 0 can be used to mean 'undefined'
            #print(min(tokens), max(tokens), type(tokens), tokens.dtype, tokens.shape) # For debug
        return tokens
