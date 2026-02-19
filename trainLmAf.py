# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Damien Teney damien.teney@idiap.ch
# SPDX-License-Identifier: MIT

"""
LM-AFs: Train a transformer language model with learnable activation functions (AFs) in MLP and attention blocks.
"""

import os
import sys
import subprocess
from glob import glob
import time
import math
import functools
import builtins
from datetime import datetime
import argparse
from dataclasses import dataclass, asdict
from itertools import chain
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from dataBin import DataLoaderBin as DataLoader
from activationFunctions import af
from livePlots import LivePlotLoss, LivePlotAct, LivePlotAf

# -----------------------------------------------------------------------------
# Define hyperparameters and default values
@dataclass
class Hyperparameters:
    runName : str = "" # Name of result files
    seed : int = 0 # Random seed
    plotMinAcc : float = 0.0 # Lower limit of Y axis in acc plot
    plotMaxAcc : float = 0.4 # Upper limit of Y axis in acc plot
    plotMinLoss : float = 3.0 # Lower limit of Y axis in loss plot
    plotMaxLoss : float = 15.0 # Upper limit of Y axis in loss plot
    shuffleTrData : int = 1 # 1: shuffle sequences; 2: shuffle tokens within sequences
    tokenDocSep : int = 50256 # <|endoftext|> in GPT-2 tokenizer; set to -1 if unused

    # Architecture
    nLayers : int = 4
    nHeads : int = 4
    dimEmb : int = 256
    tieEmbeddings : bool = True # Share weights between input/output embeddings

    # Optimization
    lr : float = 0.0006
    freezeEmbeddings : int = 0 # 1: freeze wte; 2: freeze wte + lm_head; 3: freewe lm_head; same values but <0: train *only* these (freezing everything else)
    batchSize : int = 1 # Batch size (only implemented as gradient accumulation)
    seqLength : int = 4096 # Sequence length (number of tokens per instance in the batch)
    contextSize : int = 1024 # Actual span of attention mask
    nSteps : int = 20000 # Number of training iterations
    nStepsWarmup : int = 2000 # Number of iterations in LR schedule
    nStepsCooldown : int = 10000
    lrSchedule : str = "trap" # trap (trapezoidal); warmup (linear warmup then constant); cosine (warmup then cosine decay)
    wtDecay : float = 0.0 # L2 weight regularization
    nModels : int = 1 # Use >1 for multi-model training of AFs
    staggeredStarts : int = 2 # For multi-model training; 1: staggered starts with common LR schedule; 2: staggered starts with delayed LR schedule (best)
    sameDataAcrossModels : bool = False # For multi-model training
    sameInitAcrossModels : bool = False # For multi-model training
    adamB1 : float = 0.8
    adamB2 : float = 0.95
    gradientClipping : float = 0

    # Activation functions
    afType : str = "gelu;att-linear" # Baseline; use "spline;att-spline" for fully-learned AFs
    afLayerSpecific : int = 1 # 0: shared across layers; 1: layer-specific; 2: shared across middle layer (not first/last)
    afRange : float = 15 # Max expected absolute value of activations (pre-non-linearities)
    afNAnchors : int = 64 # Number of spline anchors (knots)
    afInit : float = 0.01 # Small positive value, or -1 to initialize as linear function
    afLr : float = 0.1 # Learning rate for the AFs: >0 for Adam, <0 for SGD
    afAdamB1 : float = 0.99 # Hyperparameter for Adam updating the AFs
    afAdamB2 : float = 0.999 # Hyperparameter for Adam updating the AFs

    # Paths and saving
    afFileToLoad : str = ""
    dirResults : str = r"./results-fineweb10B"
    filesTokensTr : str = r"./data-fineweb10B/fineweb_train_*.bin"
    filesTokensVa : str = r"./data-fineweb10B/fineweb_val_*.bin"

    # Evaluation and logging
    nTokensVa : int = 10*1024*1024 # Number of tokens of validation data
    valEvery : int = 500 # Every how many steps to evaluate val loss; 0: only at the end; -1: disabled
    saveModelEvery : int = -1 # Every how many steps to save model; 0: only at the end; -1: disabled
    saveAfEvery : int = 0 # Every how many steps to save AF; 0: only at the end, -1: disabled
    plotEvery : int = 200 # Every how many steps to update plots
    plottingLevel : int = 2 # Plotting level: 0 (none), 1 (loss/acc/AF), 2 (+ activation magnitudes)

    # Computational efficiency
    dtype : str = "float32" # Could be e.g. "bfloat16" for mixed precision
    flexBlockSize : int = 32 # Block size for flex attention; only affects speed/memory usage; large can be faster, reduce if OOM

def parseArgs() -> Hyperparameters:
    parser = argparse.ArgumentParser()
    def boolify(v):
        allowed = ('1', 'true', 'True', '0', 'False', 'false')
        assert v in allowed, f"Invalid boolean string: {v}. Allowed: {allowed}"
        return v in ('1', 'true', 'True')
    [parser.add_argument(f'--{k}', type=boolify if isinstance(v, bool) else type(v), default=v)
        for k, v in asdict(Hyperparameters()).items()]
    return Hyperparameters(**vars(parser.parse_args()))

assert __name__ == "__main__", "This file is only meant as a standalone entry point"
config = parseArgs()

# Split the string afType (e.g. "relu;att-spline") into 2 values for MLP/att
config.afType = config.afType.split(";")
assert len(config.afType) == 2, f"Invalid parameter: {config.afType}"
config.afType[1] = config.afType[1].removeprefix("att-")
 
if ("spline" not in config.afType[0]) and ("spline" not in config.afType[0]): # Non-learnable AFs
    config.afLr = 0
    config.afLayerSpecific = 0
if config.afLr == 0:
    config.saveAfEvery = -1 # Frozen or non-learnable AF: do not save

# Handle config.dtype
try:
    config.dtype = getattr(torch, config.dtype) # Parse string such as "bfloat16" into corresponding torch.dtype
    if not isinstance(config.dtype, torch.dtype): raise AttributeError
except Exception as e:
    raise ValueError(f"Unknown torch dtype: '{config.dtype}'")
if config.dtype == torch.bfloat16:
    torch.set_float32_matmul_precision('medium') # bfloat16
else:
    torch.set_float32_matmul_precision('high') # Allow reduced-precision math if supported by hardware: fp32 on V100, tf32 on RTX3090

assert torch.cuda.is_available()
torch.cuda.set_device('cuda:0')
torch._inductor.config.max_autotune_gemm = 0 # Avoid pytorch warnings
device = torch.cuda.current_device()

# Helper function for debug
def printMagnitudes(x, name):
    if x.numel() > 1:
        print(f"  range=[{x.flatten().min():.2f}, {x.flatten().max():.2f}]  mean={x.flatten().mean():.2f}  meanAbs={x.flatten().abs().mean():.2f}  std={x.flatten().std():.2f}  max|x|={x.flatten().abs().max():.2f} ({name})")
    elif x.numel() == 1:
        print(f"  value=[{x.flatten().max():.2f}]  ({name})")
    else:
        print(f"  empty  ({name})")

# -----------------------------------------------------------------------------
# Wrapper for torch.model to train multiple instances of the same model
class MultiModel(nn.Module):
    def __init__(self, model, nModels=2, sameInit=False):
        # model: base nn.Module to replicate
        # nModels: number of copies
        super().__init__()
        self.nModels = nModels
        self.idActive = 0

        # Create deep copies of the given model
        assert isinstance(model, nn.Module)
        self.models = nn.ModuleList()
        for modelId in range(nModels):
            m = copy.deepcopy(model)
            if not sameInit and (modelId > 0): # Set different initialization for each model: in-place shuffle every set of parameters
                with torch.no_grad():
                    for name, param in m.named_parameters():
                        if ("blockLambdas" not in name): # Parameters that should not be shuffled because their initialization is not random
                            param.copy_(param.view(-1)[torch.randperm(param.numel())].view_as(param)) # Flatten, permute, and reshape back to original shape
            self.models.append(m)

        # Share parameters across models
        for modelId in range(1, nModels): # For all models > 0, replace with pointers to model 0
            for layerId in range(self.models[0].nLayers):
                self.models[modelId].transformer.h[layerId].attn.af = self.models[0].transformer.h[layerId].attn.af
                self.models[modelId].transformer.h[layerId].mlp.af = self.models[0].transformer.h[layerId].mlp.af

    @property
    def activeModel(self):
        return self.models[self.idActive]

    def forward(self, *args, **kwargs):
        return self.activeModel(*args, **kwargs) # Run the active model only

    def train(self, mode=True, idActive=0):
        super().train(mode)
        self.idActive = idActive
        for model in self.models:
            model.train(mode) # Recursive call on every model
        return self

    def eval(self):
        return self.train(False)

    def dispEmptyGrads(self): # Method for debug: display which parameters have empty gradients
        nParams = 0
        nParamsEmpty = 0
        print(f"Parameters with empty gradients:", flush=True)
        for name, param in self.named_parameters():
            nParams += 1
            if param.grad is None:
                nParamsEmpty += 1
                print(f"  {name}")
        print(f"  {nParamsEmpty} / {nParams}\n", flush=True)

    @staticmethod
    def setattrRecursive(module, name, value): # Assign 'value' to nested attribute 'name', e.g. 'layer1.X.Y.weight'
        parts = name.split('.')
        for part in parts[:-1]:
            module = getattr(module, part) # Iterate through children, e.g. layer1.X.Y.weight
        setattr(module, parts[-1], value)

# -----------------------------------------------------------------------------
# Architecture definitions
def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# Wrapper for linear layer that casts the weights to match the input's dtype
class CastedLinear(nn.Linear):
    def __init__(self, dimIn, dimOut, bias=False): # No biases by default
        super().__init__(dimIn, dimOut, bias=bias)
        nn.init.normal_(self.weight, mean=0.0, std=0.02) # Small initial weights
    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))
        else: # No bias
            return F.linear(x, self.weight.to(x.dtype))

# Wrapper for linear layer that casts the input to match the weights' dtype
class CastedLinear2(nn.Linear):
    def __init__(self, dimIn, dimOut, bias=False): # No biases by default
        super().__init__(dimIn, dimOut, bias=bias)
    def forward(self, x):
        if self.bias is not None:
            return F.linear(x.to(self.weight.dtype), self.weight, self.bias)
        else: # No bias
            return F.linear(x.to(self.weight.dtype), self.weight)

class ScalarAffine(nn.Module):
    def __init__(self, dim, initWeight=1.0, initBias=0.0):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.tensor(initWeight))
        self.bias = nn.Parameter(torch.tensor(initBias))
    def forward(self, x):
        # Need to expand the parameters to be compatible with torch.compile
        w = torch.eye(self.dim, device=x.device, dtype=x.dtype) * self.weight
        b = self.bias.expand(self.dim)
        return F.linear(x, w, b)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.freqsInv = None
        self.seqLength_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seqLength = x.shape[1]
        if seqLength != self.seqLength_cached:
            self.freqsInv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seqLength_cached = seqLength
            t = torch.arange(seqLength, device=x.device).type_as(self.freqsInv)
            freqs = torch.outer(t, self.freqsInv)
            self.cos_cached = freqs.cos().to(config.dtype)
            self.sin_cached = freqs.sin().to(config.dtype)
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        assert x.ndim == 4 # Multihead attention
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

flex_attention = torch.compile(flex_attention, dynamic=False)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dimEmb
        self.nHeads = config.nHeads
        assert (self.dim % self.nHeads) == 0, f"dimEmb={self.dim} must be a multiple of nHeads={self.nHeads}"
        if config.flexBlockSize is None:
            self.flexKernelOptions = None
        else:
            #if config.flexBlockSize > config.dimEmb: config.flexBlockSize = config.dimEmb
            self.flexKernelOptions = {
                "BLOCK_M": config.flexBlockSize, "BLOCK_N": config.flexBlockSize, # Tile sizes for query/key block in forward
                "BLOCK_M1": config.flexBlockSize, "BLOCK_N1": config.flexBlockSize, # Tile sizes for gradients wrt Q/K
                "BLOCK_M2": config.flexBlockSize, "BLOCK_N2": config.flexBlockSize, # Tile sizes for gradients wrt V
            }

        if config.afType[1] == "none": # Standard case
            self.af = None
        else: # AF before QKV
            self.projAtt = ScalarAffine(self.dim) # Additional small learned scaling (helps a little bit
            self.af = af(config.afType[1], config.afRange, config.afNAnchors, config.afInit, self.dim, config.dtype)

            # Initialize monitoring of activation magnitudes
            if config.plottingLevel >= 2: # Will store activation magnitudes
                self.register_buffer("actMonitor2", torch.tensor(0.0, dtype=torch.float32))
                self.register_buffer("actMonitor3", torch.tensor(0.0, dtype=torch.float32))

        # Initialize parameters
        self.projQ = CastedLinear(self.dim, self.dim, bias=False)
        self.projK = CastedLinear(self.dim, self.dim, bias=False)
        self.projV = CastedLinear(self.dim, self.dim, bias=False)
        self.lambdas = nn.Parameter(torch.tensor(0.5)) # Value residual lambda
        self.rotaryEmbedding = RotaryEmbedding(dim=self.dim // self.nHeads)
        self.projOut = CastedLinear(self.dim, self.dim, bias=False) # Output projection
        if config.freezeEmbeddings >= 0: # Init attention output with 0s (on next line) except when freezeEmbeddings < 0 (whole model frozen, except embeddings)
            with torch.no_grad(): self.projOut.weight.data.zero_() # Zero init of attention last layer

    def forward(self, x, v1, mask):
        batchSize, seqLength = x.size(0), x.size(1)
        assert batchSize == 1, "Must use batch size = 1 for FlexAttention"

        if config.afType[1] != "none": # AF before QKV
            x0 = x # Make copy for residual connection
            x = self.projAtt(x) # Learned affine projection
            if (config.plottingLevel >= 2): # Monitor pre-AF magnitudes
                self.actMonitor2.fill_(x.abs().mean().detach())
                self.actMonitor3.fill_(x.abs().max().detach())
            x = self.af(x)
            x = x0 + x # Residual connection (seems slightly better)

        q = self.projQ(x)
        k = self.projK(x)
        v = self.projV(x)
        q = q.view(batchSize, seqLength, self.nHeads, -1)
        k = k.view(batchSize, seqLength, self.nHeads, -1)
        v = v.view(batchSize, seqLength, self.nHeads, -1)
        if v1 is None: v1 = v # First block, return reference to v to be accessed by subsequent blocks
        v = (1 - self.lambdas) * v + self.lambdas * v1.view_as(v) # Combine current v with v1 from first block
        q, k = norm(q), norm(k) # QK norm
        q, k = self.rotaryEmbedding(q), self.rotaryEmbedding(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=mask,
            kernel_options=self.flexKernelOptions
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # Re-assemble all head outputs side by side
        y = self.projOut(y)
        return y, v1

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dimEmb
        self.fc   = CastedLinear(self.dim, 4 * self.dim, bias=False)
        self.proj = CastedLinear(4 * self.dim, self.dim, bias=False)
        if config.freezeEmbeddings >= 0: # Do not init with 0 when freezeEmbeddings < 0 (whole model frozen except embeddings)
            self.proj.weight.data.zero_() # Zero init of MLP last layer

        self.af = af(config.afType[0], config.afRange, config.afNAnchors, config.afInit, 4 * self.dim, config.dtype)

        # Initialize monitoring of activation magnitudes and normalization
        if config.plottingLevel >= 2: # Will store activation magnitudes
            self.register_buffer("actMonitor2", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer("actMonitor3", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = self.fc(x)

        # Monitoring of activation magnitudes
        if config.plottingLevel >= 2: # Monitor pre-AF magnitudes
            self.actMonitor2.fill_(x.abs().mean().detach())
            self.actMonitor3.fill_(x.abs().max().detach())

        x = self.af(x)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.blockLambdas = nn.Parameter(torch.tensor([1., 0.])) # U-Net

        # Normalization with affine parameters
        #self.normAtt = nn.LayerNorm(config.dimEmb, eps=1e-5)
        #self.normMlp  = nn.LayerNorm(config.dimEmb, eps=1e-5)
        self.normAtt = nn.RMSNorm(config.dimEmb, eps=1e-5)
        self.normMlp  = nn.RMSNorm(config.dimEmb, eps=1e-5)

    def forward(self, x, v1, x0, mask):
        # U-Net
        x = self.blockLambdas[0] * x + self.blockLambdas[1] * x0
        xTmp, v1 = self.attn(self.normAtt(x), v1, mask)
        x = x + xTmp
        xTmp = self.mlp(self.normMlp(x))
        x = x + xTmp
        return x, v1

        # Vanilla transformer (in case you don't want the U-Net architecture)
        '''
        x = x + self.attn(self.normAtt(x), None, mask)
        x = x + self.mlp(self.normMlp(x))
        return x
        '''

# -----------------------------------------------------------------------------
create_block_mask = torch.compile(create_block_mask, dynamic=False)

class Gpt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.contextSize = torch.tensor(config.contextSize, dtype=torch.int, device='cuda')
        self.tokenDocSep = config.tokenDocSep
        self.nLayers = config.nLayers
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocabSize, config.dimEmb),
            h = nn.ModuleList([Block(config) for lyId in range(self.nLayers)]),
        ))

        # U-net architecture
        self.nLayersEncoder = (config.nLayers + 1) // 2 # Half of the layers for encoder (ceil: 1 if nLayers = 1)
        self.nLayersDecoder = (config.nLayers - self.nLayersEncoder) # Remaining for decoder (0 if nLayers = 1)
        self.skipWeights = nn.Parameter(torch.ones(self.nLayersDecoder)) # Learnable skip connection weights for decoder layers

        self.normWte = nn.LayerNorm(config.dimEmb, eps=1e-5) # Normalization with affine parameters
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02) # Small initial weights

        if config.tieEmbeddings:
            self.lm_head = CastedLinear2(config.dimEmb, config.vocabSize, bias=False)
            assert self.lm_head.weight.shape == self.transformer.wte.weight.shape, f"Non-matching shapes: {self.lm_head.weight.shape} and {self.transformer.wte.weight.shape}" # Sanity check before sharing (vocabSize x dimEmb)
            self.lm_head.weight = self.transformer.wte.weight # Replace the weights of the CastedLinear2() layer with a reference to a shared tensor
        else:
            self.lm_head = CastedLinear(config.dimEmb, config.vocabSize, bias=False)
            if (config.freezeEmbeddings == 2) or (config.freezeEmbeddings == 3) or (config.freezeEmbeddings == -1): # Freezing lm_head: cannot init with 0s
                with torch.no_grad(): self.lm_head.weight.copy_(self.transformer.wte.weight[torch.randperm(config.vocabSize)]) # Init with shuffled values from wte (ensure same distribution of magnitudes) (seems usually)
            else: # Standard case (lm_head is trained)
                with torch.no_grad(): self.lm_head.weight.data.zero_() # Zero init of wte (seems best)

        if (config.afLayerSpecific == 0) and (config.nLayers > 1): # Share AFs across layers
            for layerId in range(1, self.nLayers): # For every layer > 0, replace parameters with pointer to layer 0
                self.transformer.h[layerId].attn.af = self.transformer.h[0].attn.af
                self.transformer.h[layerId].mlp.af = self.transformer.h[0].mlp.af
        elif (config.afLayerSpecific == 2) and (config.nLayers >= 4): # Share AFs across middle layers (not first/last)
            for layerId in range(2, self.nLayers-1): # For every layer > 1, replace parameters with pointer to layer 1
                self.transformer.h[layerId].attn.af = self.transformer.h[1].attn.af
                self.transformer.h[layerId].mlp.af = self.transformer.h[1].mlp.af

    def forward(self, tokens, targets):
        #print(f"tokens:  {tokens.cpu().float().numpy().reshape(-1)}") # For debug
        #print(f"targets: {targets.cpu().float().numpy().reshape(-1)}") # For debug

        docIds = (tokens == self.tokenDocSep).cumsum(0) # Find end-of-document tokens
        def makeAttMask(batchSize, nHeads, q, kv):
            maskCausal = (q >= kv)
            maskDocument = (docIds[q] == docIds[kv]) # Prevent attention across document
            maskWindow = ((q - kv) < self.contextSize) # Local sliding window
            return maskCausal & maskDocument & maskWindow
        seqLength = len(tokens)
        assert self.contextSize <= seqLength, f"contextSize={self.contextSize}, seqLength={seqLength}"
        attMask = create_block_mask(makeAttMask, None, None, seqLength, seqLength, device="cuda", _compile=True)

        x = tokens.unsqueeze(0) # Add batch dimension (1, seqLength)
        x = self.transformer.wte(x) # Token embeddings (batchSize, seqLength, dimEmb)
        x = self.normWte(x)

        # U-Net
        x0 = x
        v1 = None
        skipConnections = [] # Will store activations for U-Net skip connections
        for i in range(self.nLayersEncoder): # Encoder (first half of the blocks)
            x, v1 = self.transformer.h[i](x, v1, x0, attMask)
            skipConnections.append(x)
        for i in range(self.nLayersDecoder): # Decoder (second half of the blocks)
            x = x + self.skipWeights[i] * skipConnections.pop() # Weighted skip connections
            x, v1 = self.transformer.h[self.nLayersEncoder + i](x, v1, x0, attMask)

        # Vanilla transformer (in case you don't want the U-Net architecture)
        #for i in range(self.nLayers): x = self.transformer.h[i](x, None, None, attMask)

        logits = self.lm_head(x)
        logits = logits.float() # bfloat16 to float32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # Assume 'targets' is 'tokens' shifted by 1 token
        acc = computeAcc(logits, targets)
        return loss, acc

def computeAcc(logits, targets):
    with torch.no_grad():
        pred = logits.argmax(dim=-1) # Argmax over vocab
        targets = targets.reshape(pred.shape) # (batchSize, nTokensToKeep), 'reshape' is necessary because 'targets' may have no batchSize dimension
        acc = (pred == targets).float().mean() # Token-wise accuracy; average over batch and tokens
    return acc

def countParameterizedAfs(model): # Count unique AFs with isParameterized == True
    if hasattr(model, "models"): model = model.models[0] # Handle 'MultiModel' wrapper
    uniqueAfs = set()
    for m in model.modules():
        if isinstance(m, af) and m.isParameterized: uniqueAfs.add(m) # Add to set (unique elements)
    return len(uniqueAfs)

# -----------------------------------------------------------------------------
def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# -----------------------------------------------------------------------------
# Init output files
runName = datetime.now().strftime("%y%m%d%H%M") if config.runName == "" else config.runName # Use current date/time if config.runName is empty
os.makedirs(config.dirResults, exist_ok=True) # Create directory to save the results
def getFileNameCsvAf(step):
    return os.path.join(config.dirResults, f'ptAf-{runName}-step{step:04d}.csv')
def getFileNamePtAf(step):
    return os.path.join(config.dirResults, f'ptAf-{runName}-step{step:04d}.pt')
def getFileNamePt(step):
    return os.path.join(config.dirResults, f'pt-{runName}-step{step:04d}.pt')

def saveAfCsv(model, fileName):
    if hasattr(model, "_orig_mod"): model = model._orig_mod # With torch.compile
    afs = [] # Will contain all MLP AFs, then all att AFs
    for layerId in range(model.nLayers): afs.append(model.transformer.h[layerId].mlp.af)
    for layerId in range(model.nLayers): afs.append(model.transformer.h[layerId].attn.af)
    with open(fileName, 'w') as fileTmp: # Write values separated by commas, one tensor per line
        for af in afs:
            if (af is not None) and af.isParameterized:
                afAnchors = af.afAnchors.cpu().tolist()
                afVals = af.afVals.detach().cpu().float().numpy().reshape(-1) # Flatten if 2D (splinePerDim)
                fileTmp.write(','.join(map(str, afAnchors)) + '\n')
                fileTmp.write(','.join(map(str, afVals)) + '\n')

def saveAfPt(config, model, fileName):
    if hasattr(model, "_orig_mod"): model = model._orig_mod # With torch.compile
    afValsMlp, afValsAtt = [], []
    afAnchors = None
    for layerId in range(model.nLayers):
        af = model.transformer.h[layerId].mlp.af
        if (af is not None) and (af.afVals is not None):
            afValsMlp.append(af.afVals)
            afAnchors = af.afAnchors
        af = model.transformer.h[layerId].attn.af
        if (af is not None) and (af.afVals is not None):
            afValsAtt.append(af.afVals)
            afAnchors = af.afAnchors
    assert afAnchors is not None
    if config.afType[0].startswith("spline"): assert len(afValsMlp) == config.nLayers
    if config.afType[1].startswith("spline"): assert len(afValsAtt) == config.nLayers
    torch.save(dict(config=config, afAnchors=afAnchors, afValsMlp=afValsMlp, afValsAtt=afValsAtt), getFileNamePtAf(step+1))

# Init log file to catch stdout/stderr
fileNameLog = os.path.join(config.dirResults, f'log-{runName}.txt')
if os.path.exists(fileNameLog): # Do not overwrite existing results
    with open(fileNameLog, "r", encoding="utf-8") as f:
        # Allow overwriting partial results, e.g. interrupted before reaching 'config.nSteps' training steps
        if any(line.startswith(f"Step {config.nSteps}/{config.nSteps} ") for line in f):
            print(f"Stopping: log file exists and contains >= {config.nSteps} training steps: {fileNameLog}")
            sys.exit(0) # Stop here
fileLog = open(fileNameLog, "a", encoding="utf-8", buffering=1) # Create new log file (or append to existing file with partial results)
try:
    import fcntl # Not available on Windows
    fcntl.flock(fileLog, fcntl.LOCK_EX | fcntl.LOCK_NB) # Make sure no two processes are writing to the same file (when running parallel experiments)
except (ModuleNotFoundError, ImportError):
    print("Not on Linux: no file lock")
except BlockingIOError:
    raise RuntimeError(f"Log file is already open by another process: {fileNameLog}")
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        if isinstance(data, bytes): data = data.decode(errors="replace")
        for f in self.files: f.write(data)
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, fileLog)
sys.stderr = Tee(sys.stderr, fileLog)
builtins.print = functools.partial(builtins.print, flush=True) # use Flush=True by default in all calls to print()

# Display hyperparameters, pytorch version, nvidia-smi
print("Hyperparameters:")
for k, v in asdict(config).items():
    print(f'  {k} = {"\"" + v + "\"" if isinstance(v, str) else v}') # Enclose strings in double quotes
print("="*100)
print(f"pytorch {torch.version.__version__}, CUDA {torch.version.cuda}")
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(f'{result.stdout}')

# -----------------------------------------------------------------------------
# Init plots
livePlots = {}
if (config.plottingLevel >= 1):
    livePlots["loss"] = LivePlotLoss("tr-" + runName, config.plotMinAcc, config.plotMaxAcc, config.plotMinLoss, config.plotMaxLoss, config.nModels, config.nSteps)
    livePlots["af"] = None # Will be created lated
if (config.plottingLevel >= 2):
    livePlots["act"] = None # Will be created lated

# Determine values on which to evaluate the AF for plotting
if config.afType[0].startswith("spline") or config.afType[1].startswith("spline"): # Learned AFs
    stepSize = (2 * config.afRange) / (config.afNAnchors - 1)
    afRangeExtended = config.afRange + math.ceil(0.1 * config.afNAnchors) * stepSize # A little bit beyond the range of the learned anchors
    xs = torch.arange(-afRangeExtended, afRangeExtended + stepSize, stepSize/5, dtype=torch.float32).to(device='cuda')
else: # Fixed AFs
    xs = torch.arange(-10, 10, 0.05, dtype=torch.float32).to(device='cuda') # Default small range

# Function to plot current AFs: evaluate the function, then create/update the figure
def plotAfs(config, model, xs, step):
    if config.plottingLevel == 0: return
    with torch.no_grad():
        ys = [model.transformer.h[layerId].mlp.af.forward(xs).cpu().numpy() for layerId in range(config.nLayers if config.afLayerSpecific else 1)]
        if model.transformer.h[0].attn.af is not None:
            ys = ys + [model.transformer.h[layerId].attn.af.forward(xs).cpu().numpy() for layerId in range(config.nLayers if config.afLayerSpecific else 1)]
    if livePlots["af"] is None: # First iteration: create the plot
        nTiles = len(ys)
        livePlots["af"] = LivePlotAf("af-" + runName, nTiles, config.afRange, config.afNAnchors)
    livePlots["af"].plot(step, xs.cpu().numpy(), ys)

# -----------------------------------------------------------------------------
# Load data (tokens)
assert config.filesTokensTr.endswith(".bin")
if (config.nModels == 1) or config.sameDataAcrossModels: # Only 1 DataLoader
    dataLoaderTr = [DataLoader(config.filesTokensTr, config.seqLength, initFileId=config.seed if config.shuffleTrData else 0, shuffle=config.shuffleTrData, shuffleTokens=(config.shuffleTrData == 2))]
else: # When nModels > 1, one dataLoaderTr per model
    dataLoaderTr = [DataLoader(config.filesTokensTr, config.seqLength, initFileId=(modelId + config.seed) if config.shuffleTrData else modelId, shuffle=config.shuffleTrData, shuffleTokens=(config.shuffleTrData == 2)) for modelId in range(config.nModels)]
print(f"Tr data: {dataLoaderTr[0].nTokens:,} tokens, {len(dataLoaderTr[0].fileNames)} files ({config.filesTokensTr})")
config.vocabSize = dataLoaderTr[0].vocabSize

if config.filesTokensVa:
    dataLoaderVa = DataLoader(config.filesTokensVa, config.seqLength, trainingData=False)
    print(f"Va data: {dataLoaderVa.nTokens:,} tokens, {len(dataLoaderVa.fileNames)} files ({config.filesTokensVa})")
    nStepsVa = max(1, abs(config.nTokensVa) // config.seqLength)
    print(f"Number of validation batches: {nStepsVa:,} ({config.nTokensVa:,} tokens, {config.seqLength:,} tokens/sequence)")
    assert dataLoaderVa.vocabSize == config.vocabSize
else:
    dataLoaderVa = None

print("="*100)

# -----------------------------------------------------------------------------
# Count tokens
nTokensTr = config.nSteps * config.batchSize * config.seqLength
print(f"Projected number of training tokens:")
print(f"  {config.nSteps:,} steps * {config.batchSize:,} batchSize * {config.seqLength:,} seqLength")
print(f"  = {nTokensTr:,} tokens")
print(f"  = {(nTokensTr/dataLoaderTr[0].nTokens):.2f} epochs")
print(f"Vocabulary size: {config.vocabSize:,}")
print("="*100)

# -----------------------------------------------------------------------------
# Create model
setSeed(config.seed)
model = Gpt(config)

# -----------------------------------------------------------------------------
# Load learned AF
if config.afFileToLoad:
    assert config.afType[0].startswith("spline") or config.afType[1].startswith("spline") # Learnable AF
    print(f"Loading: {config.afFileToLoad}")
    loaded = torch.load(config.afFileToLoad, weights_only=False, map_location='cpu'); assert isinstance(loaded, dict)
    if 'afVals' in loaded: # Load legacy file
        afAnchors, afValsTmp, config2 = loaded['afAnchors'], loaded['afVals'], loaded['config'] # Rename loaded variables
        if config.afType[0].startswith("spline"): afValsMlp = afValsTmp
        if config.afType[1].startswith("spline"): afValsAtt = afValsTmp
    else:
        afAnchors, afValsMlp, afValsAtt, config2 = loaded['afAnchors'], loaded['afValsMlp'], loaded['afValsAtt'], loaded['config'] # Rename loaded variables
    config.afRange = config2.afRange
    config.afNAnchors = config2.afNAnchors
    fineTune = (config.afLr != 0)
    if fineTune: print(f"Fine-tuning AF, LR={config.afLr}") # Always worse than frozen AF in our experience
    if config.afType[0].startswith("spline"):
        if len(afValsMlp) != config.nLayers:
            assert not config.afLayerSpecific
            #afValsMlp = [afValsMlp[0] for _ in range(config.nLayers)] # Repeat the shared parameters for each layer
            afValsMlp = [nn.Parameter(afValsMlp[0].data.clone()) for _ in range(config.nLayers)] # Clone the shared parameters for each layer
        assert len(afValsMlp) == config.nLayers, f"len(afValsMlp)={len(afValsMlp)}, nLayers={config.nLayers}"
        for layerId in range(config.nLayers): model.transformer.h[layerId].mlp.af.loadTrainedAf(config2.afType[0], afAnchors, afValsMlp[layerId], fineTune)
    if config.afType[1].startswith("spline"):
        if len(afValsAtt) != config.nLayers:
            assert not config.afLayerSpecific
            #afValsAtt = [afValsAtt[0] for _ in range(config.nLayers)] # Repeat the shared parameters for each layer
            afValsAtt = [nn.Parameter(afValsMlp[0].data.clone()) for _ in range(config.nLayers)] # Clone the shared parameters for each layer
        assert len(afValsAtt) == config.nLayers, f"len(afValsAtt)={len(afValsAtt)}, nLayers={config.nLayers}"
        for layerId in range(config.nLayers): model.transformer.h[layerId].attn.af.loadTrainedAf(config2.afType[1], afAnchors, afValsAtt[layerId], fineTune)

# -----------------------------------------------------------------------------
# Move model to GPU and set precision
model = model.cuda()
model = model.to(config.dtype) # Set reduced precision if config.dtype is e.g. bfloat16
for m in model.modules():
    if isinstance(m, CastedLinear): m.float() # Prevent linear layers from using reduced precision
model = torch.compile(model) # Compile model (comment for easier debug)

plotAfs(config, model, xs, 0) # Plot initial AF
livePlots["af"].flushFig() # Make sure the figure is drawn

# Freeze parts of the model
if config.freezeEmbeddings > 0: # Freeze embeddings
    if config.tieEmbeddings and (config.freezeEmbeddings == 1): raise ValueError(f"Cannot freeze 'wte' without freezing 'lm_head' (tieEmbeddings={config.tieEmbeddings})")
    if config.tieEmbeddings and (config.freezeEmbeddings == 2): raise ValueError(f"Cannot freeze 'lm_head' without freezing 'wte' (tieEmbeddings={config.tieEmbeddings})")
    done = [False, False]
    for name, module in model.named_modules():
        if name.endswith("wte") and ((config.freezeEmbeddings == 1) or (config.freezeEmbeddings == 3)):
            done[0] = True; print(f"Freezing: {name}")
            for param in module.parameters(): param.requires_grad = False
        if name.endswith("lm_head") and ((config.freezeEmbeddings == 2) or (config.freezeEmbeddings == 3)):
            done[1] = True; print(f"Freezing: {name}")
            for param in module.parameters(): param.requires_grad = False
    if (config.freezeEmbeddings == 1) or (config.freezeEmbeddings == 3): assert done[0]
    if (config.freezeEmbeddings == 2) or (config.freezeEmbeddings == 3): assert done[1]

elif config.freezeEmbeddings < 0: # Freeze whole model except embeddings
    if config.tieEmbeddings and (config.freezeEmbeddings == -1): raise ValueError(f"Cannot train 'wte' without also training 'lm_head' (tieEmbeddings={config.tieEmbeddings})")
    if config.tieEmbeddings and (config.freezeEmbeddings == -2): raise ValueError(f"Cannot train 'lm_head' without also training 'wte' (tieEmbeddings={config.tieEmbeddings})")
    done = [False, False]
    for name, module in model.named_modules():
        if name.endswith("wte") and ((config.freezeEmbeddings == -1) or (config.freezeEmbeddings == -3)):
            done[0] = True; print(f"Not freezing: {name}")
            for param in module.parameters(): param.requires_grad = True # Do NOT freeze
        elif name.endswith("lm_head") and ((config.freezeEmbeddings == -2) or (config.freezeEmbeddings == -3)):
            done[1] = True; print(f"Not freezing: {name}")
            for param in module.parameters(): param.requires_grad = True # Do NOT freeze
        else:
            for name, param in module.named_parameters():
                if ("afVals" in name) and (config.afLr != 0):
                    param.requires_grad = True
                else:
                    param.requires_grad = False # Freeze everything else
            #for param in module.parameters(): param.requires_grad = False # Freeze everything else
    if (config.freezeEmbeddings == -1) or (config.freezeEmbeddings == -3): assert done[0]
    if (config.freezeEmbeddings == -2) or (config.freezeEmbeddings == -3): assert done[1]

# -----------------------------------------------------------------------------
# Count parameters and tokens
nParamsFrozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
nParamsTrainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"Number of model parameters (frozen): {nParamsFrozen:,}")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"  {name:50s} {param.numel():,}")
print(f"Number of model parameters (trainable): {nParamsTrainable:,}")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name:50s} {param.numel():,}")
print(f"{nTokensTr/nParamsTrainable:.1f} tokens/parameter")
print("="*100)

# -----------------------------------------------------------------------------
# Multi-model training
if config.nModels > 1:
    model = MultiModel(model, nModels=config.nModels, sameInit=config.sameInitAcrossModels)
    if config.staggeredStarts:
        startStepPerModel = np.round(np.linspace(0, config.nSteps, config.nModels + 1)).astype(int) # Start step of each model

# -----------------------------------------------------------------------------
# Init optimizers
optimizers = []
if (config.nModels == 1) or not config.staggeredStarts:
    paramNonAf = [param for name, param in model.named_parameters() if ("afVals" not in name) and param.requires_grad]
    paramAf    = [param for name, param in model.named_parameters() if ("afVals"     in name) and param.requires_grad]
    optimizers.append(torch.optim.Adam(params=paramNonAf, lr=config.lr, betas=(config.adamB1, config.adamB2), fused=True, weight_decay=config.wtDecay, decoupled_weight_decay=True))
else: # (config.nModels > 1) and config.staggeredStarts: different optimizer for each model
    for m in range(config.nModels):
        paramNonAf = [param for name, param in model.models[m].named_parameters() if ("afVals" not in name) and param.requires_grad]
        paramAf    = [param for name, param in model.models[m].named_parameters() if ("afVals"     in name) and param.requires_grad] # Parameters shared across models
        optimizers.append(torch.optim.Adam(params=paramNonAf, lr=config.lr, betas=(config.adamB1, config.adamB2), fused=True, weight_decay=config.wtDecay, decoupled_weight_decay=True))

# Add optimizer for AF
if config.afLr > 0: # Optimize AF with Adam
    optimizers.append(torch.optim.Adam(paramAf, lr=config.afLr, betas=(config.afAdamB1, config.afAdamB2), fused=True, decoupled_weight_decay=True))
elif config.afLr < 0: # Optimize AF with SGD
    optimizers.append(torch.optim.SGD(paramAf, lr=abs(config.afLr), momentum=0))

# Debug display: list of optimizers/parameters
# (before defining LR schedule, because get('lr') will return the scheduled LR at step 0, e.g. smaller if there is a warmup)
id2name = {} # Reverse lookup: id(param) -> name
for name, param in model.named_parameters():
    id2name[id(param)] = name
for i, opt in enumerate(optimizers):
    print(f"Optimizer #{i}/{len(optimizers)-1}: {type(opt).__name__}")
    for g, group in enumerate(opt.param_groups):
        print(f"  Param group {g}: lr={group.get('lr')}, weight_decay={group.get('weight_decay')}")
        for param in group["params"]:
            name = id2name.get(id(param), "<UNKNOWN PARAM>")
            print(f"  {name}")
print("="*100)

# Define LR schedule
assert all(n >= 0 for n in (config.nStepsWarmup, config.nStepsCooldown))
assert (config.nStepsWarmup + config.nStepsCooldown) < config.nSteps # Check there are *some* steps of constant LR
if config.lrSchedule == "trap":
    def getLrFactor(step): # Trapezoidal LR schedule
        assert step <= config.nSteps
        if   step < 0:                                       return 0.0 # Training has not started yet
        elif step < config.nStepsWarmup:                     return (step + 1) / config.nStepsWarmup # Linear warmup
        elif step < (config.nSteps - config.nStepsCooldown): return 1.0 # Constant LR
        elif step < (config.nSteps):                         return (config.nSteps - step) / config.nStepsCooldown # Linear cooldown
        else:                                                return 0.05 # Extended cooldown at small fixed LR
elif config.lrSchedule == "warmup":
    def getLrFactor(step): # Warmup then constant
        assert step <= config.nSteps
        if   step < 0:                   return 0.0 # Training has not started yet
        elif step < config.nStepsWarmup: return (step + 1) / config.nStepsWarmup # Linear warmup
        else:                            return 1.0
elif config.lrSchedule == "cos":
    def getLrFactor(step): # Cosine LR schedule
        minFactor = 5e-4
        assert step <= config.nSteps
        if   step < 0:                   return 0.0 # Training has not started yet
        elif step < config.nStepsWarmup: return (step + 1) / config.nStepsWarmup # Linear warmup
        else:                            return minFactor + 0.5 * (1 - minFactor) * (1 + math.cos(math.pi * (step - config.nStepsWarmup) / (config.nSteps - config.nStepsWarmup))) # Cosine decay
else:
    raise ValueError(f"Unknown lrSchedule: '{config.lrSchedule}'")
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, getLrFactor) for opt in optimizers] # Adjusts LR according to multiplier given by getLrFactor()

# -----------------------------------------------------------------------------
# Training loop
setSeed(config.seed)
lossTrToPlot = np.zeros(config.nModels, dtype=float)
accTrToPlot = np.zeros(config.nModels, dtype=float)
torch.cuda.synchronize()
timeSpentTraining = 0 # In seconds
timeStartOverall = time.time(); timeLast = 0; stepLast = -1 # Start clock
if (((config.nModels == 1) and not hasattr(model, "_orig_mod")) or
    ((config.nModels  > 1) and not hasattr(model.models[0], "_orig_mod"))):
    print("!!! WARNING: running non-compiled model (slower) !!!")
for step in range(config.nSteps): # 0 to (nSteps-1)
    isLastStep = (step == (config.nSteps - 1))
    torch.cuda.synchronize(); timeStartCurrentStep = time.time() # Start clock
  
    # -----------------------------------------------------------------------------
    # Training
    model.zero_grad(set_to_none=True)
    lossTr = np.zeros(config.nModels, dtype=float)
    accTr = np.zeros(config.nModels, dtype=float)
    nModelsTrained = 0
    for modelId in range(config.nModels):
        if (config.nModels > 1) and config.staggeredStarts and (step < startStepPerModel[modelId]): continue # Skip models (forward and backward) before their staggered start (optional but faster)
        if (modelId > 0) and config.sameDataAcrossModels:
            dataLoaderTr[0].rewind(config.batchSize) # Will get the same data again
        if config.nModels > 1:
            model.train(idActive=modelId)  # Set training mode and advance to another model
        else:
            model.train() # Set training mode
        nModelsTrained += 1 # Count the number of accumulated gradients (variable because of staggered starts)
        for accumStep in range(0, config.batchSize): # For each gradient accumulation step
            if (config.nModels <= 1) or config.sameDataAcrossModels: # Only 1 dataLoaderTr
                assert len(dataLoaderTr) == 1, f"len(dataLoaderTr)={len(dataLoaderTr)}, {dataLoaderTr}"
                dataLoaderIdToUse = 0
            else: # Different dataLoaderTr per model
                assert len(dataLoaderTr) == config.nModels, f"len(dataLoaderTr)={len(dataLoaderTr)}, {dataLoaderTr}"
                dataLoaderIdToUse = modelId
            xTr, yTr = dataLoaderTr[dataLoaderIdToUse].getNextBatch()
            lossTrTmp, accTrTmp = model(xTr, yTr) # Model evaluation
            lossTrTmp.backward() # Accumulate gradients
            lossTr[modelId] += lossTrTmp.detach().cpu().item() # Accumulate statistics (only relevant for logging/plotting)
            accTr[modelId]  += accTrTmp.detach().cpu().item()
    lossTr /= config.batchSize; accTr /= config.batchSize # Average statistics (only relevant for logging/plotting)

    # Average accumulated gradients
    assert (config.batchSize >= 1) and (nModelsTrained >= 1)
    if (config.batchSize > 1) or (nModelsTrained > 1):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if "afVals" in name: # Parameters shared across models (accumulated over batchSize * nModels)
                    param.grad /= (nModelsTrained * config.batchSize) # Average over accumulations and models (gradients were accumulated as many times as there are models)
                else: # Parameters NOT shared across models (accumulated over batchSize)
                    param.grad /= config.batchSize

    if config.plottingLevel >= 3: # Print weight magnitudes (before gradient step)
        if (step == 5) or ((config.plotEvery > 0) and (step % config.plotEvery) == 0) or isLastStep:
            print("-"*100)
            with torch.no_grad(): [printMagnitudes(param, name) for name, param in model.named_parameters()]

    if config.gradientClipping:
        assert config.gradientClipping > 0
        nn.utils.clip_grad_norm_(model.parameters(), config.gradientClipping)

    #if config.nModels > 1: model.dispEmptyGrads() # Debug: check for empty gradients

    # -----------------------------------------------------------------------------
    # Gradient step
    if (config.nModels > 1) and config.staggeredStarts:
        assert len(optimizers) == (config.nModels + 1)
        for i in range(len(optimizers)):
            if i == config.nModels: # AF: update at every step
                optimizers[i].step(), schedulers[i].step()
            else: # i < config.nModels
                if (step >= startStepPerModel[i]) or (config.staggeredStarts < 1):
                    optimizers[i].step()
                if (step >= startStepPerModel[i]) or (config.staggeredStarts < 2): # When staggeredStarts=1, advance scheduler even if model was not updated (will trigger a PyTorch warning)
                    schedulers[i].step()
    else: # Standard simultaneous optimization of weights and AF at every step
        for opt, sched in zip(optimizers, schedulers): opt.step(), sched.step() # Step optimizers and schedulers

    # -----------------------------------------------------------------------------
    # Pause the clock (do not include evaluation in timing)
    torch.cuda.synchronize()
    timeSpentTraining += (time.time() - timeStartCurrentStep)
    timeSpentTotal = (time.time() - timeStartOverall) # Include including evaluation

    # -----------------------------------------------------------------------------
    # Evaluation on val/test data
    if (step == 5) or (step == 10) or (step == 20) or isLastStep or \
       ((config.valEvery > 0) and (step % config.valEvery == 0)) or \
       ((config.plotEvery > 0) and (step % config.plotEvery) == 0):
        if config.nModels > 1:
            modelEval = model.models[0]
            #modelEval = model.models[-1] # For debug, should be just as good as model.models[0]
        else: # Standard case
            modelEval = model # Just a pointer (with a different name) to the model
        modelEval.eval() # Set eval model

    if isLastStep or ((config.valEvery > 0) and (step % config.valEvery == 0)) and (dataLoaderVa is not None):
        dataLoaderVa.reset()
        lossVa = 0.0; accVa = 0.0
        for i in range(nStepsVa):
            with torch.no_grad():
                xVa, yVa = dataLoaderVa.getNextBatch()
                if xVa is None: # Trying to get more batches than available in the dataLoader
                    print(f"Reducing nStepsVa from {nStepsVa} to {i}")
                    nStepsVa = i # Number of va batches actually available
                    break
                lossVaTmp, accVaTmp = modelEval(xVa, yVa) # Model evaluation
                lossVa += lossVaTmp.cpu().item(); accVa += accVaTmp.cpu().item() # Accumulate
        lossVa /= nStepsVa; accVa /= nStepsVa # Average
        print(f'Step {step+1}/{config.nSteps}   lossVa {lossVa:.4f}   accVa {100*accVa:.1f}')
        if (config.plottingLevel >= 1):
            livePlots["loss"].plot(step+1, None, lossVa, None, None, accVa, None, timeElapsed1=timeSpentTraining, timeElapsed2=timeSpentTotal) # Update plot (val curve)

    # -----------------------------------------------------------------------------
    # Logging/plotting/saving
    if (step == 5) or (step == 10) or isLastStep:
        print(f"Step {step+1}/{config.nSteps}   Peak CUDA memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GB")
        torch.cuda.reset_peak_memory_stats()
    lossTrToPlot += lossTr; accTrToPlot += accTr # Accum over 'plotEvery' steps; much more stable than plotting the tr loss of every batch
    if (step == 5) or ((config.plotEvery > 0) and (step % config.plotEvery) == 0) or isLastStep:
        timePerStep = (timeSpentTraining - timeLast) / (step - stepLast) # Average per step since the last plotting
        lossTrToPlot /= (step - stepLast); accTrToPlot /= (step - stepLast) # Average over 'plotEvery' steps
        print(f"Step {step+1}/{config.nSteps}   lossTr {lossTrToPlot[0]:.4f}   accTr {100*accTrToPlot[0]:.1f}   {timeSpentTraining:.0f}s = {timePerStep:.2f}s/step")
        timeLast = timeSpentTraining; stepLast = step # Reset
        if config.plottingLevel >= 2: # Update plot (activation magnitudes)
            for k in range(1, 4):
                attrName = f"actMonitor{k}"
                magnitudes = [getattr(m, attrName).detach().cpu().item() for m in modelEval.modules() if hasattr(m, attrName)]
                if not magnitudes: continue
                if livePlots["act"] is None: # First iteration: create the plot
                    nMagnitudes = len(magnitudes)
                    livePlots["act"] = LivePlotAct("act-" + runName, "Magnitude of activations", 0, 1*config.afRange, nMagnitudes, 3, config.nSteps)
                livePlots["act"].plot(k, step+1, magnitudes) # Update the plot
                print(f"Step {step+1}/{config.nSteps}   Activation magnitudes ({k}): {" ".join(f"{x:4.2f}" for x in magnitudes)}")
        if (config.plottingLevel >= 1):
            livePlots["loss"].plot(step+1, lossTrToPlot, None, None, accTrToPlot, None, None, timeElapsed1=timeSpentTraining, timeElapsed2=timeSpentTotal) # Update plot (tr curve)
        lossTrToPlot.fill(0); accTrToPlot.fill(0) # Reset
        if (countParameterizedAfs(model) > 0) and (config.afLr != 0): # Not a fixed AF and not a loaded/frozen AF
            plotAfs(config, modelEval, xs, step+1) # Plot AF
        if (config.plottingLevel >= 1) and livePlots["loss"].shouldStopTraining(): # The loss/acc figure has just been closed by the user
            print(f"Stop requested")
            isLastStep = True # Will stop after saving checkpoint
        for tmp in livePlots.values(): tmp.drawFig() # Draw figures and make UI responsive
    elif ((step % 5) == 0): # Regularly but not too often (slowish)
        for tmp in livePlots.values(): tmp.flushFig() # Make UI responsive

    # Save AF
    if (isLastStep and (config.saveAfEvery >= 0)) or ((config.saveAfEvery > 0) and ((step % config.saveAfEvery) == 0) and (step > 1)):
        print(f"Saving AF:       {getFileNamePtAf(step+1)}", flush=True);  saveAfPt(config, modelEval,  getFileNamePtAf(step+1))
        print(f"Saving AF (CSV): {getFileNameCsvAf(step+1)}", flush=True); saveAfCsv(modelEval, getFileNameCsvAf(step+1))
    # Save model
    if (isLastStep and (config.saveModelEvery >= 0)) or ((config.saveModelEvery > 0) and ((step % config.saveModelEvery) == 0) and (step > 1)):
        print(f"Saving model: {getFileNamePt(step+1)}", flush=True)
        torch.save(dict(config=config, step=step, code=code, model=modelEval.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]), getFileNamePt(step+1))
    if isLastStep:
        break

# -----------------------------------------------------------------------------
# Save plots as image files
if (config.plottingLevel >= 1):
    fileNameFig = livePlots["loss"].saveFig(config.dirResults)
    if fileNameFig: print(f"Saving plot (loss/acc): {fileNameFig}", flush=True)
    fileNameFig = livePlots["af"].saveFig(config.dirResults)
    if fileNameFig: print(f"Saving plot (AF): {fileNameFig}", flush=True)
if (config.plottingLevel >= 2):
    fileNameFig = livePlots["act"].saveFig(config.dirResults)
    if fileNameFig: print(f"Saving plot (act): {fileNameFig}", flush=True)
