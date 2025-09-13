"""
This is a direct rebuilding of the code from the video "Let's reproduce GPT-2" by Andrej Karpathy.
I added my own comments for educational purposes to help with my own understanding and to properly show the work I have put in.
The state of this code will diverge from Karpathy's as I continue to make my own additions. 
These additions will be documented here.

Additions:
    Dropout and biases
"""


import os
import time
import inspect
import tiktoken

from dataclasses import dataclass
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from hellaswag import iterate_examples, render_example

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # check if the number of embeddings is a multiple of the number of heads
        # ensures that the embeddings can be properly split to each attn head
        assert config.n_embd % config.n_head == 0

        # project to 3 for query, key, values
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # seems pointless but important to be used to mix info from concatenated heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # tag for scaling down the variance of the residual stream
        self.c_proj.SCALE_INIT = 1

        # added dropout for regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        # initialize number of heads and embeddings
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # -------- only needed if flash attention isn't used or is_causal=False
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        # get all dimensions
        B, T, C = x.size()

        # get query, key, values
        qkv = self.c_attn(x)

        # get three B, T, C, split along dim=2 (C)
        q, k, v = qkv.split(self.n_embd, dim=2) # split into size of self.n_embd along the 2nd dim

        # make the number of heads into a batch dimension like B
        # want pytorch to treat them as batches
        # all turn into -> (batch size, number of heads, sequence length, embd size for each head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # ------- manual attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v      # B, T, nh, hs
        
        # ------- flash attention (need CUDA)
        # is_causal=True so the mask isn't needed
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # concat all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 'contiguous' since transpose alone dn lay it into memory but view() needs it to be

        # mixes info from all head outputs and adds dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

# GPT-2 MLP but with added dropout regularization
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 4x projection, exactly as the original transformer
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # uses approximate GELU; not needed anymore but used for proper replication
        # GELU over RELU to remove dead neurons
        self.gelu    = nn.GELU(approximate='tanh')

        # back from the 4x projection down to size of n_embd
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # tag for scaling down the variance of the residual stream
        self.c_proj.SCALE_INIT = 1

        # regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # pass through the input x
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# GPT-2 transformer blocks
# different from the original transformer in GPT-2 because layernorm is added before attn as well
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)
    
    def forward(self, x):
        # residual stream + attn
        x = x + self.attn(self.ln_1(x))
        # residual stream + mlp
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int  = 1024  # max context/sequence length
    vocab_size: int  = 50257 # number of tokens: 256 bytes tokens, 1 EoT token, and 50,000 BPE merges
    n_layer: int     = 12    # number of layers
    n_head: int      = 12    # number of attn heads 
    n_embd: int      = 768   # embedding dimension
    dropout: float   = 0.2   # percentage of neurons dropped out
    bias: bool       = True  # add bias or not

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict = dict but with inherited nn.Module
        self.transformer = nn.ModuleDict(dict(

            # gives each token from the vocab an embedding of size n_embd
            wte  = nn.Embedding(config.vocab_size, config.n_embd), 
            # gives each position an embedding of size n_embd
            wpe  = nn.Embedding(config.block_size, config.n_embd),

            # ModuleList = list but with inherited nn.Module
            # creates n_layer amount of attn Blocks to split embeddings
            h    = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            # after the attn block mlp + residual stream, final layernorm
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # projects the embeddings up to the vocab size for classification
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT-2 does not use bias

        # tied weights:
        # to reduce weight number and improve consistency with the representation space
        self.transformer.wte.weight = self.lm_head.weight

        # initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # NOTE: Layernorm has proper default initialization w/ scale set to 1 and bias set to 0
        # so nothing extra is required here; we keep it defaulted
        # but for nn.Linear, linear weight initialization are set to a uniform distribution
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # so initializate weights according to GPT-2 -> normal dist w/ 0.02 std 

        # NOTE: 0.02 std is roughly around the Xavier initialization so no need for 1/sqrt(d_model)
        if isinstance(module, nn.Linear):
            std = 0.02 

            # need to control the activation growth of the residual stream
            # so we scale the weights down of the activations at the end of each block
            # hence, assign an attribute that acts as a tag for scaling
            # also, 2 * n_layer is done because each block adds 2 residual contributions (attn and mlp)
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # set bias to zero instead of the pytorch defaulted uniform dist
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

        # set std to 0.02 instead of 1 
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx, targets=None):

        # idx of shape (B, T) -> batch size by sequence length 
        B, T = idx.size()

        # ensure that the sequence length is not bigger than context 
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        # position of each token up to the sequence length, T
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # input the positions to get their embeddings
        pos_emb = self.transformer.wpe(pos)

        # input the tokens to get their embeddings
        tok_emb = self.transformer.wte(idx)

        # combine the token embeddings and the positional embeddings
        x = tok_emb + pos_emb

        # pass through the 12 blocks, each w/ their layernorms, attn, and mlp
        for block in self.transformer.h:
            x = block(x)

        # pass through the layernorm after the attn mlp's
        x = self.transformer.ln_f(x)

        # get the logits via linear layer, i.e., from embedding size transformed to vocab size
        logits = self.lm_head(x) # (B, T, 50257)

        loss = None
        # for training
        if targets is not None:
            # takes (B*T, V), the logits, and (B*T), the targets, as input to calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    # regularize weights
    # goal: pull down weights so the network finds a solution that doesn't involve large weights
    # improves generalization; prevents overfitting
    def configure_optimizers(self, weight_decay, learning_rate, device):

        # 'param_dict' returns a dict containing only those parameters that has a gradient
        # goal: want only the trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}               # gets an iterator of (name, param) from the model
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filters out 'frozen' params 

        # split the params into those that will be weight decayed and those that will not be
        # goal: want to weight decay matrices; don't want to weight decay biases or 1D tensors (doesn't make sense to decay single biases or the scale/shift in layernorm)
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]  # matrices containing params
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2] # 1D tensors: layernorm, biases

        # pass into AdamW to tune only the 'decay_params' with weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # sanity check
        # get the total number of params for each group
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:        
            # print them out to verify correct behavior
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # fused is a parameter option for AdamW that 'fuses' the kernels together for all parameters 
        # without it, there are a lot of individual kernels (e.g., mul, add, decay); this just combines them all into one
        # goal: gets rid of a lot of overhead by calling one kernel on all parameter operations
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        # betas and eps are from GPT 3 paper
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
    
    # use the pretrained GPT-2 for evaluation comparison
    # goal: want to compare my model against GPT-2
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



# ---------------------------------------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # convert to int32 because pytorch prefers it
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T

        # for when ddp is being used
        self.process_rank = process_rank
        self.num_processes = num_processes

        # TODO: between shard shuffling
        # goal: smooth out training loss curve 
        # the curve isn't smooth because each shard likely reflects data from specific areas of the internet
        # e.g., shard 1-5 might be Wikipedia, then 6-10 might be all of Arxiv, etc.
        # so shuffling between shards makes training loss smoother rather than jagged

        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root) # inputs all the files into a list
        shards = [s for s in shards if split in s] # selects only those shards that are assoc w/ the split (val or train)
        shards = sorted(shards) # sort the shards because listdir does not sort them
        shards = [os.path.join(data_root, s) for s in shards] # joins the data_root path with the filename
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B, T = self.B, self.T

        # extract tokens based on the current position
        # ensures that each batch isn't overlapping
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # +1 to get the target token

        # parse tokens into B, T
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))

        # increases position based on what data has already been used
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous() # up to -1 because when -1 there is no next logit to predict
    shift_tokens = (tokens[..., 1:]).contiguous()
    
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)    
    
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token


    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# ------------------------------------------------------------------------------------------------------------
# ddp: Distributed Data Parallel
# ddp requires the use of torchrun
# torchrun will run the script the amount of times equal to WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run or not

if ddp:
    # nccl is the backend for GPU to GPU communication in CUDA 
    init_process_group(backend='nccl')

    # each rank will run the script
    # need to ensure that each rank will run on different parts of the data
    ddp_rank       = int(os.environ['RANK'])       # the global rank
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # the GPU index on a specific node 
    ddp_world_size = int(os.environ['WORLD_SIZE']) # all processes (e.g., total GPU's found)
    device         = f'cuda:{ddp_local_rank}'      # bind this process to a specific GPU on the node
    torch.cuda.set_device(device)                  # set each local rank to their own device
    master_process = ddp_rank == 0                 # master process does extra by logging, checkpoints, etc.
else:
    # normal run without ddp
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# pytorch is strict about device_type vs device
device_type = "cuda" if device.startswith("cuda") else "cpu"

# seeeeeeeding
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # ~0.5M batch size in the number of tokens; used a "nice number" (2**19)
B = 16    # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure that the batch size is divisible by B * T * ddp_world_size"
# going to do many forward and backwards without updating
# goal: want to simulate/handle the GPT-3 Small batch size but with less compute, i.e., cannot do B=488 to get 0.5M batch size

# do forward and backward grad_accum_steps number of times
# e.g., 2**19 // (16 * 1024) = 32
# so do forward an backward 32 times before updating
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"the full batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

# set the data/val loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# using TF32 (10-bit mantissa instead of 23-bit) for speed, as long as I have some reasonable accuracy
# useful for training/inference
torch.set_float32_matmul_precision('high') 

# override vocab size to be a "nice number"
# "nice numbers" have more flops but are more efficient due to how GPU's are made
# kernel's will generally operate on 64 x 64 blocks 
# after computing all these, they will then perform on the odd numbers not within this multiple
# these kernels that do this extra part are inefficient
# so padding the input results in these optimized kernels working with all the numbers.
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

use_compile = False
if use_compile:
    model = torch.compile(model) 

if ddp:
    # DDP synchronizes and averages the gradients across all ranks
    model = DDP(model, device_ids=[ddp_local_rank])
# since using DDP makes it a new object, need .module to access nn.Module
raw_model = model.module if ddp else model


# GPT-3 Small hyperparameters
max_lr       = 6e-4           # peak lr
min_lr       = max_lr * 0.1   # final lr (10% of max)

# GPT-3 paper says 375M tokens are for warmup
# 375e6 / 2**19 = 715
warmup_steps = 715           

# 10**9 / 2**19 = 19073 ==> 10B tokens / 524288 batch size = 19073 steps
max_steps    = 19073          

def get_lr(it):
    # linear warmup start
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    # after max_steps, lr constant at min_lr
    if it > max_steps:
        return min_lr
    
    # between warmup and max, use cosine decay 
    # basically just a manual implementation of CosineAnnealingLR in pytorch
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0 
    return min_lr + coeff * (max_lr - min_lr)

# set the optimizer using the function w/ proper inputs
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)


# checkpoint and hellaswag eval log directory
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
# open the file to clear it out before appending to it later
# goal: start from scratch each time a training run is done
with open(log_file, "w") as f:
    pass

# main training loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # validation loop once in a while
    if step % 250 == 0 or last_step:
        # eval mode
        model.eval()
        # start the data at 0, but only for the 'val' set which has one shard
        val_loader.reset()
        # exactly the same as the training loop except no eval mode and no_grad
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        
        # write the validation loss to the log file and set up a model checkpoint
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt") # pad with zeros for consistent parsing
                checkpoint = {
                    'model': raw_model.state_dict(),   # model weights
                    'config': raw_model.config,        # the config settings
                    'step': step,                      # the current step
                    'val_loss': val_loss_accum.item()  # the val loss at this step
                }
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0

        # 
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # train mode
    model.train()

    # initialize/reset all gradients to zero
    optimizer.zero_grad()

    # accumulate loss for reporting
    loss_accum = 0.0

    # implements a way to do ~0.5M batch size in tokens to simulate what GPT-3 Small actually does
    # can't do it how they did it because of lack of GPUs
    for micro_step in range(grad_accum_steps):

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # don't want to synchronize after every loss.backward() during each micro step
        # so toggle 'require_backward_grad_sync' = True only when it is the last step
        # goal: only sync gradients at the end of the accum cycle, resulting in simulating a normal batch forward/backward
        # also, this is required during a forward pass
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        
        # ------ CHECK GPU TYPES BEFORE RUNNING!!! ------
        with torch.autocast(dtype=torch.bfloat16): # only allowed with Ampere and newer GPUs
            logits, loss = model(x, y)

        # scale each loss by the total steps to match a normal training loop
        # if no scaling, then the gradients are SUMing instead of MEANing
        # i.e., cross entropy uses 'reduction=mean' not 'reduction=sum'
        # so we need to emulate this behavior by adding each normalized gradient
        loss = loss / grad_accum_steps 

        # detach to add the float to loss_accum 
        loss_accum += loss.detach()

        # compute gradients across nodes
        loss.backward()

    # Uses pytorch distributed to all-reduce (average) the loss values across processes,
    # so every rank ends up with the same global mean loss
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)


    # ensures gradients squared and summed (global norm) do not go over 1.0
    # i.e., the magnitude of all the gradients do not go over 1.0
    # want gradients to not diverge too much if they become wild
    # sometimes a batch can end up with v high loss and this will end up w/ v high gradient
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # using the lr function, set the lr in each parameter group found in the optimizer
    lr = get_lr(step)
    # iterate over the groups and set the lr based on the conditions of the function
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # update weights
    optimizer.step()

    # ensure CPU and GPU are in sync
    # i.e., finish all GPU work before moving on
    torch.cuda.synchronize()

    # tracking metrics for printing while training
    t1 = time.time()
    dt = (t1 - t0)*1000
    # calculates number of tokens processed globally in one optimizer step
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec   = tokens_processed / dt

    # only print out and log the master process metrics for tracking
    # goal: don't want multiple printouts for each process
    if master_process:
        print(f"step {step:4d}, | loss: {loss_accum.item():.6f}, | lr: {lr:.4e} | norm: {norm:.4f} | dt {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# destroy the process group to avoid a printed error
# not necessarily required since we're using torchrun (destroys each process automatically)
if ddp:
    destroy_process_group()