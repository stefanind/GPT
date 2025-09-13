import math

def cosine_warmup_lr(it, min_lr, max_lr, warmup_steps, max_steps):
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