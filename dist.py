import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group


def setup_ddp():
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

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process