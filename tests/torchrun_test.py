# test.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    model = DummyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    # Dummy input/output
    x = torch.randn(2, 10).to(device)
    y = model(x)

    print(f"[Rank {rank}] Successfully ran forward pass on device {device}")
    dist.barrier()

    # All-reduce test
    tensor = torch.tensor([rank], dtype=torch.float32).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] Sum of ranks across all processes: {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
