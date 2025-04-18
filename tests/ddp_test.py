# ddp_test.py
import os
import socket
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    hostname = socket.gethostname()
    
    torch.cuda.set_device(local_rank)
    
    print(f"Hello from rank {rank}/{world_size} on {hostname}, using GPU {local_rank}")
    
    # Dummy computation
    tensor = torch.tensor([rank], device=local_rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] Tensor after all_reduce: {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

