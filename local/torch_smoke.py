import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all-reduce mismatch: {value.item()} != {expected}")
    print(f"rank={dist.get_rank()} gpu={torch.cuda.get_device_name()} sum={value.item():.0f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
