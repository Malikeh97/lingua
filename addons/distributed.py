"""
Distributed primitives for sequence parallelism.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from lingua.distributed import DistributedArgs as LinguaDistributedArgs


@dataclass
class DistributedArgs(LinguaDistributedArgs):
    """Extended distributed args with sequence parallelism support."""
    sp_size: int = 1


# Keep for backwards compatibility
SequenceParallelArgs = DistributedArgs


def get_sp_group(sp_size: int) -> Optional[dist.ProcessGroup]:
    """
    Create process group for sequence parallelism.

    Groups consecutive ranks:
    - sp_size=2, world=8: [(0,1), (2,3), (4,5), (6,7)]
    """
    if sp_size == 1:
        return None

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % sp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sp_size})")

    num_groups = world_size // sp_size
    groups = []
    for i in range(num_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        groups.append(dist.new_group(ranks))

    return groups[rank // sp_size]


def get_dp_group(sp_size: int) -> Optional[dist.ProcessGroup]:
    """
    Create process group for data parallelism (complementary to sp).

    Ranks with the same sp_rank form a dp group:
    - sp_size=2, world=8:
      dp_group_0 = [0, 2, 4, 6]  (all sp_rank=0)
      dp_group_1 = [1, 3, 5, 7]  (all sp_rank=1)
    """
    if sp_size == 1:
        return None  # whole world is dp group (default)

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % sp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sp_size})")

    # sp_rank = rank % sp_size, each dp_group has ranks with same sp_rank
    groups = []
    for sp_rank in range(sp_size):
        ranks = list(range(sp_rank, world_size, sp_size))
        groups.append(dist.new_group(ranks))

    return groups[rank % sp_size]


def ring_send_recv(tensor: torch.Tensor, sp_group: dist.ProcessGroup) -> torch.Tensor:
    """Send tensor to next rank in ring, receive from previous."""
    sp_rank = dist.get_rank(sp_group)
    sp_size = dist.get_world_size(sp_group)

    recv_tensor = torch.empty_like(tensor)
    send_op = dist.P2POp(dist.isend, tensor, (sp_rank + 1) % sp_size, group=sp_group)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, (sp_rank - 1) % sp_size, group=sp_group)

    for req in dist.batch_isend_irecv([send_op, recv_op]):
        req.wait()

    return recv_tensor


def ring_send(tensor: torch.Tensor, sp_group: dist.ProcessGroup) -> None:
    """Send tensor to next rank."""
    sp_rank = dist.get_rank(sp_group)
    sp_size = dist.get_world_size(sp_group)
    if sp_rank < sp_size - 1:
        dist.send(tensor, dst=sp_rank + 1, group=sp_group)


def ring_recv(tensor: torch.Tensor, sp_group: dist.ProcessGroup) -> torch.Tensor:
    """Receive tensor from previous rank."""
    sp_rank = dist.get_rank(sp_group)
    if sp_rank == 0:
        return tensor
    recv = torch.empty_like(tensor)
    dist.recv(recv, src=sp_rank - 1, group=sp_group)
    return recv
