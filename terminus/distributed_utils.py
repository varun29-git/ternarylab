"""Node-local data certification without eight redundant multi-TB scans."""
import os
import socket
from dataclasses import asdict
from pathlib import Path

import torch.distributed as dist

from .integrity import fingerprint, validate_manifest


def validate_node_manifest(manifest, root, config, *, development_mode=False):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return validate_manifest(manifest, root, config, development_mode=development_mode)
    world, rank = dist.get_world_size(), dist.get_rank()
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if local_world <= 0 or world % local_world:
        raise ValueError("node validation requires homogeneous contiguous torchrun ranks")
    identities = [None] * world
    path = Path(root).resolve()
    stat = path.stat()
    identity = (socket.gethostname(), str(path), stat.st_dev, stat.st_ino,
                fingerprint(asdict(manifest)), fingerprint(asdict(config)))
    dist.all_gather_object(identities, identity)
    for first in range(0, world, local_world):
        if len(set(identities[first:first + local_world])) != 1:
            raise ValueError("node ranks do not share the same host and data directory")
    own_group = None
    leader = rank // local_world * local_world
    for first in range(0, world, local_world):
        group = dist.new_group(list(range(first, first + local_world)))
        if first == leader:
            own_group = group
    message = [None]
    if rank == leader:
        try:
            message[0] = {"report": validate_manifest(manifest, root, config, development_mode=development_mode)}
        except Exception as error:
            message[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(message, src=leader, group=own_group)
    errors = [None] * world
    dist.all_gather_object(errors, message[0].get("error"))
    if any(errors):
        raise ValueError(f"node data certification failed: {next(error for error in errors if error)}")
    return message[0]["report"]
