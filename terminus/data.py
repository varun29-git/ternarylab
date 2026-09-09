from __future__ import annotations

import json
from functools import lru_cache
import math
import queue
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_TOTAL_TRAIN_TOKENS = 5_600_000_000
DEFAULT_COSMOPEDIA_STORIES_PREFIX_TOKENS = 0
DEFAULT_VAL_TOKENS = 50_000_000
DEFAULT_SEQ_LEN = 4096
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL_VOCAB_SIZE = 32_000


DTYPE_BY_NAME = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


@dataclass(frozen=True)
class TokenDataConfig:
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    total_train_tokens: int = DEFAULT_TOTAL_TRAIN_TOKENS
    cosmopedia_stories_prefix_tokens: int = DEFAULT_COSMOPEDIA_STORIES_PREFIX_TOKENS
    validation_tokens: int = DEFAULT_VAL_TOKENS
    token_dtype: str = "uint16"
    train_split: str = "train"
    val_split: str = "val"
    prefix_source: str = ""
    source_mix: Dict[str, float] = field(default_factory=dict)
    phase_source_mixes: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("seq_len", "batch_size", "total_train_tokens", "cosmopedia_stories_prefix_tokens", "validation_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_train_tokens <= 0:
            raise ValueError("total_train_tokens must be positive")
        if self.cosmopedia_stories_prefix_tokens < 0:
            raise ValueError("cosmopedia_stories_prefix_tokens must be non-negative")
        if self.cosmopedia_stories_prefix_tokens > self.total_train_tokens:
            raise ValueError("prefix token budget cannot exceed total train tokens")
        if self.validation_tokens < 0:
            raise ValueError("validation_tokens must be non-negative")
        if self.token_dtype not in DTYPE_BY_NAME:
            raise ValueError(f"unsupported token_dtype: {self.token_dtype}")
        if self.source_mix:
            if self.cosmopedia_stories_prefix_tokens:
                raise ValueError("source_mix cannot be combined with a prefix curriculum")
            if self.phase_source_mixes:
                raise ValueError("source_mix cannot be combined with phase_source_mixes")
            if any(float(weight) <= 0.0 for weight in self.source_mix.values()):
                raise ValueError("source_mix weights must be positive")
            if not math.isclose(sum(self.source_mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("source_mix weights must sum to exactly 1.0")
            if any(
                not math.isclose(float(weight) * 1000, round(float(weight) * 1000), abs_tol=1e-9)
                for weight in self.source_mix.values()
            ):
                raise ValueError("source_mix weights must be representable in 0.001 increments")
        if self.phase_source_mixes:
            if self.cosmopedia_stories_prefix_tokens:
                raise ValueError("phase_source_mixes cannot be combined with a prefix curriculum")
            phase_token_sum = 0
            for index, phase in enumerate(self.phase_source_mixes):
                if not isinstance(phase, dict):
                    raise ValueError("each phase_source_mixes entry must be a dict")
                tokens = phase.get("tokens", 0)
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                    raise ValueError(f"phase {index} tokens must be positive")
                if tokens < self.seq_len or (index + 1 < len(self.phase_source_mixes) and tokens % self.seq_len):
                    raise ValueError("each phase needs a full sequence; nonfinal phase boundaries must align to seq_len")
                source_mix = phase.get("source_mix", {})
                if not isinstance(source_mix, dict) or not source_mix:
                    raise ValueError(f"phase {index} source_mix must be a non-empty dict")
                if any(float(weight) <= 0.0 for weight in source_mix.values()):
                    raise ValueError(f"phase {index} source_mix weights must be positive")
                if not math.isclose(sum(float(weight) for weight in source_mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(f"phase {index} source_mix weights must sum to exactly 1.0")
                if any(
                    not math.isclose(float(weight) * 1000, round(float(weight) * 1000), abs_tol=1e-9)
                    for weight in source_mix.values()
                ):
                    raise ValueError(
                        f"phase {index} source_mix weights must be representable in 0.001 increments"
                    )
                phase_token_sum += tokens
            if phase_token_sum != self.total_train_tokens:
                raise ValueError(
                    "phase_source_mixes token budgets must sum exactly to total_train_tokens"
                )

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def total_steps(self) -> int:
        return math.ceil(self.total_examples / self.batch_size)

    @property
    def total_examples(self) -> int:
        return self.total_train_tokens // self.seq_len

    @property
    def usable_train_tokens(self) -> int:
        return self.total_examples * self.seq_len

    @property
    def prefix_steps(self) -> int:
        return math.ceil(self.cosmopedia_stories_prefix_tokens / self.tokens_per_step)

    @property
    def prefix_examples(self) -> int:
        return math.ceil(self.cosmopedia_stories_prefix_tokens / self.seq_len)

    @property
    def prefix_effective_tokens(self) -> int:
        return self.prefix_examples * self.seq_len


@dataclass(frozen=True)
class TokenShardSpec:
    path: str
    source: str
    split: str
    num_tokens: int
    dtype: str = "uint16"
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    source_split: Optional[str] = None
    sha256: Optional[str] = None
    language: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.num_tokens, bool) or not isinstance(self.num_tokens, int) or self.num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer")
        if self.dtype not in DTYPE_BY_NAME:
            raise ValueError(f"unsupported shard dtype: {self.dtype}")
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256.lower())
        ):
            raise ValueError("shard sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class TokenManifest:
    config: TokenDataConfig
    shards: List[TokenShardSpec]
    tokenizer: Dict[str, object] = field(default_factory=dict)
    provenance: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TokenManifest":
        payload = json.loads(Path(path).read_text())
        if set(payload) - {"config", "shards", "tokenizer", "provenance"}:
            raise ValueError("unknown manifest fields")
        config = TokenDataConfig(**payload["config"])
        shards = [TokenShardSpec(**item) for item in payload["shards"]]
        return cls(config=config, shards=shards, tokenizer=payload.get("tokenizer", {}), provenance=payload.get("provenance", {}))

    def save(self, path: str | Path) -> None:
        payload = {
            "config": asdict(self.config),
            "shards": [asdict(shard) for shard in self.shards],
            "tokenizer": self.tokenizer,
            "provenance": self.provenance,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    def shards_for(self, *, split: str, source: Optional[str] = None) -> List[TokenShardSpec]:
        selected = [shard for shard in self.shards if shard.split == split]
        if source is not None:
            selected = [shard for shard in selected if shard.source == source]
        return selected


class TokenShard:
    def __init__(self, spec: TokenShardSpec, root: str | Path = ".") -> None:
        self.spec = spec
        self.path = Path(root) / spec.path
        self.tokens = self._open_tokens(self.path, spec)
        if len(self.tokens) != spec.num_tokens:
            raise ValueError(
                f"{self.path} has {len(self.tokens):,} tokens, manifest says {spec.num_tokens:,}"
            )

    @staticmethod
    def _open_tokens(path: Path, spec: TokenShardSpec) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            if arr.dtype != np.dtype(DTYPE_BY_NAME[spec.dtype]):
                raise ValueError(f"{path} dtype {arr.dtype} does not match manifest {spec.dtype}")
            if arr.ndim != 1:
                raise ValueError(f"{path} must contain a 1D token array")
            return arr

        dtype = DTYPE_BY_NAME[spec.dtype]
        byte_count = path.stat().st_size
        item_size = np.dtype(dtype).itemsize
        if byte_count % item_size != 0:
            raise ValueError(f"{path} size is not divisible by dtype size {item_size}")
        return np.memmap(path, dtype=dtype, mode="r")

    @property
    def num_tokens(self) -> int:
        return int(len(self.tokens))

    def read(self, start: int, length: int) -> np.ndarray:
        if start < 0:
            raise ValueError("start must be non-negative")
        end = start + length
        if end > self.num_tokens:
            raise IndexError(
                f"read [{start:,}, {end:,}) exceeds shard length {self.num_tokens:,}: {self.path}"
            )
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class TokenSpan:
    def __init__(self, specs: Sequence[TokenShardSpec], *, root: str | Path = ".") -> None:
        if not specs:
            raise ValueError("TokenSpan requires at least one shard")
        self.shards = [TokenShard(spec, root=root) for spec in specs]
        self.ends: List[int] = []
        running = 0
        for shard in self.shards:
            running += shard.num_tokens
            self.ends.append(running)
        self.num_tokens = running

    def assert_budget(self, required_tokens: int, *, name: str) -> None:
        needed = required_tokens + 1
        if self.num_tokens < needed:
            raise ValueError(
                f"{name} has {self.num_tokens:,} tokens, needs at least {needed:,}"
            )

    def read(self, start: int, length: int) -> np.ndarray:
        if start < 0:
            raise ValueError("start must be non-negative")
        if start + length > self.num_tokens:
            raise IndexError(
                f"read [{start:,}, {start + length:,}) exceeds span length {self.num_tokens:,}"
            )

        out = np.empty(length, dtype=np.int64)
        remaining = length
        out_pos = 0
        cursor = start
        shard_index = int(np.searchsorted(self.ends, cursor, side="right"))

        while remaining:
            shard_start = 0 if shard_index == 0 else self.ends[shard_index - 1]
            shard = self.shards[shard_index]
            offset = cursor - shard_start
            take = min(remaining, shard.num_tokens - offset)
            out[out_pos : out_pos + take] = shard.read(offset, take)
            out_pos += take
            cursor += take
            remaining -= take
            shard_index += 1

        return out

    def block(self, block_index: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = block_index * seq_len
        tokens = self.read(start, seq_len + 1)
        inputs = torch.from_numpy(tokens[:-1].copy()).long()
        targets = torch.from_numpy(tokens[1:].copy()).long()
        return inputs, targets


@lru_cache(maxsize=256)
def schedule_prefix_counts(source: str, schedule: Tuple[str, ...]) -> Tuple[int, ...]:
    counts = [0]
    for scheduled_source in schedule:
        counts.append(counts[-1] + int(scheduled_source == source))
    return tuple(counts)


class TrainingTokenRouter:
    """Maps global training-token offsets to the correct curriculum span.

    If a prefix is configured, the first prefix span comes from
    `config.prefix_source`. Otherwise all train shards are treated as the main
    stream in manifest order.
    With ``source_mix``, examples are deterministically interleaved by a fixed
    weighted-fair schedule while remaining sequential within each source.
    """

    def __init__(self, manifest: TokenManifest, *, root: str | Path = ".") -> None:
        self.config = manifest.config
        train_specs = manifest.shards_for(split=self.config.train_split)
        self.prefix: Optional[TokenSpan] = None
        self.mixed_spans: Dict[str, TokenSpan] = {}
        self.mix_schedule: Tuple[str, ...] = ()
        self.phase_specs: List[Dict[str, object]] = []

        if self.config.phase_source_mixes:
            available_sources = {spec.source for spec in train_specs}
            requested_sources = {
                str(source)
                for phase in self.config.phase_source_mixes
                for source in dict(phase["source_mix"]).keys()
            }
            missing_sources = sorted(requested_sources - available_sources)
            if missing_sources:
                raise ValueError(
                    f"phase_source_mixes refer to unknown train sources: {missing_sources}"
                )
            phase_source_examples_needed: Dict[str, int] = {source: 0 for source in requested_sources}
            phase_specs: List[Dict[str, object]] = []
            phase_start_example = 0
            remaining_examples = self.config.total_examples
            for index, phase in enumerate(self.config.phase_source_mixes):
                source_mix = {
                    str(source): float(weight)
                    for source, weight in dict(phase["source_mix"]).items()
                }
                requested_phase_sources = set(source_mix)
                schedule = self._weighted_schedule(source_mix)
                phase_name = str(phase.get("name", f"phase_{index + 1}"))
                phase_examples = int(phase["tokens"]) // self.config.seq_len
                if index == len(self.config.phase_source_mixes) - 1:
                    phase_examples = remaining_examples
                remaining_examples -= phase_examples
                source_prefix_examples = dict(phase_source_examples_needed)
                for source in requested_phase_sources:
                    phase_source_examples_needed[source] += self._source_count_before(
                        source, phase_examples, schedule
                    )
                phase_specs.append(
                    {
                        "name": phase_name,
                        "schedule": schedule,
                        "start_example": phase_start_example,
                        "end_example": phase_start_example + phase_examples,
                        "source_prefix_examples": source_prefix_examples,
                    }
                )
                phase_start_example += phase_examples
            if phase_start_example != self.config.total_examples:
                raise ValueError(
                    f"phase_source_mixes cover {phase_start_example} examples, expected {self.config.total_examples}"
                )
            for source in sorted(requested_sources):
                span = TokenSpan(
                    [spec for spec in train_specs if spec.source == source],
                    root=root,
                )
                required_examples = int(phase_source_examples_needed[source])
                span.assert_budget(required_examples * self.config.seq_len, name=source)
                self.mixed_spans[source] = span
            self.phase_specs = phase_specs
            self.main = None
            return

        if self.config.source_mix:
            available_sources = {spec.source for spec in train_specs}
            requested_sources = set(self.config.source_mix)
            if available_sources != requested_sources:
                raise ValueError(
                    f"source_mix keys {sorted(requested_sources)} do not match train sources "
                    f"{sorted(available_sources)}"
                )
            self.mix_schedule = self._weighted_schedule(self.config.source_mix)
            total_examples = self.config.total_examples
            for source in sorted(requested_sources):
                span = TokenSpan(
                    [spec for spec in train_specs if spec.source == source],
                    root=root,
                )
                required_examples = self._source_count_before(source, total_examples, self.mix_schedule)
                span.assert_budget(required_examples * self.config.seq_len, name=source)
                self.mixed_spans[source] = span
            self.main = None
            return

        if self.config.prefix_effective_tokens > 0:
            prefix_specs = manifest.shards_for(
                split=self.config.train_split,
                source=self.config.prefix_source,
            )
            main_specs = [
                shard for shard in train_specs if shard.source != self.config.prefix_source
            ]
            self.prefix = TokenSpan(prefix_specs, root=root)
            self.prefix.assert_budget(
                self.config.prefix_effective_tokens,
                name=self.config.prefix_source,
            )
        else:
            main_specs = train_specs

        self.main: Optional[TokenSpan] = TokenSpan(main_specs, root=root)
        self.main.assert_budget(
            max(0, self.config.usable_train_tokens - self.config.prefix_effective_tokens),
            name="train",
        )

    @staticmethod
    def _weighted_schedule(source_mix: Dict[str, float], cycle_size: int = 1000) -> Tuple[str, ...]:
        sources = sorted(source_mix)
        counts = {source: int(round(source_mix[source] * cycle_size)) for source in sources}
        difference = cycle_size - sum(counts.values())
        counts[max(sources, key=lambda item: source_mix[item])] += difference
        if any(count <= 0 for count in counts.values()):
            raise ValueError(
                f"source_mix weights are too small for the {cycle_size}-example schedule: {counts}"
            )
        remaining = dict(counts)
        accumulator = {source: 0 for source in sources}
        schedule: List[str] = []
        for _ in range(cycle_size):
            for source in sources:
                accumulator[source] += counts[source]
            candidates = [source for source in sources if remaining[source] > 0]
            selected = max(candidates, key=lambda item: (accumulator[item], item))
            schedule.append(selected)
            accumulator[selected] -= cycle_size
            remaining[selected] -= 1
        return tuple(schedule)

    @staticmethod
    def _source_count_before(
        source: str,
        example_index: int,
        schedule: Sequence[str],
    ) -> int:
        prefix = schedule_prefix_counts(source, tuple(schedule))
        full_cycles, offset = divmod(example_index, len(schedule))
        return full_cycles * prefix[-1] + prefix[offset]

    def _span_and_offset(self, token_offset: int) -> Tuple[TokenSpan, int]:
        if self.phase_specs:
            if token_offset % self.config.seq_len:
                raise ValueError("phase-mixed token offsets must align to seq_len")
            example_index = token_offset // self.config.seq_len
            for phase in self.phase_specs:
                start_example = int(phase["start_example"])
                end_example = int(phase["end_example"])
                if start_example <= example_index < end_example:
                    phase_local_example = example_index - start_example
                    schedule = tuple(phase["schedule"])
                    source = schedule[phase_local_example % len(schedule)]
                    source_prefix_examples = dict(phase["source_prefix_examples"])
                    source_example_index = int(source_prefix_examples.get(source, 0)) + self._source_count_before(
                        source,
                        phase_local_example,
                        schedule,
                    )
                    return self.mixed_spans[source], source_example_index * self.config.seq_len
            raise IndexError(f"phase-mixed example index {example_index} is out of range")
        if self.mix_schedule:
            if token_offset % self.config.seq_len:
                raise ValueError("mixed-source token offsets must align to seq_len")
            example_index = token_offset // self.config.seq_len
            source = self.mix_schedule[example_index % len(self.mix_schedule)]
            source_example_index = self._source_count_before(source, example_index, self.mix_schedule)
            return self.mixed_spans[source], source_example_index * self.config.seq_len
        if self.prefix is not None and token_offset < self.config.prefix_effective_tokens:
            return self.prefix, token_offset
        assert self.main is not None
        return self.main, token_offset - self.config.prefix_effective_tokens

    def block_at_token_offset(self, token_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokens_at_token_offset(token_offset)
        inputs = torch.from_numpy(tokens[:-1].copy()).long()
        targets = torch.from_numpy(tokens[1:].copy()).long()
        return inputs, targets

    def tokens_at_token_offset(self, token_offset: int) -> np.ndarray:
        span, local_offset = self._span_and_offset(token_offset)
        return span.read(local_offset, self.config.seq_len + 1)


class TokenBatcher:
    def __init__(
        self,
        manifest: TokenManifest,
        *,
        root: str | Path = ".",
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device | str] = None,
        validation_only: bool = False,
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank >= world_size:
            raise ValueError("rank must be smaller than world_size")

        self.manifest = manifest
        self.config = manifest.config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device) if device is not None else None
        self.router = None if validation_only else TrainingTokenRouter(manifest, root=root)

        val_specs = manifest.shards_for(split=self.config.val_split)
        self.val = TokenSpan(val_specs, root=root)
        self.val.assert_budget(self.config.validation_tokens, name="validation")

    def train_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.router is None:
            raise RuntimeError("validation-only TokenBatcher cannot produce training batches")
        if step < 0:
            raise ValueError("step must be non-negative")

        global_examples_before_step = step * self.config.batch_size * self.world_size
        remaining = self.config.total_examples - global_examples_before_step - self.rank
        rows = min(self.config.batch_size, max(0, math.ceil(remaining / self.world_size)))
        if not rows:
            if global_examples_before_step < self.config.total_examples:
                shape = (1, self.config.seq_len)
                return torch.zeros(shape, dtype=torch.long, device=self.device), torch.full(shape, -100, dtype=torch.long, device=self.device)
            raise StopIteration("training token budget exhausted")
        # Allocate a complete batch once, avoiding per-sequence tensor copies
        # followed by two stack allocations. X/Y share the shifted CPU window.
        windows = np.empty((rows, self.config.seq_len + 1), dtype=np.int64)
        for local_index in range(rows):
            global_example_index = (
                global_examples_before_step
                + local_index * self.world_size
                + self.rank
            )
            token_offset = global_example_index * self.config.seq_len



            windows[local_index] = self.router.tokens_at_token_offset(token_offset)
        window_tensor = torch.from_numpy(windows)
        x_batch, y_batch = window_tensor[:, :-1], window_tensor[:, 1:]
        if self.device is not None:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
        return x_batch, y_batch

    def val_batches(self, max_batches: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        usable_blocks = max(0, min(self.val.num_tokens - 1, self.config.validation_tokens) // self.config.seq_len)
        total_batches = math.ceil(usable_blocks / (self.config.batch_size * self.world_size))
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        for batch_index in range(total_batches):
            xs: List[torch.Tensor] = []
            ys: List[torch.Tensor] = []
            for local_index in range(self.config.batch_size):
                block_index = (batch_index * self.config.batch_size + local_index) * self.world_size + self.rank
                if block_index >= usable_blocks:
                    break
                inputs, targets = self.val.block(block_index, self.config.seq_len)
                xs.append(inputs)
                ys.append(targets)
            if not xs:
                continue
            x_batch = torch.stack(xs)
            y_batch = torch.stack(ys)
            if self.device is not None:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
            yield x_batch, y_batch

    def curriculum_status(self, step: int) -> Dict[str, int | str]:
        consumed = min(
            step * self.config.tokens_per_step * self.world_size,
            self.config.usable_train_tokens,
        )
        if self.config.phase_source_mixes:
            example_index = min(self.config.total_examples, consumed // self.config.seq_len)
            stage = str(self.config.phase_source_mixes[-1].get("name", f"phase_{len(self.config.phase_source_mixes)}"))
            running_examples = 0
            for index, phase in enumerate(self.config.phase_source_mixes):
                phase_examples = int(phase["tokens"]) // self.config.seq_len
                if index == len(self.config.phase_source_mixes) - 1:
                    phase_examples = self.config.total_examples - running_examples
                running_examples += phase_examples
                if example_index < running_examples:
                    stage = str(phase.get("name", f"phase_{index + 1}"))
                    break
        elif self.config.prefix_effective_tokens <= 0:
            stage = "train"
        elif consumed >= self.config.prefix_effective_tokens:
            stage = "main_train"
        else:
            stage = f"{self.config.prefix_source}_prefix"
        return {
            "step": step,
            "tokens_consumed": consumed,
            "stage": stage,
            "prefix_tokens": self.config.cosmopedia_stories_prefix_tokens,
            "prefix_effective_tokens": self.config.prefix_effective_tokens,
            "total_train_tokens": self.config.total_train_tokens,
            "usable_train_tokens": self.config.usable_train_tokens,
        }


_END = object()


def _maybe_pin_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    *,
    pin_memory: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not pin_memory or not torch.cuda.is_available():
        return batch
    inputs, targets = batch
    return inputs.pin_memory(), targets.pin_memory()


class PrefetchingTokenLoader:
    """Background-prefetched training batches.

    A worker thread keeps a bounded CPU queue full of ready batches. On CUDA,
    batches are pinned in the worker and copied to the target device on a
    dedicated CUDA stream before they are yielded, so host reads and H2D copies
    can overlap the model's forward/backward work.
    """

    def __init__(
        self,
        manifest: TokenManifest,
        *,
        root: str | Path = ".",
        rank: int = 0,
        world_size: int = 1,
        start_step: int = 0,
        max_steps: Optional[int] = None,
        prefetch_batches: int = 8,
        pin_memory: bool = True,
        device: Optional[torch.device | str] = None,
        return_step: bool = False,
    ) -> None:
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")

        self.manifest = manifest
        self.root = Path(root)
        self.rank = rank
        self.world_size = world_size
        self.start_step = start_step
        self.max_steps = max_steps
        self.prefetch_batches = prefetch_batches
        self.pin_memory = pin_memory
        self.device = torch.device(device) if device is not None else None
        self.return_step = return_step

    def __iter__(self) -> "_PrefetchingTokenIterator":
        return _PrefetchingTokenIterator(self)


class _PrefetchingTokenIterator:
    def __init__(self, loader: PrefetchingTokenLoader) -> None:
        self.loader = loader
        self.queue: queue.Queue[object] = queue.Queue(maxsize=loader.prefetch_batches)
        self.closed = threading.Event()
        self.exhausted = False
        self.use_cuda = (
            loader.device is not None
            and loader.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.scheduled: Optional[Tuple[int, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        if self.use_cuda:
            self.copy_stream = torch.cuda.Stream(device=loader.device)
            self._schedule_next_cuda_batch()

    def _worker(self) -> None:
        try:
            if self.use_cuda:
                torch.cuda.set_device(self.loader.device)
            batcher = TokenBatcher(
                self.loader.manifest,
                root=self.loader.root,
                rank=self.loader.rank,
                world_size=self.loader.world_size,
                device=None,
            )
            step = self.loader.start_step
            produced = 0
            while not self.closed.is_set():
                if self.loader.max_steps is not None and produced >= self.loader.max_steps:
                    break
                try:
                    batch = batcher.train_batch(step)
                except StopIteration:
                    break
                batch = _maybe_pin_batch(batch, pin_memory=self.loader.pin_memory and self.use_cuda)
                self._put((step, batch))
                step += 1
                produced += 1
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(_END)

    def _put(self, item: object) -> None:
        while not self.closed.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _get(self, *, block=True) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        if self.exhausted:
            raise StopIteration
        item = self.queue.get(block=block)
        if item is _END:
            self.exhausted = True
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.exhausted = True
            self.close()
            raise item
        step, batch = item
        return step, batch

    def _schedule_next_cuda_batch(self, *, block=True) -> None:
        assert self.copy_stream is not None
        try:
            step, (x_cpu, y_cpu) = self._get(block=block)
        except (StopIteration, queue.Empty):
            self.scheduled = None
            return

        with torch.cuda.stream(self.copy_stream):
            x_gpu = x_cpu.to(self.loader.device, non_blocking=True)
            y_gpu = y_cpu.to(self.loader.device, non_blocking=True)
        self.scheduled = (step, (x_gpu, y_gpu))

    def __iter__(self) -> "_PrefetchingTokenIterator":
        return self

    def __next__(self):
        if not self.use_cuda:
            step, batch = self._get()
            return (step, *batch) if self.loader.return_step else batch

        if self.scheduled is None:
            self._schedule_next_cuda_batch()
        if self.scheduled is None:
            self.close()
            raise StopIteration

        assert self.copy_stream is not None
        current_stream = torch.cuda.current_stream(device=self.loader.device)
        current_stream.wait_stream(self.copy_stream)
        step, batch = self.scheduled
        for tensor in batch:
            tensor.record_stream(current_stream)

        self._schedule_next_cuda_batch(block=False)
        return (step, *batch) if self.loader.return_step else batch

    def close(self) -> None:
        self.exhausted = True
        self.closed.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)

    def __enter__(self) -> "_PrefetchingTokenIterator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
