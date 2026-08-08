from __future__ import annotations

import json
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
DEFAULT_BATCH_SIZE = 14
DEFAULT_MODEL_VOCAB_SIZE = 32_000


DTYPE_BY_NAME = {
    "uint16": np.uint16,
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
                tokens = int(phase.get("tokens", 0))
                if tokens <= 0:
                    raise ValueError(f"phase {index} tokens must be positive")
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

    def __post_init__(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
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

    @classmethod
    def load(cls, path: str | Path) -> "TokenManifest":
        payload = json.loads(Path(path).read_text())
        config = TokenDataConfig(**payload["config"])
        shards = [TokenShardSpec(**item) for item in payload["shards"]]
        return cls(config=config, shards=shards)

    def save(self, path: str | Path) -> None:
        payload = {
            "config": asdict(self.config),
            "shards": [asdict(shard) for shard in self.shards],
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
        x = torch.from_numpy(tokens[:-1].copy()).long()
        y = torch.from_numpy(tokens[1:].copy()).long()
        return x, y


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
                required_examples = self._source_count_before(source, total_examples)
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
        cycle_size = len(schedule)
        full_cycles, offset = divmod(example_index, cycle_size)
        per_cycle = schedule.count(source)
        return full_cycles * per_cycle + tuple(schedule[:offset]).count(source)

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
        span, local_offset = self._span_and_offset(token_offset)
        tokens = span.read(local_offset, self.config.seq_len + 1)
        x = torch.from_numpy(tokens[:-1].copy()).long()
        y = torch.from_numpy(tokens[1:].copy()).long()
        return x, y


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

        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        global_examples_before_step = step * self.config.batch_size * self.world_size

        for local_index in range(self.config.batch_size):
            global_example_index = (
                global_examples_before_step
                + local_index * self.world_size
                + self.rank
            )
            token_offset = global_example_index * self.config.seq_len
            # Never construct a nominally full sequence that crosses the exact
            # token budget. Test 16's final partial step could request such a
            # sequence and then fail in the shard router.
            if token_offset + self.config.seq_len > self.config.usable_train_tokens:
                break
            x, y = self.router.block_at_token_offset(token_offset)
            xs.append(x)
            ys.append(y)

        if not xs:
            raise StopIteration("training token budget exhausted")

        x_batch = torch.stack(xs)
        y_batch = torch.stack(ys)
        if self.device is not None:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
        return x_batch, y_batch

    def val_batches(self, max_batches: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        usable_blocks = max(0, (min(self.val.num_tokens, self.config.validation_tokens) - 1) // self.config.seq_len)
        total_batches = usable_blocks // self.config.batch_size
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        for batch_index in range(total_batches):
            xs: List[torch.Tensor] = []
            ys: List[torch.Tensor] = []
            for local_index in range(self.config.batch_size):
                block_index = batch_index * self.config.batch_size + local_index
                x, y = self.val.block(block_index, self.config.seq_len)
                xs.append(x)
                ys.append(y)
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
    x, y = batch
    return x.pin_memory(), y.pin_memory()


def _row_text(row: Dict[str, object]) -> str:
    for key in ("text", "prompt", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return " ".join(value for value in row.values() if isinstance(value, str))


def _bounded_token_ids(
    ids: Sequence[int],
    *,
    eos_id: int,
    vocab_size: int = DEFAULT_MODEL_VOCAB_SIZE,
) -> List[int]:
    """Keep streaming tokenization on the same 32k-vocab contract as Test 1.

    The prepared Test 1 memmaps were already bounded before training. The
    streaming path tokenizes raw Hugging Face rows live, and some
    tokenizers can expose added/special IDs outside the model embedding table.
    Replace those rare IDs with EOS on CPU so CUDA never sees an illegal index.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    fallback = eos_id if 0 <= int(eos_id) < vocab_size else 0
    bounded: List[int] = []
    for token_id in ids:
        token = int(token_id)
        bounded.append(token if 0 <= token < vocab_size else fallback)
    return bounded


class StreamingTextTokenLoader:
    """Hugging Face streaming text loader with online tokenization.

    This is the convenience path for rented GPU notebooks where local token
    shards are not prepared. It keeps a background queue full of already
    tokenized CPU batches, then optionally stages them to CUDA with the same
    copy-stream pattern as the memmap loader.
    """

    def __init__(
        self,
        manifest: TokenManifest,
        *,
        tokenizer_name: str,
        start_step: int = 0,
        max_steps: Optional[int] = None,
        prefetch_batches: int = 8,
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
        self.config = manifest.config
        self.tokenizer_name = tokenizer_name
        self.start_step = start_step
        self.max_steps = max_steps
        self.prefetch_batches = prefetch_batches
        self.device = torch.device(device) if device is not None else None
        self.return_step = return_step

    def __iter__(self) -> "_StreamingTextTokenIterator":
        return _StreamingTextTokenIterator(self)


class _StreamingTextTokenIterator:
    def __init__(self, loader: StreamingTextTokenLoader) -> None:
        self.loader = loader
        self.queue: queue.Queue[object] = queue.Queue(maxsize=loader.prefetch_batches)
        self.closed = threading.Event()
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
            from datasets import load_dataset
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.loader.tokenizer_name, use_fast=True)
            tokenizer.model_max_length = int(1e30)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("streaming tokenizer must define eos_token_id")

            streams = self._open_streams(load_dataset)
            token_buffers: Dict[str, List[int]] = {source: [] for source in streams}
            skip_tokens_by_source = self._skip_tokens_by_source()
            produced = 0
            step = self.loader.start_step

            while not self.closed.is_set():
                if self.loader.max_steps is not None and produced >= self.loader.max_steps:
                    break
                examples = []
                targets = []
                while len(examples) < self.loader.config.batch_size:
                    chunk_index = step * self.loader.config.batch_size + len(examples)
                    source = self._source_for_chunk_index(chunk_index)
                    token_buffer = token_buffers[source]
                    while len(token_buffer) < self.loader.config.seq_len + 1:
                        try:
                            row = next(streams[source])
                        except StopIteration:
                            streams[source] = self._open_stream(load_dataset, source)
                            row = next(streams[source])
                        ids = tokenizer.encode(_row_text(row), add_special_tokens=False)
                        if not ids:
                            continue
                        ids.append(eos_id)
                        ids = _bounded_token_ids(ids, eos_id=eos_id)
                        skip_tokens = skip_tokens_by_source.get(source, 0)
                        if skip_tokens > 0:
                            take = min(skip_tokens, len(ids))
                            ids = ids[take:]
                            skip_tokens_by_source[source] = skip_tokens - take
                            if not ids:
                                continue
                        token_buffer.extend(ids)
                    chunk = token_buffer[: self.loader.config.seq_len + 1]
                    # Advance by seq_len, retaining the final target token as the
                    # next chunk's first input. This matches memmap LM packing and
                    # avoids dropping one real token at every sequence boundary.
                    token_buffers[source] = token_buffer[self.loader.config.seq_len :]
                    tensor = torch.tensor(chunk, dtype=torch.long)
                    examples.append(tensor[:-1])
                    targets.append(tensor[1:])

                batch = (torch.stack(examples), torch.stack(targets))
                batch = _maybe_pin_batch(batch, pin_memory=self.use_cuda)
                self._put((step, batch))
                step += 1
                produced += 1
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(_END)

    def _stage_for_token_offset(self, token_offset: int) -> str:
        return "prefix" if token_offset < self.loader.config.prefix_effective_tokens else "main"

    def _local_skip_for_token_offset(self, token_offset: int) -> int:
        if token_offset < self.loader.config.prefix_effective_tokens:
            return token_offset
        return token_offset - self.loader.config.prefix_effective_tokens

    def _source_for_chunk_index(self, chunk_index: int) -> str:
        if self.loader.config.prefix_effective_tokens > 0:
            prefix_chunks = self.loader.config.prefix_effective_tokens // self.loader.config.seq_len
            if chunk_index < prefix_chunks:
                return "tinystories"
            chunk_index -= prefix_chunks
        return "cosmopedia"

    def _skip_tokens_by_source(self) -> Dict[str, int]:
        chunks_before_start = self.loader.start_step * self.loader.config.batch_size
        skip_tokens = {"tinystories": 0, "cosmopedia": 0}
        tokens_per_chunk = self.loader.config.seq_len
        for chunk_index in range(chunks_before_start):
            skip_tokens[self._source_for_chunk_index(chunk_index)] += tokens_per_chunk
        return skip_tokens

    def _open_streams(self, load_dataset) -> Dict[str, Iterator[Dict[str, object]]]:
        sources = {"cosmopedia"}
        if self.loader.config.prefix_effective_tokens > 0:
            sources.add("tinystories")
        return {source: self._open_stream(load_dataset, source) for source in sources}

    def _open_stream(self, load_dataset, stage: str) -> Iterator[Dict[str, object]]:
        if stage == "tinystories":
            return iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
        if stage == "cosmopedia":
            return iter(load_dataset("BEE-spoke-data/cosmopedia-v2-mincols", split="train", streaming=True))
        raise ValueError(f"unknown streaming source: {stage}")

    def _put(self, item: object) -> None:
        while not self.closed.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _get(self) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        item = self.queue.get()
        if item is _END:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        step, batch = item
        return step, batch

    def _schedule_next_cuda_batch(self) -> None:
        assert self.copy_stream is not None
        try:
            step, (x_cpu, y_cpu) = self._get()
        except StopIteration:
            self.scheduled = None
            return
        with torch.cuda.stream(self.copy_stream):
            x_gpu = x_cpu.to(self.loader.device, non_blocking=True)
            y_gpu = y_cpu.to(self.loader.device, non_blocking=True)
        self.scheduled = (step, (x_gpu, y_gpu))

    def __iter__(self) -> "_StreamingTextTokenIterator":
        return self

    def __next__(self):
        if not self.use_cuda:
            step, batch = self._get()
            return (step, *batch) if self.loader.return_step else batch
        if self.scheduled is None:
            self.close()
            raise StopIteration
        assert self.copy_stream is not None
        current_stream = torch.cuda.current_stream(device=self.loader.device)
        current_stream.wait_stream(self.copy_stream)
        step, batch = self.scheduled
        for tensor in batch:
            tensor.record_stream(current_stream)
        self._schedule_next_cuda_batch()
        return (step, *batch) if self.loader.return_step else batch

    def close(self) -> None:
        self.closed.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)


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
                batch = batcher.train_batch(step)
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

    def _get(self) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        item = self.queue.get()
        if item is _END:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        step, batch = item
        return step, batch

    def _schedule_next_cuda_batch(self) -> None:
        assert self.copy_stream is not None
        try:
            step, (x_cpu, y_cpu) = self._get()
        except StopIteration:
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
            self.close()
            raise StopIteration

        assert self.copy_stream is not None
        current_stream = torch.cuda.current_stream(device=self.loader.device)
        current_stream.wait_stream(self.copy_stream)
        step, batch = self.scheduled
        for tensor in batch:
            tensor.record_stream(current_stream)

        self._schedule_next_cuda_batch()
        return (step, *batch) if self.loader.return_step else batch

    def close(self) -> None:
        self.closed.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)

    def __enter__(self) -> "_PrefetchingTokenIterator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_manifest_template(path: str | Path) -> None:
    manifest = TokenManifest(
        config=TokenDataConfig(),
        shards=[
            TokenShardSpec(
                path="data/cosmopedia_v2_mincols_train.uint16.bin",
                source="cosmopedia_v2_mincols",
                split="train",
                num_tokens=5_600_000_001,
                dataset_name="BEE-spoke-data/cosmopedia-v2-mincols",
                dataset_config="default",
                source_split="train",
            ),
            TokenShardSpec(
                path="data/cosmopedia_v2_mincols_val.uint16.bin",
                source="cosmopedia_v2_mincols_val",
                split="val",
                num_tokens=50_000_001,
                dataset_name="BEE-spoke-data/cosmopedia-v2-mincols",
                dataset_config="default",
                source_split="validation_proxy",
            ),
        ],
    )
    manifest.save(path)


def load_manifest(path: str | Path) -> TokenManifest:
    return TokenManifest.load(path)


def build_batcher(
    manifest_path: str | Path,
    *,
    root: str | Path = ".",
    rank: int = 0,
    world_size: int = 1,
    device: Optional[torch.device | str] = None,
) -> TokenBatcher:
    return TokenBatcher(
        load_manifest(manifest_path),
        root=root,
        rank=rank,
        world_size=world_size,
        device=device,
    )


def build_prefetching_train_loader(
    manifest: TokenManifest | str | Path,
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
) -> PrefetchingTokenLoader:
    resolved_manifest = manifest if isinstance(manifest, TokenManifest) else load_manifest(manifest)
    return PrefetchingTokenLoader(
        resolved_manifest,
        root=root,
        rank=rank,
        world_size=world_size,
        start_step=start_step,
        max_steps=max_steps,
        prefetch_batches=prefetch_batches,
        pin_memory=pin_memory,
        device=device,
        return_step=return_step,
    )


def _write_raw_tokens(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(path)


def _smoke_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seq_len = 8
        batch_size = 2
        config = TokenDataConfig(
            seq_len=seq_len,
            batch_size=batch_size,
            total_train_tokens=96,
            cosmopedia_stories_prefix_tokens=32,
            validation_tokens=32,
            prefix_source="tinystories",
        )

        _write_raw_tokens(root / "cosmo.bin", np.arange(0, 64, dtype=np.uint16))
        _write_raw_tokens(root / "main.bin", np.arange(1000, 1100, dtype=np.uint16))
        _write_raw_tokens(root / "val.bin", np.arange(2000, 2050, dtype=np.uint16))

        manifest = TokenManifest(
            config=config,
            shards=[
                TokenShardSpec("cosmo.bin", "tinystories", "train", 64),
                TokenShardSpec("main.bin", "main_mixture", "train", 100),
                TokenShardSpec("val.bin", "cosmopedia", "val", 50),
            ],
        )
        batcher = TokenBatcher(manifest, root=root)

        x0, y0 = batcher.train_batch(0)
        assert x0.shape == (batch_size, seq_len)
        assert y0.shape == (batch_size, seq_len)
        assert x0[0, 0].item() == 0
        assert y0[0, -1].item() == seq_len

        x2, _ = batcher.train_batch(2)
        assert x2[0, 0].item() == 1000
        assert batcher.curriculum_status(0)["stage"] == "tinystories_prefix"
        assert batcher.curriculum_status(2)["stage"] == "main_train"

        non_aligned_config = TokenDataConfig(
            seq_len=seq_len,
            batch_size=batch_size,
            total_train_tokens=96,
            cosmopedia_stories_prefix_tokens=30,
            validation_tokens=32,
            prefix_source="tinystories",
        )
        assert non_aligned_config.prefix_effective_tokens == 32

        val_x, val_y = next(batcher.val_batches(max_batches=1))
        assert val_x[0, 0].item() == 2000
        assert val_y[0, -1].item() == 2008

        loader = PrefetchingTokenLoader(
            manifest,
            root=root,
            max_steps=3,
            prefetch_batches=2,
            return_step=True,
        )
        seen_steps = []
        for step, x_batch, y_batch in loader:
            seen_steps.append(step)
            assert x_batch.shape == (batch_size, seq_len)
            assert y_batch.shape == (batch_size, seq_len)
        assert seen_steps == [0, 1, 2]

    print("data smoke test passed")


if __name__ == "__main__":
    _smoke_test()
