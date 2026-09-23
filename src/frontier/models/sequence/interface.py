"""State and accounting contracts shared by causal sequence modules."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol

import torch
from torch import nn

from frontier.config import ModelConfig


@dataclass(frozen=True)
class AttentionState:
    """Persistent key/value tensors and the absolute position of the first key."""

    key: torch.Tensor
    value: torch.Tensor
    key_start_position: int = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Allow the historical ``for key, value in state`` cache idiom."""
        yield self.key
        yield self.value

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int | slice) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return (self.key, self.value)[index]


@dataclass(frozen=True)
class RecurrentState:
    """Container for a recurrent module's nested persistent tensor state."""

    values: object


KVCache = AttentionState
SequenceState = Any
SequenceFactory = Callable[[ModelConfig], nn.Module]
_REGISTRY: dict[str, SequenceFactory] = {}


@dataclass(frozen=True)
class SequenceCostDescriptor:
    """Declared compute semantics; estimates remain owned by versioned estimators."""

    training_estimator: str | None
    executed_work: str
    notes: str


@dataclass(frozen=True)
class SequenceStateDescriptor:
    """Declared persistent-state type, growth rule, and active-region convention."""

    kind: str
    storage_model: str
    growth: str
    active_regions: str


@dataclass(frozen=True)
class SequenceModuleDescriptor:
    """Machine-readable cost and state contract for one configured sequence module."""

    name: str
    cost: SequenceCostDescriptor
    state: SequenceStateDescriptor
    parameters: tuple[tuple[str, int | float | str | bool], ...] = ()


@dataclass
class DecoderCache:
    """Per-layer state plus the absolute position for the next input token."""

    states: list[SequenceState | None]
    position: int


class SequenceModule(Protocol):
    """A causal sequence mixer with an inspectable optional decode state."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state: SequenceState | None = None,
        use_cache: bool = False,
        position_start: int | None = None,
    ) -> tuple[torch.Tensor, SequenceState | None]: ...

    def state_tensors(self, state: SequenceState) -> object:
        """Return the nested tree backing allocated persistent sequence state."""
        ...

    def active_state_tensors(self, state: SequenceState) -> object:
        """Return a nested tree of contiguous views for logically active state."""
        ...

    def sequence_descriptor(self) -> SequenceModuleDescriptor:
        """Describe executed work and persistent-state accounting semantics."""
        ...


def iter_state_tensors(value: object) -> Iterator[torch.Tensor]:
    """Recursively visit tensors in dataclasses, mappings, tuples, and lists.

    The traversal is deterministic, tolerates repeated references and cycles in
    supported containers, and raises for opaque objects instead of silently
    treating unknown state as zero bytes. Tensor storage aliases are yielded;
    callers that count memory must union their backing-storage ranges.
    """

    visited_containers: set[int] = set()
    visited_tensors: set[int] = set()

    def visit(item: object) -> Iterator[torch.Tensor]:
        if isinstance(item, torch.Tensor):
            identity = id(item)
            if identity not in visited_tensors:
                visited_tensors.add(identity)
                yield item
            return
        if item is None or isinstance(item, (str, bytes, int, float, complex, bool)):
            return

        identity = id(item)
        if identity in visited_containers:
            return
        if isinstance(item, Mapping):
            visited_containers.add(identity)
            for nested in item.values():
                yield from visit(nested)
            return
        if isinstance(item, (tuple, list)):
            visited_containers.add(identity)
            for nested in item:
                yield from visit(nested)
            return
        if is_dataclass(item) and not isinstance(item, type):
            visited_containers.add(identity)
            for field in fields(item):
                yield from visit(getattr(item, field.name))
            return
        raise TypeError(f"unsupported opaque sequence state value: {type(item).__name__}")

    yield from visit(value)


def describe_sequence_module(module: nn.Module) -> SequenceModuleDescriptor:
    """Read and validate an instantiated module's declared accounting contract."""

    describe = getattr(module, "sequence_descriptor", None)
    if not callable(describe):
        raise TypeError(f"{type(module).__name__} does not expose sequence_descriptor")
    descriptor = describe()
    if not isinstance(descriptor, SequenceModuleDescriptor):
        raise TypeError("sequence_descriptor must return SequenceModuleDescriptor")
    return descriptor


def register_sequence_module(name: str, factory: SequenceFactory) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"sequence module name is empty or already registered: {name!r}")
    _REGISTRY[name] = factory


def build_sequence_module(name: str, config: ModelConfig) -> nn.Module:
    try:
        factory = _REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(
            f"sequence module {name!r} is not registered (available: {available})"
        ) from error
    return factory(config)
