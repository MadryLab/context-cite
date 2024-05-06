import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
from abc import ABC, abstractmethod
from .utils import split_text


class BaseContextPartitioner(ABC):
    """
    A base class for partitioning a context into sources.

    Attributes:
        context (str):
            The context to partition.

    Methods:
        num_sources(self) -> int:
            Property. The number of sources within the context.
        split_context(self) -> None:
            Split the context into sources.
        get_source(self, index: int) -> str:
            Get a represention of the source corresponding to a given index.
        get_context(self, mask: Optional[NDArray] = None) -> str:
            Get a version of the context ablated according to the given mask.
        sources(self) -> List[str]:
            Property. A list of all sources within the context.
    """

    def __init__(self, context: str) -> None:
        self.context = context

    @property
    @abstractmethod
    def num_sources(self) -> int:
        """The number of sources."""

    @abstractmethod
    def split_context(self) -> None:
        """Split the context into sources."""

    @abstractmethod
    def get_source(self, index: int) -> str:
        """Get a represention of the source corresponding to a given index."""

    @abstractmethod
    def get_context(self, mask: Optional[NDArray] = None):
        """Get a version of the context ablated according to the given mask."""

    @property
    def sources(self) -> List[str]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]


class SimpleContextPartitioner(BaseContextPartitioner):
    """
    A simple context partitioner that splits the context into sources based on
    a separator.
    """

    def __init__(self, context: str, source_type: str = "sentence") -> None:
        super().__init__(context)
        self.source_type = source_type
        self._cache = {}

    def split_context(self):
        """Split text into parts and cache the parts and separators."""
        parts, separators, _ = split_text(self.context, self.source_type)
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property
    def parts(self):
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[NDArray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        parts = np.array(self.parts)[mask]
        context = ""
        for i, (separator, part) in enumerate(zip(separators, parts)):
            if i > 0:
                context += separator
            context += part
        return context
