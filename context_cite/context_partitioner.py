import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod
from .utils import split_into_sentences, split_into_words


class BaseContextPartitioner(ABC):
    def __init__(self, context: str) -> None:
        self.context = context

    @property
    @abstractmethod
    def num_sources(self) -> int:
        """The number of sources."""

    @abstractmethod
    def get_source(self, index: int) -> str:
        """Get a represention of the source corresponding to a given index."""

    @abstractmethod
    def get_context(self, mask: Optional[np.ndarray] = None):
        """Get a version of the context ablated according to the given mask."""

    @property
    def sources(self) -> List[str]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]


class SentenceContextPartitioner(BaseContextPartitioner):
    def __init__(self, context: str) -> None:
        super().__init__(context)
        self._cache = {}

    @property
    def sentences(self):
        if self._cache.get("sentences") is None:
            self._cache["sentences"], self._cache["separators"] = split_into_sentences(
                self.context
            )
        return self._cache["sentences"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self._cache["sentences"], self._cache["separators"] = split_into_sentences(
                self.context
            )
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.sentences)

    def get_source(self, index: int) -> str:
        return self.sentences[index]

    def get_context(self, mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        sentences = np.array(self.sentences)[mask]
        context = ""
        for i, (separator, sentence) in enumerate(zip(separators, sentences)):
            if i > 0:
                context += separator
            context += sentence
        return context


class WordContextPartitioner(BaseContextPartitioner):
    def __init__(self, context: str) -> None:
        super().__init__(context)
        self._cache = {}

    @property
    def words(self) -> List[str]:
        if self._cache.get("words") is None:
            self._cache["words"], self._cache["separators"] = split_into_words(
                self.context
            )
        return self._cache["words"]

    @property
    def separators(self) -> List[str]:
        if self._cache.get("separators") is None:
            self._cache["words"], self._cache["separators"] = split_into_words(
                self.context
            )
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.words)

    def get_source(self, index: int) -> str:
        return self.words[index]

    def get_context(self, mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        words = np.array(self.words)[mask]
        context = ""
        for i, (separator, word) in enumerate(zip(separators, words)):
            if i > 0:
                context += separator
            context += word


PARTITION_TYPE_TO_PARTITIONER = {
    "sentence": SentenceContextPartitioner,
    "word": WordContextPartitioner,
}
