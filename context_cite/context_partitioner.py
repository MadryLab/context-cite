import nltk
import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod


class BaseContextPartitioner(ABC):
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

    def split_context(self):
        """Split text into sentences and cache the sentences and separators."""
        sentences = []
        separators = []

        # first split by newlines
        lines = self.context.splitlines()
        for line in lines:
            sentences.extend(nltk.sent_tokenize(line))

        cur_start = 0
        for sentence in sentences:
            cur_end = self.context.find(sentence, cur_start)
            separators.append(self.context[cur_start:cur_end])
            cur_start = cur_end + len(sentence)

        self._cache["sentences"] = sentences
        self._cache["separators"] = separators

    @property
    def sentences(self):
        if self._cache.get("sentences") is None:
            self.split_context()
        return self._cache["sentences"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
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

    def split_context(self):
        """Split the context into words and cache the words and separators."""
        separators = []
        words = nltk.word_tokenize(self.context)

        cur_start = 0
        for word in words:
            cur_end = self.context.find(word, cur_start)
            separators.append(self.context[cur_start:cur_end])
            cur_start = cur_end + len(word)

        self._cache["words"] = words
        self._cache["separators"] = separators

    @property
    def words(self) -> List[str]:
        if self._cache.get("words") is None:
            self.split_context()
        return self._cache["words"]

    @property
    def separators(self) -> List[str]:
        if self._cache.get("separators") is None:
            self.split_context()
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
