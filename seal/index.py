# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import pickle
import struct
import tempfile
from typing import List, Set, Tuple, Optional, Iterable, Iterator

from .cpp_modules.fm_index import FMIndex as _FMIndex
from .cpp_modules.fm_index import load_FMIndex

SHIFT = 10
BUFSZ = 1_000_000
FORMAT = '<l'

class FMIndex(_FMIndex):
    """
    FMIndex class that interfaces with the low-level `sdsl-lite` implementation.
    """

    beginnings: List[int]
    occurring: Set[int]
    occurring_distinct: List[int]
    occurring_counts: List[int]
    labels: Optional[List[str]]

    def __init__(self):
        super().__init__()
        self.beginnings = [0]
        self.occurring = set()
        self.occurring_distinct = []
        self.occurring_counts = []
        self.labels = None

    def initialize(self, sequences: Iterable[List[int]], in_memory: bool = False) -> None:
        """
        Initialize the FM-index.
        Params:
            sequences: An iterable of list of integers, e.g. token ids.
            in_memory: If False, builds the FM-index using a temporary cache file
        """
        occurring = set()
        if in_memory:
            data = []
            for seq in sequences:
                self.beginnings.append(self.beginnings[-1] + len(seq))
                occurring |= set(seq)
                seq = [x + SHIFT for x in seq[::-1]]
                data.extend(seq)
            self.occurring = list(occurring)
            super().initialize(data)
        else:
            with tempfile.NamedTemporaryFile() as tmp:
                for seq in sequences:
                    self.beginnings.append(self.beginnings[-1] + len(seq))
                    occurring |= set(seq)
                    seq = [x + SHIFT for x in seq[::-1]]
                    tmp.write(b''.join([struct.pack(FORMAT, x) for x in seq]))
                tmp.flush()
                self.occurring = list(occurring)
                super().initialize_from_file(tmp.name, 4)
        self.occurring_distinct, self.occurring_counts = self.get_distinct_count(0, len(self))

    def get_doc(self, doc_index: int) -> List[int]:
        """
        Returns the document (as a list of ids) given its index in the index.
        """
        doc = self.extract_text(
            self.beginnings[doc_index], self.beginnings[doc_index+1])
        doc = [x - SHIFT for x in doc]
        return doc

    def get_doc_index(self, token_index: int) -> int:
        """
        Returns the index of the document containing the token identified by the input index.
        """
        doc_index = bisect.bisect_right(self.beginnings, token_index) - 1
        return doc_index

    def get_doc_length(self, doc_index: int) -> int:
        """
        Returns the length of the document matching `doc_index`.
        """
        return self.beginnings[doc_index + 1] - self.beginnings[doc_index]

    def get_token_index_from_row(self, row: int) -> int:
        """
        Locates a range of FM-index rows in the corpus.
        """
        return self.locate(row)

    def get_doc_index_from_row(self, row: int) -> int:
        """
        Returns the `doc_index` of the document containing the token in the input row of the Wavelet Tree.
        """
        return self.get_doc_index(self.locate(row))

    def get_range(self, sequence: List[int]) -> Tuple[int, int]:
        """
        Finds the FM-index rows that match the input prefix `sequence`.
        """
        start_row = 0
        end_row = self.size()
        for token in sequence:
            start_row, end_row = self.backward_search_step(token + SHIFT, start_row, end_row)
        end_row += 1
        return start_row, end_row

    def get_count(self, sequence: List[int]) -> int:
        """
        Counts the number of occurrences of the input prefix `sequence` in the FM-index.
        """
        start, end = self.get_range(sequence)
        return end - start

    def get_doc_indices(self, sequence: List[int]) -> Iterator[int]:
        """
        Finds the documents that contain the input prefix `sequence`.
        """
        start, end = self.get_range(sequence)
        for row in range(start, end):
            yield self.get_doc_index_from_row(row)

    def get_continuations(self, sequence: List[int]) -> List[int]:
        """
        Finds all tokens that appear at least once as successors for the input prefix.
        """
        start, end = self.get_range(sequence)
        conts = self.get_distinct(start, end)
        return conts

    def get_distinct(self, low: int, high: int) -> List[int]:
        """
        Finds all distinct symbols that appear in the last column of the FM-index in a given range.
        """
        distinct = self.distinct(low, high)
        distinct = [c - SHIFT for c in distinct if c > 0]
        return distinct

    def get_distinct_count(self, low: int, high: int) -> Tuple[List[int], List[int]]:
        """
        Finds all distinct symbols that appear in the last column of the FM-index in a given range, and also return their
        counts.
        """
        data = self.distinct_count(low, high)
        distinct = []
        counts = []
        for d, c in zip(data[0::2], data[1::2]):
            if d > 0:
                distinct.append(d - SHIFT)
                counts.append(c)
        return distinct, counts

    def get_distinct_count_multi(self, lows: List[int], highs: List[int]) -> List[Tuple[List[int], List[int]]]:
        """
        Multithreaded version of `get_distinct_count`.
        """
        ret = []
        for data in self.distinct_count_multi(lows, highs):
            distinct = []
            counts = []
            for d, c in zip(data[0::2], data[1::2]):
                if d > 0:
                    distinct.append(d - SHIFT)
                    counts.append(c)
            ret.append((distinct, counts))
        return ret

    def __len__(self) -> int:
        """
        FM-index length (in tokens).
        """
        return self.beginnings[-1]

    @property
    def n_docs(self) -> int:
        """
        Number of documents in the FM-index.
        """
        return len(self.beginnings) - 1

    def save(self, path: str) -> None:
        """
        Serialize the FM-index at the given path.
        """
        with open(path + '.oth', 'wb') as f:
            pickle.dump((self.beginnings, self.occurring, self.labels), f)
        return super().save(path + '.fmi')

    @classmethod
    def load(cls, path: str) -> 'FMIndex':
        """
        Initialize the FM-index from the given path.
        """
        index = load_FMIndex(path + '.fmi')
        index.__class__ = cls
        with open(path + '.oth', 'rb') as f:
            index.beginnings, index.occurring, index.labels = pickle.load(f)
        index.occurring_distinct, index.occurring_counts = index.get_distinct_count(0, len(index))
        return index
    