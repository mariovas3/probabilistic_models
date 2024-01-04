from collections.abc import Iterable


class RecencyStore(Iterable):
    def __init__(self, max_len=10):
        self.max_len = max_len
        self.i = 0
        self.store = []

    def add_item(self, item):
        if len(self.store) < self.max_len:
            self.store.append(item)
        else:
            self.store[self.i] = item
        self.i = (self.i + 1) % self.max_len

    def __iter__(self):
        if len(self.store) == 0:
            yield None
        else:
            i = self.i - 1
            for j in range(len(self.store)):
                pos = (i - j) % len(self.store)
                yield self.store[pos]

    def __len__(self):
        return len(self.store)
