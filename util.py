import logging
import random


class stream_log(object):
    def __init__(self, name, n, stream):
        self.idx = 0
        self.name = name
        self.stream = stream
        self.n = n

    def __iter__(self):
        for el in self.stream:
            if self.idx % self.n == 0:
                logging.info("{} {}".format(self.name, self.idx))
            self.idx += 1
            yield el


def sample(it, p):
    for el in it:
        if p is None or random.random() < p:
            yield el
