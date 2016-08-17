import logging
import random
import scipy.sparse as sparse


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


def scale(self):
    return (self - self.mean()) / self.std()


def d_to_sparse(d, shape=None):
    data = []
    row = []
    col = []
    for (doc_idx, word_idx), value in d.viewitems():
        data.append(value)
        row.append(doc_idx)
        col.append(word_idx)
    if shape is None:
        return sparse.coo_matrix((data, (row, col)))
    else:
        return sparse.coo_matrix((data, (row, col)), shape=shape)


def join(*dfs):
    df = dfs[0]
    for df_other in dfs[1:]:
        df = df.join(df_other, how="inner", rsuffix="_other")
    return df
