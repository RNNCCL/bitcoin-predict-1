import re
import urlparse
import operator as op
import datetime as dt
import itertools as it

import simplejson as json
import bs4
import nltk
import mistune

import util


def load_json(p=None, include_is=True, include_oos=False, filter_deleted=True):
    assert include_is or include_oos
    files = []
    if include_is:
        files.append("data-comments/all_btc.jsonl")
    if include_oos:
        files.append("data-comments/oos_btc.jsonl")

    filter_f = (
        (lambda d: d["body"] != "[deleted]")
        if filter_deleted
        else (lambda _: True)
    )

    return util.sample(
        it.ifilter(
            filter_f,
            it.chain.from_iterable(
                it.imap(lambda f: (json.loads(l) for l in open(f)), files)
            )
        ),
        p
    )


def get_normalized_texts(**load_json_args):
    preproc = TextPreProcessor()
    return (preproc.preprocess_text(d["body"]) for d in load_json(**load_json_args))


def get_processed_comments(**load_json_args):
    preproc = TextPreProcessor()
    for d in load_json(**load_json_args):
        text = preproc.preprocess_text(d["body"])
        for sent in process(text):
            yield sent


def get_processed_full_comments(**load_json_args):
    preproc = TextPreProcessor()
    for d in load_json(**load_json_args):
        text = preproc.preprocess_text(d["body"])
        yield process(text)


get_preprocessed_comments = get_normalized_texts


def get_day_preprocessed_comments(**load_json_args):
    def _get():
        preproc = TextPreProcessor()
        for d in load_json(**load_json_args):
            text = preproc.preprocess_text(d["body"])
            date = dt.datetime.utcfromtimestamp(float(d["created_utc"])).date()
            yield date, text.strip()

    for k, k_vs in it.groupby(_get(), lambda x: x[0]):
        yield k, (k_v[1] for k_v in k_vs)


def process(t):
    sent_toks = map(
        nltk.tokenize.wordpunct_tokenize,
        nltk.sent_tokenize(t)
    )
    return [map(op.methodcaller("lower"), sent) for sent in sent_toks]


class RegexTransformer():
    def __init__(self, regex, transformer, extractor):
        self.regex = regex
        self.transformer = transformer
        self.extractor = extractor

    def transform(self, s):
        new_s_l = []
        cur = 0
        for match in re.finditer(self.regex, s):
            new_s_l.append(s[cur: match.start()])
            new_s_l.append(self.transformer(self.extractor(match)))
            cur = match.end()
        new_s_l.append(s[cur: len(s)])
        return u"".join(new_s_l)


def convert_url(u):
    try:
        parsed = urlparse.urlparse(u)
        return parsed.netloc.replace(".", "") # + parsed.path.replace("/", "_slash_")
    except ValueError:
        return "invalidurl"


class TextPreProcessor():
    def __init__(self):
        self.md = mistune.Markdown()
        self.rts = [
            RegexTransformer(
                re.compile(
                    'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                ),
                convert_url,
                lambda x: x.group()
            ),
            RegexTransformer(
                re.compile("[/]?r/([0-9A-Za-z_]+)"),
                lambda subreddit: "subreddit" + subreddit,
                lambda x: x.groups()[0]
            ),
            RegexTransformer(
                re.compile("[/]?u/([0-9A-Za-z_-]+)"),
                lambda user: "user" + user,
                lambda x: x.groups()[0]
            )
        ]

    def preprocess_text(self, t):
        t = bs4.BeautifulSoup(self.md.parse(t)).text
        for rt in self.rts:
            t = rt.transform(t)
        return t


def ngram_iter(tokens, max_N=3):
    for N in range(1, max_N+1):
        ngrams = nltk.ngrams(tokens, N, pad_right=True, pad_left=True, pad_symbol=u"<S>")
        for ngram in ngrams:
            yield ngram


def bigrams_iter():
    for sent_tokens in get_processed_comments():
        for ngram in nltk.ngrams(sent_tokens, 2, pad_right=True, pad_left=True, pad_symbol=u"<S>"):
            yield ngram
