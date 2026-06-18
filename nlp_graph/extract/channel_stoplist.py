"""Per-channel and domain stoplists for B2B SaaS phrase extraction."""

from __future__ import annotations

import re
from typing import Iterable

from nlp_graph.extract.entities import B2B_ALLOWLIST

DOMAIN_STOPWORDS: frozenset[str] = frozenset(
    w.lower()
    for w in """
    great terrible good bad awesome awful amazing horrible excellent poor
    love hate liked disliked recommend recommended would definitely probably
    really very quite extremely totally absolutely honestly frankly
    easy hard simple difficult intuitive confusing
    use using used user users team teams company companies
    product products service services platform platforms tool tools
    software solution solutions feature features
    review reviews post posts comment comments thread
    thanks thank hello hi hey lol lmao imo tbh fwiw
    """.split()
)

FILLER_LEADING = re.compile(
    r"(?i)^(but|and|the|for|with|you|we|so|or|is|are|was|my|your|their|this|that|it|"
    r"have|has|can|will|just|more|our|they|what|how|when|where|who|why|if|an|in|on|at|to|of)\s+\w+"
)

JUNK_PHRASE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^(feels|sounds|looks) like\b"),
    re.compile(r"(?i)^(anyone else|need to|instead of|ended up|like you|trying to)\b"),
    re.compile(r"(?i)^(unpopular opinion|hot take|re rant|re is|re not|re just|re seeing)\b"),
    re.compile(r"(?i)^(re dbt|re paying|re still|re trying|re unpopular|re sagemaker|re trino|"
               r"re considering|re looking|re sticking)\b"),
]

LIGHT_VERB_JUNK = re.compile(
    r"(?i)^(designed to|able to|going to|having to|want to|need to|"
    r"used to|seem like|seems like|feel like|feels like|source of|kind of|sort of|"
    r"happy to|glad to|much more|way more|game-changer for|game changer for|"
    r"not set|set it|forget it|part of|out of|about the|get it|saw the|"
    r"forcing us|having a|re thoughts|thoughts on|much of|one of|some of)\b"
)

# Mid-phrase capitals that are common English, not brands.
_COMMON_CAPITALIZED: frozenset[str] = frozenset(
    w.lower()
    for w in """
    I We Our They This That When Where What How Why Who Which While After Before
    From Into Over Under About Between Through Without Within During Against
    """.split()
)


def _has_proper_noun_brand(phrase: str) -> bool:
    """True if phrase contains a brand-shaped proper noun (not sentence-initial common words).

    Relies on upstream lowercasing — extract/ngrams lowercases text before phrases reach
    this check. Do not call on unlowercased text; capitalized non-B2B words would pass.
    """
    words = phrase.split()
    for i, raw in enumerate(words):
        token = re.sub(r"[^\w'-]", "", raw)
        if len(token) < 2:
            continue
        # camelCase / mixedCase (BigQuery, dbtCloud, Snowflake mid-phrase)
        if any(c.isupper() for c in token[1:]):
            return True
        if token[0].isupper() and i > 0 and token.lower() not in _COMMON_CAPITALIZED:
            return True
    return False


def _term_in_phrase(term: str, phrase_lower: str, *, allow_substring: bool = False) -> bool:
    """Match terms with word boundaries; optional substring for plural domain stems (pipeline/pipelines)."""
    if " " in term or "-" in term:
        return term in phrase_lower
    if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", phrase_lower):
        return True
    if allow_substring and len(term) >= 6 and term in phrase_lower:
        return True
    return False


def has_content_token(phrase: str, tokens: list[str], stopwords: set[str]) -> bool:
    """True only if phrase contains an explicit B2B/domain signal or brand proper noun."""
    pl = phrase.lower()
    if pl in B2B_ALLOWLIST:
        return True
    if any(_term_in_phrase(a, pl) for a in B2B_ALLOWLIST):
        return True
    if any(_term_in_phrase(term, pl, allow_substring=True) for term in DOMAIN_SIGNAL_TERMS):
        return True
    if _has_proper_noun_brand(phrase):
        return True
    return False


DOMAIN_SIGNAL_TERMS = (
    "migrate",
    "migration",
    "pricing",
    "lakehouse",
    "serverless",
    "consumption",
    "governance",
    "warehouse",
    "catalog",
    "pipeline",
    "lock-in",
    "lock in",
    "vendor",
    "cloud",
    "query",
    "engine",
    "analytics",
    "platform",
    "snowflake",
    "databricks",
    "bigquery",
    "redshift",
    "trino",
    "firebolt",
    "dbt",
    "fabric",
    "starburst",
)

MAX_CORPUS_FREQ = 0.15

CHANNEL_STOPWORDS: dict[str, frozenset[str]] = {
    "reddit": frozenset("lol lmao imo tbh fwiw op edit update thanks upvote downvote throwaway imho ymmv".split()),
    "linkedin": frozenset(
        "excited thrilled humbled proud delighted share sharing thought leadership insights journey learnings".split()
    ),
    "g2": frozenset("stars star rating rated rate reviewer verified pros cons likeliness recommend".split()),
    "twitter": frozenset("rt thread threads tweet tweets follow following followers hashtag".split()),
    "unknown": frozenset(),
}

CHANNEL_PHRASE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "linkedin": [
        re.compile(r"(?i)excited to (share|announce)"),
        re.compile(r"(?i)thought leadership"),
    ],
    "g2": [
        re.compile(r"(?i)\d out of 5 stars"),
        re.compile(r"(?i)likelihood to recommend"),
    ],
}


def master_stoplist(extra: Iterable[str] | None = None) -> set[str]:
    words = set(DOMAIN_STOPWORDS)
    if extra:
        words.update(w.lower() for w in extra)
    return words


def channel_stoplist(source_channel: str | None) -> set[str]:
    key = (source_channel or "unknown").lower()
    if key in ("x", "twitter/x"):
        key = "twitter"
    base = master_stoplist()
    base.update(CHANNEL_STOPWORDS.get(key, CHANNEL_STOPWORDS["unknown"]))
    return base


def strip_channel_phrases(text: str, source_channel: str | None) -> str:
    key = (source_channel or "unknown").lower()
    if key in ("x", "twitter/x"):
        key = "twitter"
    out = text
    for pat in CHANNEL_PHRASE_PATTERNS.get(key, []):
        out = pat.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def is_corpus_frequency_junk(phrase: str, doc_freq: int, n_docs: int) -> bool:
    if n_docs <= 0:
        return False
    pl = phrase.lower()
    if pl in B2B_ALLOWLIST or any(a in pl for a in B2B_ALLOWLIST if len(a) > 3):
        return False
    return (doc_freq / n_docs) > MAX_CORPUS_FREQ


def is_filler_for_metrics(phrase: str) -> bool:
    """Independent B2B-signal classifier for checkpoint filler-rate measurement."""
    tokens = phrase.lower().split()
    if len(tokens) < 2:
        return True
    if FILLER_LEADING.match(phrase) or phrase.lower().startswith("trying to"):
        return True
    if LIGHT_VERB_JUNK.search(phrase):
        return True
    if any(pat.search(phrase) for pat in JUNK_PHRASE_PATTERNS):
        return True
    if re.match(r"(?i)^(re |part |kind |sort |much |way |one |some |get |saw )", phrase):
        return True
    if not has_content_token(phrase, tokens, master_stoplist()):
        return True
    return False


def is_junk_phrase(
    phrase: str,
    stopwords: set[str],
    *,
    doc_freq: int = 0,
    n_docs: int = 0,
) -> bool:
    tokens = phrase.lower().split()
    if len(tokens) < 2:
        return True
    if FILLER_LEADING.match(phrase) or phrase.lower().startswith("trying to"):
        return True
    if LIGHT_VERB_JUNK.search(phrase):
        return True
    if any(pat.search(phrase) for pat in JUNK_PHRASE_PATTERNS):
        return True
    if all(t in stopwords for t in tokens):
        return True
    if len(phrase) < 4:
        return True
    if any(_term_in_phrase(k, phrase.lower(), allow_substring=True) for k in DOMAIN_SIGNAL_TERMS):
        return False
    if tokens[0] in stopwords and tokens[-1] in stopwords:
        return True
    if is_corpus_frequency_junk(phrase, doc_freq, n_docs):
        return True
    return False
