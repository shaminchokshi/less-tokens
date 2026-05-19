"""
less_tokens.compressor — the eleven lexical compression techniques.

All techniques are pure functions over text and are deterministic. They are
applied in a fixed canonical order (multi-word transforms first, structural
cleanups last) so that downstream stages do not undo upstream changes.

The orchestrator is :func:`compress`. End users should normally call
:func:`less_tokens.compress` from the package root instead.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------
# We import lazily and download missing resources on first call so the user
# does not need a manual `nltk.download(...)` step.

_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    needed = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
        ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
        ("corpora/words", "words"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # Some packages don't exist on older nltk versions; carry on.
                pass
    _NLTK_READY = True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Negations and wh-words are NEVER removed: dropping them flips meaning.
NEGATIONS = {"not", "no", "nor", "never", "neither", "n't", "nothing", "nobody",
             "nowhere", "none", "without", "cannot"}
WH_WORDS = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}

# Filler phrases removed at the start of the pipeline. Word boundaries protect
# against false positives ("could you" inside another phrase).
FILLER_PHRASES = [
    r"\bi\s+was\s+wondering\s+if\s+you\s+could\b",
    r"\bi\s+would\s+like\s+to\s+know\b",
    r"\bcould\s+you\s+please\b",
    r"\bcan\s+you\s+please\b",
    r"\bwould\s+you\s+mind\b",
    r"\bif\s+(?:it\s+is\s+|it'?s\s+)?not\s+too\s+much\s+trouble\b",
    r"\bif\s+you\s+don'?t\s+mind\b",
    r"\bplease\s+kindly\b",
    r"\bkindly\s+please\b",
    r"\bi\s+would\s+appreciate\s+it\s+if\s+you\s+could\b",
    r"\bplease\s+help\s+me\s+(?:to\s+)?understand\b",
    r"\bcan\s+you\s+help\s+me\b",
    r"\bplease\s+provide\b",
    r"\bplease\s+explain\b",
]

ABBREVIATIONS = {
    "for example":          "e.g.",
    "for instance":         "e.g.",
    "that is":              "i.e.",
    "and so on":            "etc.",
    "and so forth":         "etc.",
    "as soon as possible":  "asap",
    "as much as possible":  "as much as poss.",
    "in other words":       "i.e.",
    "with respect to":      "re:",
    "with regard to":       "re:",
    "in terms of":          "re:",
    "approximately":        "approx.",
    "et cetera":            "etc.",
}

CONTRACTIONS = {
    "do not":     "don't",  "does not": "doesn't", "did not": "didn't",
    "is not":     "isn't",  "are not":  "aren't",  "was not": "wasn't",
    "were not":   "weren't","cannot":   "can't",   "could not": "couldn't",
    "would not":  "wouldn't","should not":"shouldn't","will not": "won't",
    "have not":   "haven't","has not":  "hasn't",  "had not": "hadn't",
    "i am":       "I'm",    "you are":  "you're",  "we are":  "we're",
    "they are":   "they're","it is":    "it's",    "that is": "that's",
    "what is":    "what's", "there is": "there's", "let us":  "let's",
    "i have":     "I've",   "you have": "you've",  "we have": "we've",
    "they have":  "they've","i will":   "I'll",    "you will":"you'll",
    "we will":    "we'll",  "they will":"they'll", "i would": "I'd",
    "you would":  "you'd",  "he would": "he'd",    "she would":"she'd",
}

FILLER_WORDS = {"basically", "actually", "really", "just", "kind", "sort",
                "somewhat", "perhaps", "maybe", "literally", "obviously",
                "essentially", "definitely", "absolutely", "honestly",
                "frankly", "simply", "merely", "quite", "rather"}

# Articles and common auxiliaries. The space-separated form is for the
# unambiguous "word + space" boundary check during pruning.
ARTICLES_AUX = {"a", "an", "the", "is", "am", "are", "was", "were", "be",
                "been", "being", "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "may", "might", "must",
                "can", "shall"}

# POS tags we keep under the "caveman" mode. NLTK uses Penn Treebank tags:
# NN* = nouns, VB* = verbs, JJ* = adjectives, CD = numerals, WP/WRB/WDT = wh.
POS_KEEP = {"NN", "NNS", "NNP", "NNPS",
            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
            "JJ", "JJR", "JJS",
            "CD",
            "WP", "WP$", "WRB", "WDT",
            "RB"}  # also keep most adverbs — they often carry meaning

# Mapping from Penn POS to WordNet POS for lemmatization.
_WN_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}


def _penn_to_wn(tag: str) -> str:
    return _WN_POS.get(tag[:1], "n")


# ---------------------------------------------------------------------------
# Individual techniques (pure functions over text)
# ---------------------------------------------------------------------------

def _remove_filler_phrases(text: str) -> str:
    out = text
    for pat in FILLER_PHRASES:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out


def _apply_abbreviations(text: str) -> str:
    # Replace longest first so "for example" wins over "for".
    out = text
    for phrase, abbr in sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(phrase)}\b", abbr, out, flags=re.IGNORECASE)
    return out


def _apply_contractions(text: str) -> str:
    out = text
    for phrase, contraction in sorted(CONTRACTIONS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(phrase)}\b", contraction, out, flags=re.IGNORECASE)
    return out


def _remove_filler_words(text: str) -> str:
    out = text
    for word in FILLER_WORDS:
        out = re.sub(rf"\b{word}\b", "", out, flags=re.IGNORECASE)
    return out


def _detect_named_entity_spans(tokens: List[str]) -> set:
    """Return indices (0-based, into the token list) that fall inside an NE span."""
    _ensure_nltk()
    import nltk
    try:
        tagged = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(tagged, binary=True)
    except Exception:
        return set()

    protected = set()
    idx = 0
    for chunk in chunks:
        if hasattr(chunk, "label"):
            n = len(chunk)
            for k in range(n):
                protected.add(idx + k)
            idx += n
        else:
            idx += 1
    return protected


def _remove_stopwords(text: str, preserve_ne: bool = True) -> str:
    _ensure_nltk()
    import nltk
    from nltk.corpus import stopwords as _sw
    sw = set(_sw.words("english")) - NEGATIONS - WH_WORDS

    tokens = nltk.word_tokenize(text)
    protected = _detect_named_entity_spans(tokens) if preserve_ne else set()

    out = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if i in protected:
            out.append(tok); continue
        if low in sw:
            continue
        out.append(tok)
    return _detokenize(out)


def _remove_function_words(text: str, preserve_ne: bool = True) -> str:
    _ensure_nltk()
    import nltk
    tokens = nltk.word_tokenize(text)
    protected = _detect_named_entity_spans(tokens) if preserve_ne else set()

    out = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if i in protected:
            out.append(tok); continue
        if low in ARTICLES_AUX and low not in NEGATIONS:
            continue
        out.append(tok)
    return _detokenize(out)


def _pos_keep_only(text: str, preserve_ne: bool = True) -> str:
    _ensure_nltk()
    import nltk
    tokens = nltk.word_tokenize(text)
    if not tokens:
        return text
    protected = _detect_named_entity_spans(tokens) if preserve_ne else set()
    tagged = nltk.pos_tag(tokens)

    out = []
    for i, (tok, tag) in enumerate(tagged):
        low = tok.lower()
        if i in protected or low in NEGATIONS or low in WH_WORDS:
            out.append(tok); continue
        if tag in POS_KEEP:
            out.append(tok); continue
        # punctuation is dropped here; normalize step re-adds question mark
    return _detokenize(out)


def _lemmatize(text: str) -> str:
    _ensure_nltk()
    import nltk
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    if not tokens:
        return text
    tagged = nltk.pos_tag(tokens)
    out = [lem.lemmatize(tok, _penn_to_wn(tag)) for tok, tag in tagged]
    return _detokenize(out)


def _shorten_synonyms(text: str, tokenizer=None) -> str:
    """For each long content word, try WordNet synonyms; keep shortest by token count."""
    _ensure_nltk()
    import nltk
    from nltk.corpus import wordnet as wn

    if tokenizer is None:
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            tokenizer = None

    def n_tokens(s: str) -> int:
        if tokenizer is None:
            return len(s)
        return len(tokenizer.encode(s))

    tokens = nltk.word_tokenize(text)
    out = []
    for tok in tokens:
        if (len(tok) < 5 or not tok.isalpha() or tok.lower() in NEGATIONS):
            out.append(tok); continue
        best = tok
        best_n = n_tokens(tok)
        try:
            for syn in wn.synsets(tok.lower()):
                for lemma in syn.lemmas():
                    cand = lemma.name().replace("_", " ")
                    if " " in cand:
                        continue
                    if cand.lower() == tok.lower():
                        continue
                    cn = n_tokens(cand)
                    if cn < best_n:
                        best, best_n = cand, cn
        except Exception:
            pass
        out.append(best)
    return _detokenize(out)


def _normalize_whitespace_punct(text: str) -> str:
    out = text
    out = re.sub(r"\s+", " ", out)             # collapse runs of whitespace
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)  # no space before punctuation
    out = re.sub(r"([.!?;:,])\1+", r"\1", out)  # collapse repeated punctuation
    out = re.sub(r"[—–]+", "-", out)            # em/en dashes -> hyphen
    return out.strip()


# ---------------------------------------------------------------------------
# Detokenization (NLTK's TreebankWordDetokenizer with light cleanup)
# ---------------------------------------------------------------------------

def _detokenize(tokens: List[str]) -> str:
    _ensure_nltk()
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    if not tokens:
        return ""
    return TreebankWordDetokenizer().detokenize(tokens)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

#: All eleven technique flags (canonical order of application).
TECHNIQUES: Tuple[str, ...] = (
    "remove_filler_phrases",
    "apply_abbreviations",
    "apply_contractions",
    "remove_filler_words",
    "remove_stopwords",
    "remove_function_words",
    "pos_keep_only",
    "lemmatize",
    "shorten_synonyms",
    "preserve_named_entities",       # modifier, not a stage of its own
    "normalize_whitespace_punct",
)


def _coerce_flag(v) -> bool:
    """Accept ints (0/1), bools, or strings ('0','1','on','off') as boolean toggles."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on", "y"}:
            return True
        if s in {"0", "false", "no", "off", "n", ""}:
            return False
    return bool(v)


def compress(
    prompt: str,
    *,
    remove_filler_phrases: int = 0,
    apply_abbreviations: int = 0,
    apply_contractions: int = 0,
    remove_filler_words: int = 0,
    remove_stopwords: int = 0,
    remove_function_words: int = 0,
    pos_keep_only: int = 0,
    lemmatize: int = 0,
    shorten_synonyms: int = 0,
    preserve_named_entities: int = 1,
    normalize_whitespace_punct: int = 1,
    return_trace: bool = False,
) -> "str | Dict":
    """
    Compress a prompt using lexical NLP techniques.

    Each technique is toggled via a 0/1 flag (1 = enabled, 0 = disabled).
    Bools and strings like ``"on"`` / ``"off"`` are also accepted.

    Parameters
    ----------
    prompt
        The input prompt text.
    remove_filler_phrases
        Strip hedging phrases like "I was wondering if you could".
    apply_abbreviations
        Substitute short forms: "for example" → "e.g." etc.
    apply_contractions
        Substitute contractions: "do not" → "don't" etc.
    remove_filler_words
        Drop single-word fillers: "basically", "really", "just", ...
    remove_stopwords
        Drop NLTK English stopwords (negations and wh-words always kept).
    remove_function_words
        Drop articles ("a","an","the") and auxiliaries.
    pos_keep_only
        Aggressive: keep only nouns/verbs/adjectives/numerals/wh/negations.
    lemmatize
        WordNet-aware lemmatization ("running" → "run").
    shorten_synonyms
        Replace long words with shorter WordNet synonyms.
    preserve_named_entities
        Protect NE spans from stopword/function-word pruning. (Default on.)
    normalize_whitespace_punct
        Collapse whitespace and deduplicate punctuation. (Default on.)
    return_trace
        If True, return a dict with the compressed prompt plus the
        intermediate text after each enabled stage. Useful for debugging.

    Returns
    -------
    str
        The compressed prompt (default).
    dict
        If ``return_trace=True``: ``{"compressed": str, "trace": [...]}``.

    Examples
    --------
    >>> from less_tokens import compress
    >>> compress("I was wondering if you could please explain how do I run "
    ...          "a python script", remove_filler_phrases=1,
    ...          apply_contractions=1, remove_stopwords=1)
    'explain run python script'
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")

    flags = {
        "remove_filler_phrases":      _coerce_flag(remove_filler_phrases),
        "apply_abbreviations":        _coerce_flag(apply_abbreviations),
        "apply_contractions":         _coerce_flag(apply_contractions),
        "remove_filler_words":        _coerce_flag(remove_filler_words),
        "remove_stopwords":           _coerce_flag(remove_stopwords),
        "remove_function_words":      _coerce_flag(remove_function_words),
        "pos_keep_only":              _coerce_flag(pos_keep_only),
        "lemmatize":                  _coerce_flag(lemmatize),
        "shorten_synonyms":           _coerce_flag(shorten_synonyms),
        "preserve_named_entities":    _coerce_flag(preserve_named_entities),
        "normalize_whitespace_punct": _coerce_flag(normalize_whitespace_punct),
    }

    was_question = prompt.strip().endswith("?")
    text = prompt
    trace = []

    def _stage(name, fn):
        nonlocal text
        if flags[name]:
            before = text
            text = fn(text)
            trace.append({"stage": name, "before_len": len(before), "after_len": len(text)})

    _stage("remove_filler_phrases", _remove_filler_phrases)
    _stage("apply_abbreviations",   _apply_abbreviations)
    _stage("apply_contractions",    _apply_contractions)
    _stage("remove_filler_words",   _remove_filler_words)
    if flags["remove_stopwords"]:
        before = text
        text = _remove_stopwords(text, preserve_ne=flags["preserve_named_entities"])
        trace.append({"stage": "remove_stopwords", "before_len": len(before), "after_len": len(text)})
    if flags["remove_function_words"]:
        before = text
        text = _remove_function_words(text, preserve_ne=flags["preserve_named_entities"])
        trace.append({"stage": "remove_function_words", "before_len": len(before), "after_len": len(text)})
    if flags["pos_keep_only"]:
        before = text
        text = _pos_keep_only(text, preserve_ne=flags["preserve_named_entities"])
        trace.append({"stage": "pos_keep_only", "before_len": len(before), "after_len": len(text)})
    _stage("lemmatize",             _lemmatize)
    _stage("shorten_synonyms",      _shorten_synonyms)
    _stage("normalize_whitespace_punct", _normalize_whitespace_punct)

    # Re-assert question form if it was lost during pruning.
    if was_question and not text.endswith("?"):
        text = text.rstrip(".!,;: ") + "?"

    if return_trace:
        return {"compressed": text, "trace": trace, "flags": flags}
    return text


__all__ = ["compress", "TECHNIQUES"]
