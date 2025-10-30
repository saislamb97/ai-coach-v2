# socket/text_sanitize.py
from __future__ import annotations

import re
import html
import logging
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal
from functools import lru_cache
from num2words import num2words

logger = logging.getLogger(__name__)

# =============================================================================
# Public config
# =============================================================================

RedactionMode = Literal["remove", "placeholder"]

@dataclass(frozen=True)
class SanitizeOptions:
    # core toggles
    remove_markdown: bool = True
    remove_emojis: bool = True
    remove_social_tags: bool = True
    strip_html_tags: bool = True
    unescape_html_entities: bool = True
    normalize_whitespace: bool = True
    normalize_punctuation: bool = True
    remove_links_emails_phones: bool = True
    convert_numbers: bool = True
    max_collapse_newlines: int = 2

    # redaction behavior
    redact_mode: RedactionMode = "placeholder"   # "placeholder" | "remove"
    link_placeholder: str = "[LINK]"
    email_placeholder: str = "[EMAIL]"
    phone_placeholder: str = "[PHONE]"

    # common idioms (generic English)
    idioms_en: Dict[str, str] = None

    def __post_init__(self):
        if self.idioms_en is None:
            object.__setattr__(self, "idioms_en", {
                "24/7": "twenty four seven",
                "16/9": "sixteen by nine",
                "4/3": "four by three",
            })

# =============================================================================
# Regex
# =============================================================================

# numbers
PCT_RE   = re.compile(r'(?<![\w/])([+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+\-]?\d+(?:\.\d+)?)\s?%(?![\w/])')
ORD_RE   = re.compile(r'(?<![\w/])(\d{1,6})(st|nd|rd|th)\b', re.IGNORECASE)
LIST_RE  = re.compile(r'(?<!\w)(\d+)\.(?=\s|$)')
FRAC_RE  = re.compile(r'(?<![\w/])(\d{1,3}(?:,\d{3})*|\d+)\/(\d{1,3}(?:,\d{3})*|\d+)(?![\w/])')
RANGE_RE = re.compile(
    r'(?<!\w)'
    r'([+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+\-]?\d+(?:\.\d+)?)'
    r'\-'
    r'([+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+\-]?\d+(?:\.\d+)?)'
    r'(?!\w)'
)
LEADING_DEC_RE = re.compile(r'(?<!\d)\.(\d+)\b')
PLAIN_RE       = re.compile(
    r'(?<!\w)([+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+\-]?\d+(?:\.\d+)?)'
    r'(?!\w|\.)'
)
VERSION_TOKEN_RE = re.compile(r'^\d+(?:\.\d+){2,}$')  # e.g., 1.2.3

# simple 24h times (HH:MM)
TIME_RE = re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)\b')

# naive date shield for dd/mm/yyyy or dd/mm/yy to avoid “x/y” fraction reads
DATE_SLASH_RE = re.compile(r'\b([0-3]?\d)/([01]?\d)/(?:\d{2}|\d{4})\b')

# links / contacts
URL_SCHEME_RE   = re.compile(r'\b(?:https?|ftp)://[^\s)]+', re.IGNORECASE)
URL_WWW_RE      = re.compile(r'\bwww\.[^\s)]+', re.IGNORECASE)
NAKED_DOMAIN_RE = re.compile(r'\b[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s)]+)?\b', re.IGNORECASE)
EMAIL_RE        = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
# very generic international-ish phone (e.g., +1 415 555 0100, 0415 555 0100)
PHONE_RE        = re.compile(r'(?<!\w)(?:\+\d{1,3}\s?)?(?:\(?\d{1,4}\)?[\s\-]?)?\d[\d\s\-]{6,}\d(?!\w)')

# markdown
FENCED_CODE_BLOCK_RE = re.compile(r'```.*?```', re.DOTALL)
MD_IMAGE_RE          = re.compile(r'!\[.*?\]\(.*?\)')
MD_LINK_RE           = re.compile(r'\[([^\]]+)\]\([^)]+\)')
MD_BOLD_ITALIC_RE    = re.compile(r'(\*\*|__)(.*?)\1', re.DOTALL)
MD_ITALIC_RE         = re.compile(r'(\*|_)(.*?)\1', re.DOTALL)
MD_INLINE_CODE_RE    = re.compile(r'`([^`]+)`')
MD_HEADING_RE        = re.compile(r'^\s{0,3}#+\s*(.*)', re.MULTILINE)
MD_LIST_RE           = re.compile(r'^\s*([-*+]|(\d+\.))\s+', re.MULTILINE)
MD_BLOCKQUOTE_RE     = re.compile(r'^\s*>+\s?', re.MULTILINE)
MD_TABLE_BORDER_RE   = re.compile(r'^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$', re.MULTILINE)

# social
AT_HANDLE_RE         = re.compile(r'(?<!\w)@[A-Za-z0-9_]{2,}\b')
HASHTAG_RE           = re.compile(r'(?<!\w)#[\w-]{2,}\b')

# emoji & control chars
EMOJI_RE             = re.compile(
    r'['
    r'\U0001F300-\U0001FAFF'
    r'\U00002700-\U000027BF'
    r'\U00002600-\U000026FF'
    r'\U0001F1E6-\U0001F1FF'
    r']+', flags=re.UNICODE
)

# punctuation / whitespace
DASHES_RE            = re.compile(r'[–—―]+')
ELLIPSIS_RE          = re.compile(r'…')
SMART_QUOTES_RE      = re.compile(r'[“”«»„‟]+|[‘’‹›‚‛]+')
MULTI_PUNCT_RE       = re.compile(r'([!?.]){2,}')
SPACES_AROUND_PUNCT  = re.compile(r'\s+([,;:])')

# years (1000–2099)
YEAR_RE              = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')


# =============================================================================
# Number helpers (generic English)
# =============================================================================

def _num_lang() -> str:
    return "en"

@lru_cache(maxsize=4096)
def _num_to_words_cached(raw: str) -> str:
    s = raw.replace(",", "")
    if s.startswith("."):
        s = "0" + s
    try:
        n = float(s) if "." in s else int(s)
        return num2words(n, lang=_num_lang())
    except Exception:
        return raw

def _num_to_words(raw: str) -> str:
    return _num_to_words_cached(raw)

def _ordinal_to_words(n_str: str) -> str:
    try:
        return num2words(int(n_str), to="ordinal", lang=_num_lang())
    except Exception:
        return _num_to_words(n_str)

def _year_to_words(y_str: str) -> str:
    try:
        return num2words(int(y_str), to="year", lang=_num_lang())
    except Exception:
        return _num_to_words(y_str)

def _looks_like_version(token_env: str) -> bool:
    return bool(VERSION_TOKEN_RE.match(token_env))


# =============================================================================
# Step 1: numbers → words (generic)
# =============================================================================

def _convert_all_numbers(text: str, opts: SanitizeOptions) -> str:
    percent_word, to_word, frac_word = "percent", "to", "over"
    idioms = opts.idioms_en or {}

    # Percentages
    text = PCT_RE.sub(lambda m: f"{_num_to_words(m.group(1))} {percent_word}", text)

    # Ordinals
    text = ORD_RE.sub(lambda m: _ordinal_to_words(m.group(1)), text)

    # Shield dd/mm/yyyy so we don’t read it as a fraction
    SHIELD = "\uFFFF"
    shields = {}
    def _shield_date(m: re.Match) -> str:
        k = f"{SHIELD}{len(shields)}{SHIELD}"
        shields[k] = m.group(0)
        return k
    text = DATE_SLASH_RE.sub(_shield_date, text)

    # Times HH:MM → “fourteen thirty / oh five”
    def _time_repl(m: re.Match) -> str:
        hh, mm = m.group(1), m.group(2)
        try:
            h = int(hh); m_ = int(mm)
            h_words = _num_to_words(str(h))
            mm_words = "oh " + _num_to_words(mm) if 0 < m_ < 10 else (_num_to_words(mm) if m_ else "hundred")
            spoken = f"{h_words} {mm_words}".strip()
            return re.sub(r"\s+", " ", spoken)
        except Exception:
            return m.group(0)
    text = TIME_RE.sub(_time_repl, text)

    # Slash numbers: idioms / fractions (but not versions, dates already shielded)
    def _frac_repl(m: re.Match) -> str:
        raw = m.group(0)
        if raw in idioms:
            return idioms[raw]
        a, b = m.group(1), m.group(2)
        return f"{_num_to_words(a)} {frac_word} {_num_to_words(b)}"
    text = FRAC_RE.sub(_frac_repl, text)

    # Ranges
    text = RANGE_RE.sub(lambda m: f"{_num_to_words(m.group(1))} {to_word} {_num_to_words(m.group(2))}", text)

    # Leading decimals
    text = LEADING_DEC_RE.sub(lambda m: _num_to_words("." + m.group(1)), text)

    # Standalone years
    text = YEAR_RE.sub(lambda m: _year_to_words(m.group(1)), text)

    # Plain numbers (skip obvious version environments like “v1.2.3”)
    def _plain_repl(m: re.Match) -> str:
        raw = m.group(1)
        span_start, span_end = m.span(1)
        left = text[max(0, span_start-5):span_start]
        right = text[span_end:span_end+6]
        env = (left + raw + right).strip()
        if _looks_like_version(env):
            return raw
        return _num_to_words(raw)
    text = PLAIN_RE.sub(_plain_repl, text)

    # Unshield dates
    for k, v in shields.items():
        text = text.replace(k, v)

    return text


# =============================================================================
# Step 2: links/emails/phones — remove or placeholder
# =============================================================================

def _maybe_replace(placeholder: str, mode: RedactionMode) -> str:
    return placeholder if mode == "placeholder" else ""

def _remove_links_emails_phones(text: str, opts: SanitizeOptions) -> str:
    for pat, ph in ((URL_SCHEME_RE, opts.link_placeholder), (URL_WWW_RE, opts.link_placeholder)):
        text = pat.sub(lambda _: _maybe_replace(ph, opts.redact_mode), text)

    # Naked domains (skip trivial abbreviations)
    def _naked(m: re.Match) -> str:
        s = m.group(0)
        if re.match(r'^[A-Za-z]{1,3}\.$', s):
            return s
        return _maybe_replace(opts.link_placeholder, opts.redact_mode)
    text = NAKED_DOMAIN_RE.sub(_naked, text)

    text = EMAIL_RE.sub(lambda _: _maybe_replace(opts.email_placeholder, opts.redact_mode), text)
    text = PHONE_RE.sub(lambda _: _maybe_replace(opts.phone_placeholder, opts.redact_mode), text)
    return text


# =============================================================================
# Step 3: Markdown
# =============================================================================

def _remove_markdown(text: str) -> str:
    text = FENCED_CODE_BLOCK_RE.sub("", text)
    text = MD_IMAGE_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = MD_BOLD_ITALIC_RE.sub(r"\2", text)
    text = MD_ITALIC_RE.sub(r"\2", text)
    text = MD_INLINE_CODE_RE.sub(r"\1", text)
    text = MD_HEADING_RE.sub(r"\1", text)
    text = MD_LIST_RE.sub("", text)
    text = MD_BLOCKQUOTE_RE.sub("", text)
    text = MD_TABLE_BORDER_RE.sub("", text)
    return text


# =============================================================================
# Step 4: Social tags
# =============================================================================

def _remove_social_tags(text: str) -> str:
    text = AT_HANDLE_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    return text


# =============================================================================
# Step 5: HTML
# =============================================================================

def _strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)

def _unescape_html_entities(text: str) -> str:
    return html.unescape(text)


# =============================================================================
# Step 6: Emoji & control chars
# =============================================================================

def _remove_emojis_and_controls(text: str) -> str:
    text = EMOJI_RE.sub("", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text


# =============================================================================
# Step 7: Punctuation / whitespace
# =============================================================================

def _normalize_punctuation(text: str) -> str:
    text = DASHES_RE.sub("-", text)
    text = ELLIPSIS_RE.sub("...", text)
    text = SMART_QUOTES_RE.sub(lambda m: '"' if '“' in m.group(0) or '”' in m.group(0) else "'", text)
    text = MULTI_PUNCT_RE.sub(lambda m: m.group(0)[0], text)
    text = SPACES_AROUND_PUNCT.sub(r"\1", text)
    return text

def _normalize_whitespace(text: str, max_newlines: int = 2) -> str:
    text = re.sub(r'[ \t]{2,}', " ", text)
    max_newlines = max(1, int(max_newlines))
    text = re.sub(r"\n{" + str(max_newlines + 1) + r",}", "\n" * max_newlines, text)
    return text.strip()


# =============================================================================
# Public
# =============================================================================

def sanitize_and_verbalize_text(
    text: str,
    lang: str = "en",                 # generalized English
    options: SanitizeOptions = SanitizeOptions(),
) -> str:
    """
    General English sanitization & verbalization for TTS:
      1) Numbers → words (%, ordinals, times, fractions, ranges, decimals, years, plain numbers).
      2) Redact links/emails/phones (remove or placeholders).
      3) Strip Markdown.
      4) Remove social tags.
      5) Unescape HTML entities, strip tags.
      6) Remove emojis & control chars.
      7) Normalize punctuation.
      8) Normalize whitespace.
    """
    if not text or not text.strip():
        return text

    if options.unescape_html_entities:
        text = _unescape_html_entities(text)
    if options.remove_markdown:
        text = _remove_markdown(text)
    if options.strip_html_tags:
        text = _strip_html_tags(text)
    if options.convert_numbers:
        text = _convert_all_numbers(text, options)
    if options.remove_links_emails_phones:
        text = _remove_links_emails_phones(text, options)
    if options.remove_social_tags:
        text = _remove_social_tags(text)
    if options.remove_emojis:
        text = _remove_emojis_and_controls(text)
    if options.normalize_punctuation:
        text = _normalize_punctuation(text)
    if options.normalize_whitespace:
        text = _normalize_whitespace(text, options.max_collapse_newlines)

    return text
