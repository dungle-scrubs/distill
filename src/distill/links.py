"""Relevant-link extraction for source metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")
TRAILING_PUNCTUATION = ".,;:!?"

SOCIAL_DOMAINS = {
    "bsky.app",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "lnkd.in",
    "mastodon.social",
    "threads.net",
    "tiktok.com",
    "twitter.com",
    "x.com",
}
VIDEO_DOMAINS = {
    "youtu.be",
    "youtube.com",
}
PROMOTIONAL_DOMAINS = {
    "buymeacoffee.com",
    "eventbrite.com",
    "ko-fi.com",
    "linktr.ee",
    "patreon.com",
    "sentry.io",
    "shopify.com",
}
CODE_OR_REFERENCE_DOMAINS = {
    "arxiv.org",
    "bitbucket.org",
    "codeberg.org",
    "docs.github.com",
    "gist.github.com",
    "github.com",
    "gitlab.com",
    "huggingface.co",
    "npmjs.com",
    "paperswithcode.com",
    "pypi.org",
    "readthedocs.io",
    "sourcehut.org",
}
RELEVANT_CONTEXT_KEYWORDS = {
    "article",
    "blog",
    "code",
    "deck",
    "demo",
    "doc",
    "docs",
    "documentation",
    "example",
    "github",
    "library",
    "paper",
    "package",
    "project",
    "repo",
    "repository",
    "resources",
    "skill",
    "slides",
    "source",
}
PROMOTIONAL_CONTEXT_KEYWORDS = {
    "booth",
    "coupon",
    "discount",
    "follow",
    "free trial",
    "newsletter",
    "promo",
    "sponsor",
    "sponsored",
    "subscribe",
    "tickets",
    "trial",
}


@dataclass(frozen=True)
class RelatedLink:
    url: str
    label: str
    source: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "url": self.url,
            "label": self.label,
            "source": self.source,
            "reason": self.reason,
        }


def extract_relevant_links(text: str, *, source: str = "metadata") -> list[dict[str, str]]:
    """Extract code/reference links while filtering social and promotional links."""
    links: list[RelatedLink] = []
    seen: set[str] = set()
    for line in text.splitlines():
        for match in URL_RE.finditer(line):
            url = normalize_url(match.group(0))
            if url in seen:
                continue
            classification = classify_link(url, line)
            if classification is None:
                continue
            seen.add(url)
            links.append(
                RelatedLink(
                    url=url,
                    label=link_label(line, url),
                    source=source,
                    reason=classification,
                )
            )
    return [link.to_dict() for link in links]


def classify_link(url: str, context: str) -> str | None:
    parsed = urlparse(url)
    domain = normalized_domain(parsed.netloc)
    context_lower = context.lower()
    if domain_matches(domain, SOCIAL_DOMAINS):
        return None
    if domain_matches(domain, VIDEO_DOMAINS):
        return None
    if has_keyword(context_lower, PROMOTIONAL_CONTEXT_KEYWORDS):
        return None
    if domain_matches(domain, PROMOTIONAL_DOMAINS):
        return None
    if domain_matches(domain, CODE_OR_REFERENCE_DOMAINS):
        return "code_or_reference_domain"
    if has_keyword(context_lower, RELEVANT_CONTEXT_KEYWORDS):
        return "relevant_context"
    return None


def normalize_url(url: str) -> str:
    stripped = url.rstrip(TRAILING_PUNCTUATION)
    parsed = urlparse(stripped)
    return urlunparse(parsed._replace(fragment=""))


def normalized_domain(netloc: str) -> str:
    domain = netloc.lower().split("@")[-1].split(":")[0]
    return domain[4:] if domain.startswith("www.") else domain


def domain_matches(domain: str, candidates: set[str]) -> bool:
    return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in candidates)


def has_keyword(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def link_label(line: str, url: str) -> str:
    raw_url = next(
        (match.group(0) for match in URL_RE.finditer(line) if normalize_url(match.group(0)) == url),
        url,
    )
    label = line.replace(raw_url, "").strip(f" -:\t{TRAILING_PUNCTUATION}")
    return label[:160] if label else normalized_domain(urlparse(url).netloc)
