"""Relevant-link extraction for source metadata.

This module owns which URLs in a **source**'s metadata are **related links** -
the domain and context rules that separate a code or reference link from a
social or promotional one - and the label each one carries.

It does not own where related links are written, how they are delimited in a
**render**, or what counts as a secret. A **related link** is **extracted
text** on both halves, label and destination alike: whoever wrote the
description chose both. So `RelatedLink` is a carrier, and R-19's rule holds
here as it does for keyframe text - the **redaction** policy runs at
construction, which is `artifacts`' code and not this module's (R-21).

What leaves this module is therefore the carrier and never its document. A
related link that had already been flattened into a mapping was one no sink
could check: `render` and `response` took whatever four keys they were given,
so the layer that refuses text no policy ran over had nothing to refuse. The
document is produced at the sink, by `serialize`, on the same terms as a
**frame artifact**'s (R-20).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar
from urllib.parse import urlparse, urlunparse

from .artifacts import Carrier, RedactionState
from .redact_secrets import cut_without_splitting_a_mark

URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")
TRAILING_PUNCTUATION = ".,;:!?"
# How long a **related link**'s label may be. A description's author writes the
# line, so the label is as long as they made it; this is what a **render** and a
# **manifest** will carry, and it is applied to redacted text.
LABEL_LIMIT_CHARS = 160

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


@dataclass(frozen=True, kw_only=True)
class RelatedLink(Carrier):
    """One **related link**: where it points, what it was called, and why it was kept.

    `url` and `label` are both extracted-text regions (R-21). The destination is
    the half that gets forgotten, and it is the one that carries a secret most
    plausibly - a query string is where an API key ends up when somebody pastes
    a working request into a description.

    `source` and `reason` are Distill's own words about the link, so they are
    frozen like everything else but never redacted or capped.
    """

    EXTRACTED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("url", "label")

    url: str
    label: str
    source: str
    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        # The label cap runs *after* the base class has redacted, never before.
        # Cutting first is a way of hiding a secret from the policy rather than
        # from a reader: a value straddling the cut is sliced into a prefix that
        # matches no pattern, and that prefix is then durable in full. This is
        # the same ordering `artifacts._redact` states for the 256 KiB cap, for
        # the same reason.
        object.__setattr__(
            self, "label", cut_without_splitting_a_mark(self.label[:LABEL_LIMIT_CHARS])
        )

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, redaction: RedactionState
    ) -> RelatedLink:
        """Rebuild a link from the document a **manifest** recorded.

        The cache-hit route: a **bundle** that is still servable already
        described its links, and a run that reuses it must not re-read the
        source's metadata to say so.

        `redaction` is the *reading run's* policy and is required, never a
        claim the document makes - the same rule, and the same reason, as
        `FrameArtifact.from_document`. `redact_secrets` participates in the
        **options hash** and so in the **bundle key**, so a manifest under this
        bundle key was written by a run under this policy; the policy simply
        runs again over text it has already cleaned, which changes nothing.

        Tolerant on the way in, because a manifest written by an older Distill
        may not carry every field: a missing one takes the empty string rather
        than costing the link.
        """
        return cls(
            url=str(document.get("url", "")),
            label=str(document.get("label", "")),
            source=str(document.get("source", "")),
            reason=str(document.get("reason", "")),
            redaction=redaction,
        )

    def _payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "url": self.url,
                "label": self.label,
                "source": self.source,
                "reason": self.reason,
                "redaction": self.redaction,
                "warnings": self.warnings,
            }
        )


def extract_relevant_links(
    text: str,
    *,
    source: str = "metadata",
    redact: bool = True,
) -> list[RelatedLink]:
    """Extract code/reference links while filtering social and promotional links.

    Carriers out, not documents: a caller holding a document has bypassed R-20
    already, so the sinks would have nothing left to check (finding 5).

    Classification reads the *raw* line and the raw URL, and redaction happens
    when the carrier is built from them: what a link is about is a question
    about the description as written, and a redacted destination would classify
    on a domain the redaction could have altered.

    `redact` is the `--no-redact-secrets` opt-out reaching the same policy that
    covers keyframe text, on the same terms (R-20, R-21).
    """
    policy = RedactionState.NOT_APPLIED if redact else RedactionState.DISABLED
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
                    redaction=policy,
                )
            )
    return links


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
    # Not truncated here: `RelatedLink` caps the label after the redaction
    # policy has run over it (R-19). A cut applied first would decide what the
    # policy gets to see.
    return label if label else normalized_domain(urlparse(url).netloc)
