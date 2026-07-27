from __future__ import annotations

from distill.links import extract_relevant_links


def test_extract_relevant_links_keeps_code_and_reference_links() -> None:
    """What a kept link is, and what its document says about how it was produced.

    R-21 with R-19: a **related link** is **extracted text** on both halves - a
    description's author chooses the label and the destination alike - so it is
    a carrier whose text went through the **redaction** policy at construction,
    and its document records which policy that was.
    """
    links = extract_relevant_links(
        "\n".join(
            [
                "Skill repo: https://github.com/example/catch-me-up",
                "Slides: https://speaker.dev/talks/comprehend-first",
                "Paper: https://arxiv.org/abs/2401.12345.",
            ]
        ),
        source="youtube_description",
    )

    assert links == [
        {
            "url": "https://github.com/example/catch-me-up",
            "label": "Skill repo",
            "source": "youtube_description",
            "reason": "code_or_reference_domain",
            "redaction": "applied",
            "warnings": [],
        },
        {
            "url": "https://speaker.dev/talks/comprehend-first",
            "label": "Slides",
            "source": "youtube_description",
            "reason": "relevant_context",
            "redaction": "applied",
            "warnings": [],
        },
        {
            "url": "https://arxiv.org/abs/2401.12345",
            "label": "Paper",
            "source": "youtube_description",
            "reason": "code_or_reference_domain",
            "redaction": "applied",
            "warnings": [],
        },
    ]


def test_links_are_counted_as_written_and_redacted_afterwards() -> None:
    """Redaction is at construction, which is after this module decides what a link is.

    Two destinations that differ only in the secret they carry are two links.
    Redacting before the duplicate check would collapse them into one and drop
    a reference the description actually made, so the order of the two steps is
    stated here rather than left to be read off the code.
    """
    links = extract_relevant_links(
        "\n".join(
            [
                "Repo: https://github.com/example/repo?api_key=sk-live-0123456789abcdefghij",
                "Docs: https://github.com/example/repo?api_key=sk-test-abcdefghijklmnopqrst",
            ]
        )
    )

    assert [link["label"] for link in links] == ["Repo", "Docs"]
    assert [link["url"] for link in links] == [
        "https://github.com/example/repo?api_key=[REDACTED]",
        "https://github.com/example/repo?api_key=[REDACTED]",
    ]


def test_extract_relevant_links_excludes_social_video_and_promotional_links() -> None:
    links = extract_relevant_links(
        "\n".join(
            [
                "Speaker info: https://www.linkedin.com/in/example",
                "Watch next: https://youtu.be/abc123",
                "Sponsored trial: https://sentry.io/signup",
                "Follow us: https://x.com/example",
                "Sponsor repo: https://github.com/sponsor/coupon",
            ]
        )
    )

    assert links == []
