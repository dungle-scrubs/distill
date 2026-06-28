from __future__ import annotations

from distill.links import extract_relevant_links


def test_extract_relevant_links_keeps_code_and_reference_links() -> None:
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
        },
        {
            "url": "https://speaker.dev/talks/comprehend-first",
            "label": "Slides",
            "source": "youtube_description",
            "reason": "relevant_context",
        },
        {
            "url": "https://arxiv.org/abs/2401.12345",
            "label": "Paper",
            "source": "youtube_description",
            "reason": "code_or_reference_domain",
        },
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
