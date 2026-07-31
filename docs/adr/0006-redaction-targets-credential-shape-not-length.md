# Redaction targets credential shape, not length

Distill's redaction rules removed any run of 40 or more alphanumeric characters
under a rule named "generic base64 blob". Length and opacity are not evidence of
a secret, and the rule matched far more than base64: a 40-character commit SHA, a
64-character `sha256` digest, and a Docker image digest were all replaced with
`[REDACTED]`. The rule now requires an actual base64 signal — padding, a `+` or
`/`, or mixed case — so an **identifier-shaped value** survives into the bundle
and only a **credential-shaped value** is removed.

This is a deliberate reduction in coverage, and it is the point rather than a
side effect. A bare, contextless run of 64 hex characters is genuinely ambiguous:
it is a content digest and it is a plausible API secret, and no amount of
shape-reading settles which. Distill now resolves that ambiguity toward the
reader. The same secret written the way a secret is normally written on screen —
`API_SECRET=<hex>`, `Authorization: Bearer <hex>`, inside a URL authority — is
still caught, by the assignment and userinfo rules that read the *name* beside
the value rather than the value alone. What is given up is the incidental catch
of a hex secret sitting bare on a slide with nothing around it to identify it.

What is bought is the thing Distill exists to do. A **render** is written to be
fed to an LLM agent, and a frame showing `commit f9a1a14a...` exists precisely so
a reader can look that commit up. Redacting it protects nothing — the commit is
in the repository either way — and removes the one fact the frame was carrying.
Silently deleting every hash, digest, and build id on screen is a fidelity defect
in a tool whose product is legibility, and it was invisible because the only
digest fixture in the suite (`tests/test_redaction_patterns.py`) used a truncated
`sha256:abcdef` that never reached the 40-character floor.

## Considered options

- **Exempt known identifier lengths** (pure hex of exactly 40 or 64 characters).
  Rejected as arbitrary in both directions: it exempts a 64-character hex secret
  just as readily, and still eats a 48-character hex identifier.
- **Require a nearby cue word** (`commit`, `sha256`, `digest`). Rejected: the cue
  list is a blocklist with the same open-ended failure as the one it replaces,
  and OCR routinely drops or mangles the very word the rule would depend on.
- **Keep redacting and name the shape in the mark** (`[REDACTED sha256-hex]`).
  Rejected as the primary fix — it tells the reader what they lost without giving
  it back — but adopted as a general improvement: see below.

## Consequences

- **Redaction is not a promise, and this makes that sharper.** It was already a
  shape-matching blocklist, defense-in-depth rather than a guarantee. A hex
  secret displayed bare, with no assignment and no header, is now outside its
  reach. Anyone pointing Distill at genuinely sensitive recordings should test
  the rules against their own secret formats first.
- **A `[REDACTED]` mark now names a kind** — `[REDACTED api-key]`,
  `[REDACTED base64-blob]` — so a reader can tell what stood in a place. The mark
  discloses the kind and nothing further; length was considered and rejected,
  being the disclosure a password can least afford.
- **Bundle content changes, so the pipeline version rises.** Every bundle produced
  before this change redacts text that this one keeps, and the **pipeline
  signature** makes shipping the code change without the version bump a CI
  failure rather than a silent cache collision.
