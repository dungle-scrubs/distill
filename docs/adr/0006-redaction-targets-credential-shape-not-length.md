# Redaction targets credential shape, not length

Distill's redaction rules removed any run of 40 or more alphanumeric characters
under a rule named "generic base64 blob". Length and opacity are not evidence of
a secret, and the rule matched far more than base64: a 40-character commit SHA, a
64-character `sha256` digest, and a Docker image digest were all replaced with
`[REDACTED]`. A **render** exists to be fed to an LLM agent, and a frame showing
a commit SHA exists so the reader can look that commit up. Deleting it protects
nothing and removes the only fact the frame carried.

The rule now exempts a matched run when — and only when — **both** hold: the run
is pure hexadecimal, and an identifier cue sits adjacent to it (`commit`, `sha`,
`sha1`, `sha256`, `sha384`, `sha512`, `digest`, `blob`, `tree`, `integrity`,
`checksum`, `hash`, `rev`, `revision`, or a preceding `@` or `sha256:`).
Everything else is redacted exactly as before.

## Why the exemption requires positive evidence

The obvious design — narrow the rule to things that look like base64, since
that is what it is named for — was tried first and rejected after it was shown
to fail in both directions:

- **It leaks.** The pattern ends `={0,2}\b`, and `\b` cannot occur between `=`
  and end-of-string. The regex backtracks, and the match excludes its own
  padding: `QUFB…QQ==` matches as `QUFB…QQ`. That run is uppercase-and-digits
  only, carries no base64 signal, and a rule keyed on such a signal would have
  **preserved a real secret**. (The same boundary bug is why redaction currently
  leaves `[REDACTED]==` behind — cosmetic today, load-bearing the moment a
  predicate inspects the match.)
- **It does not deliver fidelity anyway.** A 40-character lowercase-hex GitHub
  `device_code` is a documented credential that no name-based rule covers; it is
  caught today only by this rule and would have escaped. Meanwhile Subresource
  Integrity digests and SSH public-key bodies are base64 by definition, entirely
  public, and would have stayed redacted — so "identifier-shaped values survive"
  would have been false for every base64-shaped identifier.

Cue-gating inverts the failure mode, which is the whole reason it was chosen.
The exemption fires only on positive evidence that a value names something
public. A missing cue, a cue OCR mangled, an unusual identifier format — each
degrades to exactly today's behavior. The rule can fail to *improve* fidelity;
it cannot fail *open*.

## What this deliberately does not fix

- **Base64-shaped identifiers stay redacted.** SRI digests, SSH public keys, and
  base64-encoded public data are unchanged from today. Distinguishing them from
  an encoded secret needs context this rule does not have, and inventing a
  syntactic test for it is the mistake documented above.
- **Redaction remains a shape-matching blocklist.** It was defense-in-depth
  before and still is. This decision narrows what it removes; it does not change
  the technique, add entropy detection, or turn a **redaction mark** into a
  guarantee. Anyone pointing Distill at sensitive recordings should test the
  rules against their own secret formats first.

## Consequences

- **A hex credential displayed next to an identifier cue is now kept.** A frame
  reading `digest a1b2…` where `a1b2…` is in fact a secret survives into the
  bundle. This is the cost of the exemption and is accepted: the cue makes the
  value look like a name for something public, which is the same evidence a
  human reader would use.
- **A redaction mark now names a kind** — `[REDACTED:api-key]`,
  `[REDACTED:base64-blob]` — so a reader can tell what stood in a place. The mark
  discloses the kind and nothing further; length was considered and rejected,
  being the disclosure a password can least afford. It carries no whitespace, and
  every rule recognizes an existing mark and leaves it alone, because redaction
  runs twice over the same text by design.
- **The rule consumes its own padding**, so redaction stops leaving stray `=`
  characters in the text.
- **Bundle content changes, so the pipeline version rises.** Every bundle
  produced before this change redacts text that this one keeps, and the
  **pipeline signature** makes shipping the code change without the version bump
  a CI failure rather than a silent cache collision.
- **The test that hid this is replaced.** The suite's only digest fixture was
  `image ghcr.io/org/app:1.2@sha256:abcdef` — a six-character abbreviation that
  never reached the 40-character floor, so no test ever passed a full-length
  digest through `redact_text`.
