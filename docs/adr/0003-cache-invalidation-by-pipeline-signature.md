# Cache invalidation by pipeline signature over source code

Distill caches expensive output (transcription, frame extraction, vision
interpretation) under a **bundle key**, so it must know when its own code has
changed enough that cached output is no longer what the current code would
produce. We hash the *source text of the output-affecting modules themselves*
into a **pipeline signature**, and require that any change to them also raises
the **pipeline version** — which participates in the bundle key, so raising it
gives every bundle a new address and old output simply stops being found.

This is unusual enough to surprise a reader, so: the alternatives were hashing
the *outputs* (which requires producing them, defeating the cache) or trusting
contributors to bump a version by hand (which fails silently and exactly when it
matters). Hashing the inputs to the computation is the only option that fails
loudly at commit time.

## Consequences

- Signed-ness is a property, not a list. If editing a module can change bundle
  content, it is a signed module; an output-affecting module missing from the
  signature is a defect, because the mechanism's whole value is that it cannot
  be quietly incomplete.
- Any output-affecting change invalidates every existing cached bundle. This is
  correct and intended, and it means such changes are not free for users.
- The signature detects *unversioned* change. It does not verify that output is
  actually identical, and it is not a security control.
