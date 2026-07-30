# Bundle identity excludes environment facts

A **bundle key** answers one question: would running again produce different
output? Everything that changes the answer belongs in the **options hash**, and
nothing else does. The vision endpoint's URL was in it, so a Rapid-MLX server
restarted on another port gave every **bundle** a new address and orphaned the
whole cache — for output that would have been byte-identical. The model the
server serves stays in the hash, because that genuinely changes what is
produced; where it happens to be listening does not.

The consequence a reader should expect is that two endpoints serving *different
weights under the same model id* now collide: they produce one **bundle key**
for two different results, and the second run is served the first one's output.
That is the trade we are making. It is narrow — it needs someone to serve
mismatched weights under a name they do not match — and the alternative was
paying a full re-process every time a port moved, which happens by accident and
often. If the collision ever stops being theoretical, the fix is to ask the
server what it is actually serving and hash *that*, not to put the address back.

This generalizes past the endpoint: a **machine-local claim** is never part of
identity. The output directory and the resolved source path already sat outside
the hash and stay outside it. The rule is what the **bundle key** is for, and
`cache_key=False` on an option is the assertion that the option cannot change
output — not that it is unimportant.

## Considered Options

- **Leave the URL in the hash** — rejected: it charges a full re-process for a
  port change, and makes a cache unshareable between two machines that would
  produce identical output.
- **Hash a fingerprint of what the server reports it is serving** — deferred,
  not rejected. It closes the collision properly, but costs a request per run
  and depends on what Rapid-MLX exposes. This is the escalation path if the
  collision becomes real.
- **Keep the URL but normalize it** (strip port, resolve host) — rejected: it
  keeps an environment fact in identity and only narrows how often it misfires,
  while making the rule harder to state.

## Amendment (plan 02, D-012)

Narrowed for exactly one case: *"this run was configured to send keyframes to
a non-loopback vision endpoint"* is a production-provenance fact, not an
environment fact, and folds into bundle identity as the boolean
`local_vision_non_local`. The endpoint's address and credential remain
machine-local claims and stay out of identity, as this ADR requires.
