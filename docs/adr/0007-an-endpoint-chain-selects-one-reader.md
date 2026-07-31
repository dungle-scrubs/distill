# An endpoint chain selects one reader, and the selection is what identity records

ADR-0005 widened where Distill's one vision client may point. This extends that:
an operator may configure several **vision endpoints** in preference order, and
a run uses exactly one of them.

What the chain does *not* change is the shape of the client. There is still one
OpenAI-compatible client, no provider abstraction and no runtime shim — a chain
is a list of addresses that client may be pointed at, resolved once per run,
not a set of readers a run coordinates between. No **bundle** ever holds
**interpretations** from two of them.

## What enters identity

The **selected endpoint**'s model, and whether that endpoint was remote. Not the
chain. Three chains that select the same endpoint produce byte-identical
bundles, which is what keeps adding a fallback nobody used from re-keying a
whole cache to arrive at the same output.

The one thing the chain adds to identity is **vision mode**: `selected`,
`disabled`, or `chain_exhausted`. A boolean could not carry it. A run whose
endpoints were all unreachable and a run told `--no-caption-frames` both publish
a bundle with no readings, and they are not the same bundle — one would look
different on a better day and the other is exactly what was asked for. Sharing a
key would mean the first is served the second's bundle forever after.

## Cache before network, always

Every candidate key is scanned against the cache before any endpoint is asked.
A hit *below* the top of the chain is still a hit: preference order says who to
ask when work must be done, not that a bundle produced by a less-preferred
reader should be rebuilt by a better one. An offline machine serves what it
already has.

This is the property most easily lost by accident. It was, once, during
implementation — the cache check read the frame-document key while a
**manifest** embeds the response shape, so every generation counted as holding
no readings and a cache hit with vision enabled probed anyway. Nothing failed;
it was slower and reached the network when it had promised not to. The
end-to-end test that catches it is a permanent fixture for that reason.

## Degradation, not fatality

An endpoint that cannot be reached is **degradation** (ADR-0002). A chain exists
so that one wrong entry does not stop a run, so a probe that raises is
unavailability rather than an exception — a single rejected endpoint must not
take the run with it and leave the later entries unasked.

Four outcomes are recorded per entry, and none collapses into another because
each answers a different question an operator would ask: **skipped** (the memo
already knew), **unavailable** (asked, could not serve), **deadline_spent** (the
run gave up before its turn), **cache_vanished** (a hit was pruned before it
could be claimed).

## Consequences

- **Selection evidence stays out of bundles.** Which endpoints were asked or
  skipped is a fact about one machine on one run — the same category as a
  **machine-local claim**. In a bundle it would mean two machines writing
  different bundles for the same reading.
- **The negative-availability memo is a bound, not a verdict.** Endpoints come
  back, so it expires; and a clock that ran backwards expires it rather than
  trusting it, because "recorded in the future, so still fresh" would skip an
  endpoint until the clock caught up.
- **One deadline for the whole walk.** How many endpoints a chain names is the
  operator's choice; how long a run waits should not be that choice multiplied
  by a timeout.
- **Availability is not revalidated after waiting for the bundle lock.** The
  cache *is* re-asked under the lock, because a stale cache answer makes a run
  redo work already done. A stale availability answer is different in kind: it
  yields a worse reading rather than a wrong bundle, and the memo's TTL bounds
  how long it can persist. Recorded as owed rather than done.
- **Reader-identity collisions across time are not addressed.** Two endpoints
  serving different weights under one model id still collide on a **bundle
  key**, exactly as ADR-0004 says. A chain makes that easier to arrange by
  accident, and does not make it new.
