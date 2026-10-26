# A forwards-off-machine declaration marks the run non-local

Decided 2026-10-25.

Distill discloses when a run may send **keyframes** off the producing machine:
the `non_local_only_processing` warning and the fold into **bundle identity**
both read one predicate, `config_is_non_local` (D-012). Before this decision
that predicate could be made true only by `allow_remote_endpoint` on a
non-`http` non-loopback endpoint. A loopback `http` endpoint was local by
definition - and that definition had a hole exactly where ADR-0005 points
non-conforming providers: *behind an external proxy the operator points
`base_url` at*.

An operator now runs such a proxy, `http://127.0.0.1:17501/v1`, relaying every
keyframe and its surrounding transcript window to a hosted model. Through it,
Distill would publish a bundle whose identity says local-only processing while
the frames left the machine. No transport rule can close that: the proxy *is*
loopback, `http` *is* loopback-only, and the per-request resolved-address check
(R-43, D-029) proves exactly what is true - that the client reached
127.0.0.1. What the proxy does next is a fact the operator knows and the
address cannot show.

## The decision

A new boolean, `forwards_off_machine` (default false), on the local-vision
config and on each **endpoint chain** entry - everywhere `allow_remote_endpoint`
exists: the config file key, the per-endpoint entries, the CLI pair
`--local-vision-forwards-off-machine` / `--no-local-vision-forwards-off-machine`
with the same override precedence, and `local-vision-diagnostics` output.

`config_is_non_local` returns true when it is set, checked first. The
declaration does not require `allow_remote_endpoint` and changes no transport
rule: R-43's loopback-only-by-default, D-008's scheme rule, the redirect
refusal and the 32 MiB response cap hold exactly as before, because the
endpoint really is loopback. What changes is the disclosure: the warning keeps
its `non_local_only_processing` code and says, for this cause, that the
operator declared the endpoint forwards keyframes and surrounding transcript
text off the machine; and `local_vision_non_local` folds into **bundle
identity**, so a proxy-forwarded reading can never be served under a key that
claims local-only processing. D-012 keeps its shape - the warning and the fold
read the same predicate on the same input.

In a chain, the declaration follows the rule remoteness already follows
(ADR-0007): it is per entry, and the selected entry's declaration is what the
run carries. The declaration is half of an entry's identity the way remoteness
is, so two same-model entries - one declared, one not - derive different
candidate keys rather than colliding.

## Consequences

- The D-016 typo guard reads the same predicate, so a credential that is
  *configured but empty* on a declared endpoint is now refused - fail closed,
  as it already is for a remote one. An operator whose proxy injects its own
  credential configures none, which remains intentional no-auth.
- The declaration arms the vision-stage run-wide budget, as the remote opt-in
  does: a forwarding proxy spends hosted cost per frame - the proxy only
  hides whose - so the wall-clock and byte bounds are the threat model's
  answer here too. No transport rule moves with it.
- Bundle keys change for declared runs only *by this decision's own fold*.
  Carrying it lands a `PIPELINE_VERSION` bump, and that version is part of
  every options hash, so every bundle - declared or not - is re-keyed once,
  when the change lands; after that, an undeclared run derives the same key
  it would have without this decision.

## Considered Options

- **A host allowlist** - reject or flag loopback ports known to be proxies -
  rejected: a proxy's forwarding is a property the operator knows and the
  address does not show. Port 17501 today, a different port or a unix socket
  tomorrow; the allowlist would be a list of guesses that fails open on every
  one it has not heard of.
- **Treat every loopback proxy as unknowable and drop the local-only claim
  entirely** - rejected: it burns the disclosure for the ordinary
  Rapid-MLX-on-loopback run, which really is local-only and says so.
- **Extend `allow_remote_endpoint` to mean "may leave the machine" for loopback
  hosts** - rejected: that flag is R-43's opt-out and owns transport
  consequences (the `https` requirement). A loopback proxy needs none of
  those, and overloading the flag would make its transport meaning
  un-statable. The run-wide budget is no reason to prefer this route either:
  it arms on either mark now, so the declaration carries its bound without
  borrowing the flag.
