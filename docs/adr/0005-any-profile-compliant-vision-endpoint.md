# Any profile-compliant vision endpoint, with Rapid-MLX as the default

Supersedes [ADR-0001](0001-rapid-mlx-is-the-only-vision-backend.md).

Distill speaks to whatever OpenAI-compatible vision endpoint the operator
configures. A local Rapid-MLX server remains the **default** — an unconfigured
run is unchanged, still loopback-only, still local-only processing — but it is
no longer the *only* endpoint. `backend` names the wire protocol, not a
provider; there is still no provider abstraction and no runtime shim.

ADR-0001 required an eval before its reversal, and its central claim — that a
frontier cloud reader "is the only thing that would meaningfully beat the local
8-bit model" — turned out to be right. The eval measured it, on a 26-frame
human-verified corpus with six categories including injection-shaped,
safety-blocked, and reader-disagreement frames (`tests/evals/README.md`,
evidence in `tests/evals/gate_2_to_3_cloud.json`):

| | local `Qwen3-VL-8B-Instruct-8bit` | `gemini-3.6-flash` |
|---|---|---|
| text-recovery accuracy | 0.867 | **0.992** |
| invention on true negatives | 0.000 | **0.000** |
| word error rate | 0.151 | **0.062** |

The cloud reader cleared the accuracy floor decisively and invented nothing
across six textless frames, re-verified over three trials because the reader is
non-deterministic. It also read every injection-shaped and safety-blocked frame
correctly, including the schema-targeted injection that defeats the local
reader entirely.

What ADR-0001 got right is the reason this is an opt-in and not a new default:
a remote endpoint breaks **local-only processing**. So the capability is
gated on an explicit `allow_remote_endpoint`, a remote endpoint must speak
`https` (plain `http` reaches loopback only, opt-in or not), the credential
lives in a non-serializable carrier that never enters a log, an error, a
render, or a **bundle**, and a run that may send keyframes off-machine records
a `non_local_only_processing` warning in its render and folds that fact into
**bundle identity**. Local-only stopped being a property of the software and
became a per-run choice the artifact discloses.

The eval discipline ADR-0001 established is unchanged and now applies to this
decision too: the default came from an eval, so it may only be changed by
another eval.

## Considered Options

- **Keep local-only by fiat** — rejected: the eval ADR-0001 itself demanded
  came back showing a large, measured quality gap, and refusing to act on
  evidence we commissioned is not caution.
- **Make a cloud reader the new default** — rejected: it silently sends private
  recordings off-machine. The measured win does not license changing what an
  unconfigured run does.
- **Embed a provider-normalizing library** — rejected, as in ADR-0001. Distill
  defines a compatibility profile and judges an endpoint by attempting a
  completion; non-conforming shapes belong behind an external proxy the
  operator points `base_url` at.
