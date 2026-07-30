# Any profile-compliant vision endpoint, with Rapid-MLX as the default

Supersedes [ADR-0001](0001-rapid-mlx-is-the-only-vision-backend.md).

Distill speaks to whatever OpenAI-compatible vision endpoint the operator
configures. A local Rapid-MLX server remains the **default** — an unconfigured
run is unchanged, still loopback-only, still local-only processing — but it is
no longer the *only* endpoint. There is still no provider abstraction and no
runtime shim: `options.py` accepts exactly one `backend` value, and this
decision widens where that one client may point rather than how many clients
exist.

ADR-0001 required an eval before its reversal, and its central claim — that a
frontier cloud reader "is the only thing that would meaningfully beat the local
8-bit model" — turned out to be right. The eval measured it, on a 26-frame
human-verified corpus with six categories including injection-shaped,
safety-blocked, and reader-disagreement frames (`tests/evals/README.md`,
evidence in `tests/evals/gate_2_to_3_cloud.json`):

| | local `Qwen3-VL-8B-Instruct-8bit` | `gemini-3.6-flash` |
|---|---|---|
| text-recovery accuracy | clears the floor | **decisively higher** |
| invention on true negatives | none | **none** |
| word error rate | higher | **lower** |

The measured values are deliberately not restated here. They live in
`tests/evals/baseline_local.json` and `tests/evals/gate_2_to_3_cloud.json`,
which are re-recorded together whenever the corpus or the metric changes; a
number copied into this document would drift out of agreement with the
evidence it cites, which is how the first version of this ADR came to claim
something its own baseline contradicted.

The cloud reader cleared the accuracy floor decisively and invented nothing on
the true negatives, re-measured over repeated trials because the reader is
non-deterministic. Per-frame detail lives in the evidence files rather than
being restated here, where it would drift: both readers are scored on every
run, and a claim about one frame is only true of the run that produced it.

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
