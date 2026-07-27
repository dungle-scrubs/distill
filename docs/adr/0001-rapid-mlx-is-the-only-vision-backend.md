# Rapid-MLX is the only vision backend

Distill needs a local vision-language model to interpret keyframes, and several
local runtimes could serve one. We talk to a Rapid-MLX server directly over its
OpenAI-compatible HTTP API and support no other backend: a human-verified
16-frame text-recovery eval found MLX roughly 28% faster than Ollama on Apple
Silicon at equal quality, and every additional backend multiplies the
degradation paths, the prompt-compatibility surface, and the eval matrix we
would have to keep honest.

The deliberate consequence is that there is no provider abstraction and no
runtime shim — no `local_vision_provider` option, no backend branches. A
contributor who adds one is not extending Distill, they are reversing this
decision, and they owe a new eval before doing so.

## Considered Options

- **Ollama** — rejected on measured speed at equal quality.
- **A provider abstraction over both** — rejected because the abstraction's cost
  is paid on every degradation path and prompt change, in exchange for an option
  the eval says we do not want.
- **A frontier cloud reader** — rejected as a default: it is the only thing that
  would meaningfully beat the local 8-bit model, and it breaks the local-only
  property that makes Distill safe to run against private recordings.
