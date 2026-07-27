# Distill

Distill turns a recorded video into a durable, re-readable account of what was
said and shown in it, so that an LLM agent can consume the recording without
watching it. Everything here is vocabulary; no implementation detail belongs in
this file.

## Language

### Sources and identity

**Source**:
The single piece of media a run processes — one local video file, or one
YouTube video.
_Avoid_: input, video (ambiguous with the media file itself)

**Source fingerprint**:
The identity of a source's content, independent of how it will be processed.
_Avoid_: file hash, checksum

**Options hash**:
The identity of the processing choices that can change what a run produces.
_Avoid_: config hash, settings hash

**Bundle key**:
The combined identity of a **source fingerprint** and an **options hash**; it
names exactly one **bundle**.
_Avoid_: source hash (it identifies the bundle, not the source)

**Lock key**:
The identity used to stop two runs from acquiring the same remote **source** at
the same time.

### Bundle lifecycle

**Bundle**:
Everything Distill maintains for one **bundle key** — a **manifest** and every
**generation** produced under that key.
_Avoid_: output folder, cache entry

**Generation**:
One immutable, finished rendering of a **bundle**.
_Avoid_: version, revision, run output

**Active generation**:
The one **generation** a **manifest** names as authoritative; the only one a
reader is entitled to.
_Avoid_: latest generation (the newest generation is not necessarily active)

**Manifest**:
The bundle-level record that names the **active generation** and describes what
was produced.

**Staging directory**:
The mutable place a run assembles a **generation** before publishing it; never
authoritative and never served to a reader.
_Avoid_: temp directory, working directory

**Publish**:
The single indivisible transition that turns a **staging directory** into a
**generation** and makes it the **active generation**.
_Avoid_: commit, finalize

**Stage result**:
The recorded output of one completed pipeline stage, kept only so an interrupted
run can **resume** without redoing it.
_Avoid_: partial (the stage is complete; it is the *run* that is partial)

**Resume**:
Continuing an interrupted run by reusing the **stage results** it already
produced.

**Prune**:
Deliberately removing **generations** or whole **bundles** to reclaim space.
_Avoid_: cleanup, garbage collection

### Extraction

**Keyframe**:
A frame chosen from the **source** as visually distinct enough to be worth
interpreting.
_Avoid_: frame, screenshot, thumbnail

**Frame artifact**:
A **keyframe** together with everything Distill derived from it — its image,
its position in the source, its **extracted text**, its **interpretation**, its
**grounding**, and whether it has been **redacted**.
_Avoid_: frame dict, frame record

**Transcript**:
The timed, segmented speech recovered from a **source**'s audio.

**Interpretation**:
The vision model's structured reading of one **keyframe** — what is shown, what
it means, and what text it could read.
_Avoid_: caption, description

**Verbatim text**:
On-screen text the vision model reports it could actually read, as opposed to
what it inferred.

**Render**:
The markdown account of a **generation**, written to be read by a person or an
LLM agent.

**Related link**:
A code or reference URL recovered from a **source**'s metadata and judged
relevant rather than promotional.

### Trust and redaction

**Extracted text**:
Any text recovered from a **source** or its metadata — **transcript** text,
**extracted text** from images, **interpretations**, **related link** labels —
as opposed to Distill's own words.
_Avoid_: content, text (too general to carry the trust distinction)

**Untrusted-data boundary**:
The marked separation between Distill's own words and **extracted text**,
maintained wherever the two appear together.

Whoever produced a **source** chooses its **extracted text**, and a **render**
is written to be fed to an LLM agent. **Extracted text** is therefore
attacker-controlled input to a downstream model, and the boundary is what keeps
it data rather than instruction.

**Redaction**:
Replacing secret-shaped values in **extracted text** before that text becomes
durable.

**Redaction sink**:
Any point at which **extracted text** becomes durable — written to disk, or
placed into a **render**. Redaction is complete only when every sink is covered.

**Grounding**:
The assessment of whether an **interpretation** is supported by text that was
genuinely read rather than inferred.

**Corroborated**:
Two readers — the image-text reader and the vision model — independently
recovered the same text from a **keyframe**.

**Ungrounded**:
An **interpretation** exists although no reader recovered readable text that
could support it.

### Reproducibility

**Signed module**:
A unit of source code whose content can change what a **bundle** contains.
Whether a module is signed is a question of fact, not preference: if editing it
can change bundle content, it is signed.

**Pipeline signature**:
A hash over every **signed module**, used to detect that output-affecting code
changed without the **pipeline version** being raised.

**Pipeline version**:
The number identifying an output-affecting revision of the pipeline. It
participates in the **options hash**, so raising it gives every **bundle** a new
**bundle key** — this is how stale output stops being served.

### Degradation

**Optional capability**:
A capability whose absence reduces a **bundle** but still leaves it useful.

**Required capability**:
A capability whose absence means no usable **bundle** can be produced.

**Degradation**:
Continuing a run with reduced output and a recorded **warning** when an
**optional capability** is unavailable, instead of failing.
_Avoid_: fallback, graceful failure

**Warning**:
A structured, non-fatal record of something that reduced or qualified a
**bundle**, carried with it.

**Fatal error**:
A coded, staged failure that ends a run and produces no **bundle**.

### Measurement

**Eval**:
The human-verified scoring run that selects Distill's defaults; a default that
came from an eval may only be changed by another eval.

## Relationships

- A **bundle key** names exactly one **bundle**
- A **bundle** holds one **manifest** and zero or more **generations**
- A **manifest** names exactly one **generation** as the **active generation**
- A run assembles one **staging directory**, which **publish** turns into a
  **generation**
- A **staging directory** holds **stage results**; a **generation** does not
- A **generation** holds one **transcript**, many **frame artifacts**, and one
  **render**
- A **frame artifact** has one **keyframe**, at most one **interpretation**, and
  one **grounding**
- **Extracted text** flows from a **source** into **frame artifacts** and the
  **transcript**, and reaches a reader only through a **redaction sink**
- Every **signed module** contributes to the **pipeline signature**
- The **pipeline version** contributes to the **options hash**, and so to every
  **bundle key**

```
bundle key ──names──> Bundle
                        ├── Manifest ──names──> Active generation
                        ├── Generation g1        (immutable, publishable output)
                        ├── Generation g2
                        └── Staging directory    (mutable, private to one run)
                              └── Stage results  (resume scratch — never published)

              publish
Staging directory ─────────> Generation   (indivisible: the generation becomes
                                           readable and active together, or
                                           neither happens)
```

## Example dialogue

> **Dev:** "The run crashed halfway. Can I still read the bundle?"
> **Domain expert:** "Yes — the **bundle** still has whatever **generation**
> was active before. A crashed run only leaves a **staging directory**, and
> nothing reads that."
>
> **Dev:** "So the OCR output the crashed run already wrote is in the bundle?"
> **Domain expert:** "No. That's a **stage result** — it exists so the run can
> **resume**. It is not part of a **generation**, and it must never be
> published, because it hasn't been through a **redaction sink** yet."
>
> **Dev:** "The vision model read a slide the image-text reader couldn't. Is
> that **corroborated**?"
> **Domain expert:** "No — only one reader recovered anything. It might still
> be right, but 'corroborated' means two readers independently agreed. Calling
> a single confident reader corroborated is exactly the mistake that lets a
> hallucinated slide look verified."
>
> **Dev:** "The transcript has a customer's API key in it. The render shows
> `[REDACTED]`, so we're fine?"
> **Domain expert:** "Only if every **redaction sink** is covered. The render
> is one sink. If that text is also written to disk anywhere else, that's
> another sink, and the secret is still in the bundle."

## Flagged ambiguities

- **"bundle"** was used both for everything cached under one key and for the
  files a single run writes — resolved: a **bundle** is the manifest plus all
  generations; the files one run writes are a **generation**'s contents.
- **"source hash"** names the combination of source *and* options, so it
  identifies a bundle rather than a source — resolved: **bundle key**.
- **Source fingerprint** and **lock key** are the same value for a YouTube
  source, which invites treating them as one concept — resolved: they stay
  distinct, because one answers "is this the same media?" and the other answers
  "is another run already fetching this?"
- **"partial"** was used for a completed stage's recorded output — resolved:
  **stage result**, and it is resume scratch that a **generation** must not
  contain.
- **"grounded"** covered two structurally different situations: two readers
  agreeing, and one reader vouching for itself — resolved: only the first is
  **corroborated**; the vocabulary must not let self-report borrow the
  authority of agreement.
- **"frame"** was used for a raw video frame, a selected frame, its image file,
  and the record derived from it — resolved: **keyframe** for the selection,
  **frame artifact** for the record.
- **Graceful degradation** was stated as a promise for any missing dependency,
  but only holds for some — resolved: the promise applies to **optional
  capabilities**; a missing **required capability** is a **fatal error**, and
  which is which must be stated rather than implied.
- **"signed module"** was treated as a maintained list rather than a property —
  resolved: signed-ness is definitional (editing it can change bundle content),
  so an unsigned output-affecting module is a defect, not an omission.
