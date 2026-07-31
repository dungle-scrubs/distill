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

**Artifact**:
The one self-contained document a caller consumes, copied out of the **active
generation** into the project being worked in. It is the deliverable, not
derived state: a **bundle** may be reclaimed by a cache cleaner at any time,
and an artifact may not.
_Avoid_: output, final render, frame artifact (a **frame artifact** is the
record derived from one **keyframe** and lives inside a **generation**; an
artifact is the whole reading and lives outside the cache)

**Artifact directory**:
Where **artifacts** are written — normally `.distill/` at the root of the work
tree the run was invoked from. Distinct from the **output root**, which is
where **bundles** live, and resolved separately: one answers "where does the
deliverable go", the other "where does derived state live".
_Avoid_: output directory, output dir (that is the cache root)

**Active generation**:
The one **generation** a **manifest** names as authoritative; the only one a
reader is entitled to.
_Avoid_: latest generation (the newest generation is not necessarily active)

**Manifest**:
The bundle-level record that names the **active generation** and describes what
was produced.

**Machine-local claim**:
A recorded value that is true only where it was produced — a filesystem path, a
server address, a flag naming how one invocation was told to behave. It is not a
fact about what a **generation** contains, and a record carrying one stops being
true the moment it is read anywhere else.
_Avoid_: absolute path (that is one instance of the problem, not the class)

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

**Vision endpoint**:
An OpenAI-compatible service a run may send **keyframes** to for
**interpretation**, named by its address, the model it serves, and a
credential. Whether it runs on the machine producing the **bundle** or in the
cloud, and who provides it, is a configuration choice Distill does not otherwise
track. The model it serves is identity-affecting and enters the **options
hash**; the address and credential are **machine-local claims** and never enter
a **bundle**.
_Avoid_: provider, backend (Distill knows the endpoint, not who runs it)

**Endpoint chain**:
The **vision endpoints** an operator configured, in the order they should be
tried. Order is preference, not precedence over correctness: every endpoint in
a chain is one the operator is willing to have read their **keyframes**, and a
run uses exactly one of them.
_Avoid_: fallback chain (**degradation** already owns that word, and moving
from one endpoint to another is not degradation - the reading is not reduced,
a different reader produced it), provider list, tiers

**Available**:
Said of a **vision endpoint** Distill reached and got a usable completion from.
Availability is a fact about this run and this machine, established by asking:
an endpoint can be configured and unavailable - unreachable, unauthenticated,
or serving a different model - and that is the ordinary case an **endpoint
chain** exists to absorb.
_Avoid_: healthy, up, enabled (enabled is configuration; available is the
answer to having asked)

**Skipped**:
Said of a **vision endpoint** an **endpoint chain** passed over without asking,
because it was asked recently enough and found unavailable. Distinct from
unavailable, which is what an endpoint asked *this run* turned out to be, and
the distinction is user-visible: a skipped entry costs no round trip and means
"we already know", an unavailable one costs a round trip and means "we just
checked". An operator reading diagnostics needs to tell those apart.

**Selected endpoint**:
The first **available** **vision endpoint** in the **endpoint chain**: the one
that produced a run's **interpretations**, and therefore the one whose model
enters the **options hash**. A run has at most one - selection happens once,
so no **bundle** ever holds **interpretations** from two readers.
_Avoid_: active endpoint, winner, primary (primary reader already names the
vision model's role against the image-text reader in **grounding**)

**Frame salience**:
Whether a **keyframe** adds information the **transcript** does not already
convey, judged against the surrounding transcript and recorded on the **frame
artifact**. It lets a reader decide what to skip; Distill never uses it to drop
a **keyframe**. A **source** with no **transcript** has none.
_Avoid_: relevance (it is relative to the transcript, not absolute), importance

**Render**:
The markdown account of a **generation**, written to be read by a person or an
LLM agent.

**Self-contained render**:
A **render** that refers to no file outside itself, so it stays complete when it
is separated from its **bundle** and read somewhere else.
_Avoid_: standalone, exported (neither says what the property actually is)

**Related link**:
A code or reference URL recovered from a **source**'s metadata and judged
relevant rather than promotional.

**Provenance**:
The facts saying which **source** a **generation** came from: the title, channel
and description carried in the source's own metadata, and the canonical URL,
duration and processing date Distill knows for itself. The source-chosen half is
**extracted text**; the Distill-known half is Distill's own words.
_Avoid_: metadata (a **manifest** is full of metadata that is not provenance)

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

**Local-only processing**:
The property that a run sends no **source** content off the machine that
produces the **bundle**. When the **vision endpoint** is remote, **keyframes**
and their **extracted text** leave the machine and the run is not local-only.
Distinct from a **machine-local claim**, which is about a recorded value being
true only where produced, not about where processing happens.
_Avoid_: private, on-device (private is a consequence; on-device names the place, not the property)

**Grounding**:
Whether the vision model's reading of a **keyframe** was **corroborated** by a
second reader. The vision model is the primary reader and its **interpretation**
is recorded as authoritative; grounding is an informational note carried beside
it, not a gate that lowers confidence in an uncorroborated reading.

**Corroborated**:
Two readers, the image-text reader and the vision model, independently recovered
the same text from a **keyframe**. The word keeps this strict meaning: a lone
confident reader is never corroborated, however sure it is.
_Caveat_: today the vision model is shown the image-text reader's output in its
prompt, so the two are not truly independent - the vision reader can echo what
the image-text reader recovered. The reader-facing note therefore says "matches
the on-screen-text reader" and claims no independence; genuine independence would
need a blind second reading (a possible future change).

**Ungrounded**:
The vision model produced an **interpretation** that no second reader
corroborated. It is recorded as such and still trusted as the primary reader's
reading; the note says corroboration is absent, not that the reading is wrong.

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
- A **frame artifact** has one **keyframe**, at most one **interpretation**, one
  **grounding**, and at most one **frame salience**
- An **interpretation** comes from a **vision endpoint**; the model it serves
  enters the **options hash**, while the endpoint's address and credential are
  **machine-local claims** and stay out of every **bundle**
- **Frame salience** is judged against the **transcript**, so a **source** with
  no transcript has none
- **Local-only processing** holds only while the **vision endpoint** runs on the
  producing machine; a remote endpoint sends **keyframes** and their **extracted
  text** across the machine boundary
- **Extracted text** flows from a **source** into **frame artifacts** and the
  **transcript**, and reaches a reader only through a **redaction sink**
- A **generation** records the **provenance** of the **source** it came from;
  the source-chosen half crosses the **untrusted-data boundary** like any other
  **extracted text**
- A **self-contained render** carries its **provenance**, because the
  **manifest** that would otherwise answer for it is left behind
- A **manifest** describes what a **generation** contains; a **machine-local
  claim** describes where the producing happened and belongs to neither
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
>
> **Dev:** "I'll put the video's title at the top of the render as a heading, so
> you can tell the files apart."
> **Domain expert:** "The title is **extracted text** — whoever uploaded the
> video wrote it. As a heading it becomes Distill's own words, at the top of the
> document, above the warning that says which parts are untrusted. It goes
> inside the boundary like a **related link** label does. The processing date
> and the duration are Distill's own, so those can be prose."
>
> **Dev:** "The manifest records the output directory, so I know where the
> bundle lives."
> **Domain expert:** "You had to know where it lives to open it. That field is a
> **machine-local claim**: it can never tell a reader something they don't
> already have, and it becomes false the moment somebody copies the bundle. What
> a manifest describes is what the **generation** contains."
>
> **Dev:** "This frame is just someone dragging a window around while they talk.
> Low salience, so we drop it from the render?"
> **Domain expert:** "No. **Frame salience** is recorded, never a reason for
> Distill to drop a **keyframe**. It says the frame adds little the
> **transcript** doesn't already carry, so a reader can skip it, but relevance
> is the reader's call, not ours - a later question about the UI flow makes that
> exact frame the point. And salience is judged against the transcript, so a
> silent **source** has none to record."
>
> **Dev:** "I pointed the vision step at a hosted model to get better slide
> reads. Same bundle either way, right?"
> **Domain expert:** "The reads may be better, but that run is no longer doing
> **local-only processing**: the **keyframes** and their **extracted text** left
> the machine to reach a remote **vision endpoint**. The model it serves is part
> of the **options hash** because it changes what's produced; the endpoint's
> address and key are **machine-local claims** and stay out of the bundle. What
> changed isn't identity, it's that source content crossed the machine
> boundary."

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
- **"artifact"** was already taken by **frame artifact** when the deliverable
  needed a name, and "output" was already taken by `--output-dir`, which moves
  the cache rather than the deliverable — resolved: an **artifact** is the
  document a caller consumes and lives outside the cache; a **frame artifact**
  is a record inside a **generation**. The qualifier is load-bearing, and
  neither term may be shortened to the other.
- **"output"** named both the cache root and the thing a caller came for, which
  is why `--output-dir` moved the bundle store and nobody could find their
  reading — resolved: the **output root** holds **bundles**, the **artifact
  directory** holds **artifacts**, and a caller asking where the output went is
  asking about the second.
- **"fallback"** was the obvious word for trying a second **vision endpoint**,
  and **degradation** already forbids it — resolved: an **endpoint chain**
  *selects*, it does not degrade. Moving from a cloud endpoint to a local one
  produces a full reading from a different reader; only losing every endpoint
  and continuing on **extracted text** alone is **degradation**, which is the
  behavior that already existed and keeps its name.
- **"signed module"** was treated as a maintained list rather than a property —
  resolved: signed-ness is definitional (editing it can change bundle content),
  so an unsigned output-affecting module is a defect, not an omission.
- **"metadata"** covered both the facts naming which **source** a **generation**
  came from and the bookkeeping a **manifest** keeps about the run — resolved:
  **provenance** is the source-identifying subset, and its source-chosen half is
  **extracted text** rather than Distill's own words. A title is written by
  whoever produced the video, so a header presenting one as Distill's own words
  puts attacker-chosen text above the boundary that warns about it.
- A **manifest** recorded the output directory and the resolved source path as
  though they described the **generation** — resolved: **machine-local claim**.
  They are facts about the machine that produced it, and they stop being true
  when the artifact is read anywhere else.
- **"archive"** and **"distiller"** were proposed as Distill terms — rejected:
  Distill produces **bundles** and knows nothing about where a **render** is
  later kept or which machine keeps it. What Distill owes such a reader is a
  **self-contained render**; the collection and the topology are the caller's
  vocabulary, not the domain's.
- **"machine-local claim"** was conflated with the privacy property when
  "reverse the machine-local claim" was said for a change that lets frames leave
  the machine - resolved: a **machine-local claim** is a recorded value's
  locality (a path or address true only where produced), while **local-only
  processing** is the property that source content never leaves the producing
  machine. Superseding ADR-0001 changes the latter and leaves the former
  untouched.
- **"provider"** and **"backend"** were proposed for where the vision model runs
  - rejected: Distill speaks one OpenAI-compatible API to a configured **vision
  endpoint** and does not track who runs it, exactly as it does not track where
  a **render** is later kept. Local versus cloud is a property of the endpoint's
  address, not a second concept.
- **"grounding"** gated confidence: an uncorroborated vision reading was treated
  as low-confidence - resolved: the vision model is the primary reader and its
  **interpretation** is authoritative, so **grounding** is an informational note
  rather than a gate. **Corroborated** keeps its strict two-readers meaning, so
  a lone reader still cannot borrow the authority of agreement.
