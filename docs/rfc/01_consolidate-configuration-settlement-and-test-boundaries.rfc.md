---
number: 01
version: "02"
title: "Consolidate configuration, settlement, and test boundaries"
type: refactor
status: Accepted
author: Codex (draft from user-supplied audit)
date: 2026-09-05
---

# RFC-01: Consolidate configuration, settlement, and test boundaries

Version 02 records the owner-approved decisions and is accepted for implementation. Version 01 addressed the review of the unversioned draft at SHA-256 `27c150dd14cab69f87f8c8a9064b6b18439d65537d6ea3c6e92a7c924c133b91`.

## Abstract

Distill's audit identifies repeated configuration rules, competing settlement implementations, and test patches that influence production dispatch. This RFC assigns each rule to one owner and makes the existing constructor boundaries the test interface. It preserves processing contracts, cache ordering, endpoint security, and durable data formats while removing redundant implementations. It also records the audit's correctness and maintenance findings as companion work, without making those fixes depend on architectural approval.

## Introduction

Distill turns a recorded video into a durable, re-readable account of what was said and shown, so an LLM agent can consume the recording without watching it. Its users are operators processing recordings and agents consuming the resulting artifacts. Maintainers need to change those operations without reproducing configuration, identity, and locking rules at each call site.

This is a refactor RFC because ownership, dependency direction, and compatibility are design decisions. It covers configuration folding, option metadata, endpoint settlement, constructor dependencies, source identity, and the corresponding test boundaries across `src/distill/` and the test suite. The supplied whole-repository audit is its input, including both deepen and drift findings and all six extended sections.

Bug fixes, formatting, CI maintenance, documentation corrections, and instruction-file repairs are tracked below as companion tasks. They do not need an RFC to establish that incorrect behavior needs correction. This draft neither implements those tasks nor claims they have passed verification.

### Scope and fit check

The architectural changes reorganize existing behavior and hold product scope. The companion corrections to attribution, artifact-failure reporting, traceback handling, and scene-detection warnings serve existing users: they make the account accurate and let operators see failures. Each holds scope by restoring an existing documented obligation.

The following are excluded:

- New vision clients, provider dispatch, server lifecycle management, or a new default model: no existing user need in this audit requires them.
- A BundleStore split based on file length: the audit finds a small external interface protecting substantial lifecycle rules.
- Changes to redaction, grounding, source acquisition policy, or endpoint selection preference: these have existing domain decisions.
- A Pydantic, Typer, or dependency-manager migration: framework replacement is not required to remove duplicate ownership and would need a separate compatibility decision.
- Removing remote endpoints or changing branch policy to match historical PR usage: repository history is evidence of practice, not authorization to change policy.

## Terminology

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

Domain terms retain their definitions in [CONTEXT.md](../../CONTEXT.md):

- **Source fingerprint** identifies source content independent of processing choices.
- **Options hash** identifies output-affecting processing choices.
- **Bundle key** combines the source fingerprint and options hash.
- **Lock key** coordinates acquisition of a remote source.
- **Generation** is immutable published bundle content; the manifest names the active generation.
- **Artifact** is the self-contained deliverable outside the cache.
- **Selected endpoint** is the single reader whose model enters identity.
- **Machine-local claim** describes the producing machine, not generation content.

**Configuration snapshot** means the in-memory inputs read for one configuration resolution. It is not a new on-disk format. **Settlement** means claiming the appropriate bundle after a lock wait, including the existing decision about whether selection needs reconsideration. The owner chose to preserve production revalidation and correct ADR-0007.

## Current State

Evidence in this section is attributed to the user-supplied audit, not independently reproduced for this RFC. Its baseline reports 1,363 passing tests and two skips on each of two runs, clean Ruff lint, 32 files needing formatting, and one ty failure. Its CI observation reports four failing matrix legs since the 2026-08-08 PR #39 merge. Those are historical observations, not claims about current CI status.

| Concern | Audit evidence and consequence |
| --- | --- |
| Configuration | `configuration.py` copies readers and validators from `config.py`, `options.py`, and `local_vision.py`; one run reads the main configuration repeatedly. Lazy imports and ImportError fallbacks preserve competing paths. |
| Settlement | `KeySettlement` is reported as the production owner; three unused `ProcessingRun` methods duplicate it. `vision_selection.py` is reported to have no production callers and contains the ty failure. |
| Test dependencies | `acquisition.py` looks up patched names through `source.py`; `run_orchestrator.py` imports facade shims from `pipeline.py`; identity comparisons select test-dependent behavior. |
| Identity | `source_identity.py`, `media_inspect.py`, `acquisition.py`, and `source.py` repeat hashing or expose aliases for the same domain concepts. |
| Option metadata | Vision options appear in parser lists, resolver lists, special cases, and option fields. |
| Redundant surface | Four signed bundle stub modules export no implementation; several vision wrappers and private facade exports add names without owning policy. |

The draft reads the repository's domain context and ADRs. ADR-0005 supersedes the local-only endpoint restriction in ADR-0001. ADR-0007 explicitly says availability is not revalidated after a lock wait, which conflicts with the audit's description of the production settlement path. The owner resolved this inconsistency by retaining production behavior and requiring an ADR correction.

## Proposed Changes

### 1. Configuration has one reader and one folder

`configuration.py` MUST own reading configuration documents and folding their layers into resolved run configuration. A resolution MUST read each distinct configuration document at most once and reuse that snapshot for general and vision settings. This does not cache configuration across runs or suppress request-time security checks.

`options.py` MUST own option definitions and reusable option validators. The resolver MUST call those validators rather than copying them. `config.py` SHOULD retain only configuration-directory discovery; resolved result types SHOULD live with the resolver. `local_vision.py` MUST retain vision value types and settle-time validation, but MUST NOT independently read or fold configuration files.

Existing precedence, accepted representations, defaults, strict versus forgiving file behavior, and coded refusals MUST remain unchanged. Before replacement, tests MUST characterize omitted values separately from explicit false, zero, empty, and null values wherever the current contract distinguishes them. Consolidating reads MUST NOT turn malformed general configuration into a silently ignored vision configuration error.

The target dependency direction is callers to `configuration`, then to option definitions, vision value types, and directory discovery. Those lower-level modules MUST NOT import the resolver except inside the temporary compatibility adapters defined below. Production entry points MUST invoke the resolver explicitly. Duplicate readers, unused result protocols, working-note scaffolding, and ImportError fallback bodies MUST be removed after callers migrate.

Only for a known supported consumer or documented API commitment, a temporary adapter MAY remain at its existing import location. The candidate surface is `config.general_config`, `config.environment_options`, `config.resolve_options`, `DistillOptions.from_args`, `local_vision.load_local_vision_config`, and `local_vision.local_vision_config_from_args`. Before an adapter is retained, it MUST be listed with its replacement entry point, supported consumer or documented commitment, and removal release. Any additional candidate requires an explicit addition to that list. Each adapter MUST use a function-local import and delegate to the single resolver owner; it MUST NOT read files, fold layers, duplicate validators, catch ImportError as a fallback, or detect patches. Internal production callers MUST migrate off these adapters in the same change. Only argument and return-shape conversion needed to preserve the supported contract is permitted. The import exception expires when the listed adapter is removed in its recorded release; it does not alter the final dependency direction.

### 2. Option metadata describes the existing vision surface

Every existing vision option MUST have one OptionSpec row describing its parser representation, default or omission semantics, consuming layer, and identity disposition. Processing and diagnostics commands MUST derive their applicable options from that table. Command applicability MUST be explicit so diagnostics do not accidentally gain processing-only flags.

The table MUST describe existing flags, not add new flags. Declared option types MUST agree with resolved field types. A configured endpoint chain is input to selection, not itself bundle identity. Identity MUST continue to use the selected model, selected remoteness, and vision mode under ADR-0004 and ADR-0007; endpoint addresses and credentials MUST remain excluded. Tests MUST verify effective identity after selection, not only inspect `cache_key` declarations.

### 3. Settlement has one implementation

`vision_chain.resolve_chain` MUST remain the selection walk, with source resolution adapting its result for processing. `KeySettlement` MUST be the sole owner of settlement after a lock wait. The unused `ProcessingRun` settlement methods and `vision_selection.py` MUST be removed once caller and test checks confirm the audit's reachability claim.

The retained implementation MUST preserve cache-before-network ordering, candidate preference order, the run-wide deadline, lock release on all exits, and one selected reader per generation. A disappeared cache hit MUST be rechecked under the appropriate lock. A changed bundle key MUST NOT result in publishing under a key derived for another reader. Settlement MUST retain its existing finite rekey bound and abandonment outcomes unless a separately approved behavior correction changes them.

No new selection abstraction or generalized Store/Prober framework is required. Settlement MUST retain the existing threshold (waited time at least REVALIDATE_AFTER_WAIT_SEC), at most two revalidation calls, and at most one rekey. ADR-0007 MUST be corrected to describe this behavior. Revalidation does not run when the lock acquisition returns a cached snapshot.

### 4. Constructors expose the dependencies tests replace

`ProcessingRun` MUST accept typed stage callables with production defaults. `YoutubeDownloader` MUST accept the clock and the select, validate, and promote operations it uses. Existing `FrameInterpreter` probe and interpretation dependencies SHOULD remain its test surface. Injected dependencies MUST be instance-local; tests MUST NOT require global monkeypatch detection to alter dispatch.

`ProcessingRun` MUST also accept an instance-local `revalidate` callable, defaulting directly to `source.revalidate_chain`, and pass that same callable to the `KeySettlement` constructor. The callable accepts the current DistillOptions, source fingerprint, source type, and output root and returns ChainRevalidation. `KeySettlement` MUST own when it is invoked and how its result changes the held key; injecting the callable MUST NOT change the wait threshold, call bound, or rekey policy. Both divergence checks MUST use this dependency. A raised exception MUST follow settlement's existing abandonment path, release any held bundle lock, and propagate rather than being converted into endpoint unavailability. No media stage may start after that failure.

Default dependencies MUST point directly to their implementation owners. Production code MUST NOT inspect `sys.modules`, callable identity, fake class names, or caught TypeError to discover a test double. `source.py` and `pipeline.py` MUST lose exports maintained solely as patch targets after tests and callers migrate. Acquisition MUST NOT import its source facade, and orchestration MUST NOT import its pipeline facade.

Transport tests MAY replace the requestor at the transport boundary. Orchestration tests MUST replace the constructor dependency instead of reaching through a facade. Internal and test-only import names MUST be removed. A temporary compatibility adapter is permitted only for a known supported consumer or documented API commitment under Proposed Change 1.

### 5. Source identity and source types have explicit owners

`source_identity.py` MUST own one operation per domain concept: source fingerprint, lock key, and bundle key. Source resolution and acquisition MUST call that owner rather than hash video IDs inline. Duplicate query-normalization constants MUST have one owner used by identity and YouTube URL handling.

Existing fingerprint and key algorithms MUST remain byte-for-byte compatible apart from the required pipeline-version contribution. The aliases using competing vocabulary SHOULD be deleted. The unused SourceIdentity wrapper SHOULD be deleted rather than wired into production merely to preserve it.

`ProcessingRun`, `KeySettlement`, `process_resolved_source`, and acquisition-lease release MUST use the concrete SourceInfo contract, with explicit optional fields where valid. They MUST NOT accept arbitrary objects through Any or use getattr to hide absent contract fields. Type-only imports MAY avoid runtime dependency cycles; moving the value type is justified only if a runtime cycle remains.

### 6. Remove artifacts of incomplete reorganizations

The four empty `bundle_layout`, `bundle_lifecycle`, `bundle_locks`, and `bundle_prune` modules MUST be deleted after confirming that no supported caller relies on them. Their signature entries MUST be removed at the same time. The real BundleStore implementation remains intact.

Shared lock logging SHOULD be owned by the lock primitive, without callers importing `_bundle_log`. Vision call wrappers SHOULD be reduced only when they add no distinct error or probe contract. This RFC does not adopt the audit's proposed public `checked: bool`: a caller-controlled flag can bypass a required probe and needs more justification than wrapper count alone.

## Error Handling

Except for the approved artifact-delivery failure contract below, the refactor MUST preserve existing error codes, stages, warning shapes, and exit statuses. Invalid processing options MUST continue through the existing `E_BAD_OPTIONS` boundary. Required tool absence remains fatal when generation production needs that tool; cache hits retain the existing capability bypass. Optional capability loss MUST remain a structured warning. An unavailable endpoint MUST NOT prevent later configured endpoints from being considered within the existing budget.

Companion fixes restore these error obligations:

- Artifact emission failure: preserve the published generation and return a coded delivery failure, with a bounded, redacted message identifying the saved bundle and stating that artifact delivery failed. Single-source commands and call-tool MUST report failure, and CLI failure MUST use a nonzero exit status. Batch processing MUST record the affected item as failed and preserve its existing continuation policy. Job status MUST record failure. The error MUST distinguish a saved bundle from failed artifact delivery; it MUST NOT claim that generation production failed. Do not rewrite or abandon a published generation to record a later delivery failure. A normal retry MUST be able to use the saved bundle and retry delivery without processing the source again when the cache is still available. This applies to both new generations and cache hits.
- Scene-detector failure: distinguish a successful empty result from unavailable or failed detection; retain interval selection with a warning carried into the bundle.
- Unexpected exceptions: honor the documented `DISTILL_TRACEBACK` behavior at call-tool and batch conversion boundaries, with cleanup preserved. Expected DistillError handling remains coded.
- Job failure messages: use the shared bounded error representation rather than an uncapped exception string.

New warning codes, if needed, MUST follow the existing warning convention and be pinned by response tests. This RFC introduces no generic retry mechanism or new error taxonomy.

## Security Considerations

Source content and recovered text remain untrusted. Redaction sinks, prompt boundaries, credential carriers, and response caps MUST remain in force when dependencies and readers move. Configuration snapshots MUST NOT serialize credentials into debugging output, warnings, test snapshots, hashes, or bundles.

Endpoint transport MUST retain explicit remote opt-in, HTTPS for non-loopback hosts, per-request address validation, approved-address dialing, disabled redirects and proxies, and existing byte and wall-clock limits. Consolidated configuration validation MUST NOT replace request-time validation. Refactoring tests MUST use synthetic credentials and fake endpoints; it does not require reading `.env` or live account credentials.

Incorrect render attribution can mislead a downstream agent about whether a vision endpoint produced interpretations. The companion fix MUST derive attribution from persisted generation evidence, including filtered renders and cache reads, rather than the current process configuration. If old data cannot establish a reader, it MUST NOT assert that no endpoint read the frames. Model names taken from endpoint data remain untrusted text at rendering boundaries.

Artifact emission stays outside the cache and retains output-root confinement. Failure messages MUST NOT leak credentials or raw remote response bodies. SECURITY.md needs to describe already-supported remote endpoints and plaintext configuration credential storage; that documentation task does not authorize new storage or remote-processing behavior.

## Migration Strategy

Migrate one ownership boundary at a time, updating its callers and tests in the same change. No feature flag, dual resolver, or alternate settlement walk is introduced. Public command names, flag spellings, configuration formats, JSON contracts, bundle formats, and default model remain stable, except for separately tested correctness fixes.

Every landed change to signed source MUST update the pipeline signature, bump the pipeline version, and append the signature/version history entry as required by AGENTS.md. One bump covers an atomic change, not multiple later commits with different signed contents. Old bundles remain on disk; this RFC requires no destructive migration or pruning.

Rollback restores a coherent implementation and its callers. A rollback delivered as new signed source MUST follow the signature/version rules rather than reuse a historical version for different bytes. Data formats are unchanged, so rollback does not require rewriting published generations. Main remains the landing branch and each landed state MUST pass the required suite.

## Risk Assessment

| Risk | Control and stop condition |
| --- | --- |
| Changed configuration precedence or omission semantics | Characterization matrix and read-count checks; stop if resolved values differ without an approved bug fix. |
| Wrong reader identity after lock contention | Cache, prune-race, deadline, and rekey tests; stop on mismatched key or mixed-reader generation. |
| Network leaks from migrated test fakes | Hermetic integration tests that fail unexpected network access; stop before removing the old stub if replacement coverage is incomplete. |
| Broken import consumers | Classify supported import surface before deletion; retain adapters only with compatibility evidence and a removal release. |
| Stale or misleading render attribution | Test persisted evidence, filtered views, OCR-only generations, and old records without attribution. |
| Excess cache invalidation | Required version bumps are expected; avoid unrelated signed-code formatting churn inside semantic changes. |

## Testing Strategy

The implementing maintainer MUST reproduce each companion bug through its user-facing entry point before fixing it. Audit probes are useful starting evidence, not substitutes for regression tests of the repaired behavior.

Configuration tests MUST cover file/environment/argument precedence, false and omitted values, invalid types, endpoint-chain overrides, credential refusal without value disclosure, and one read per distinct file. Identity tests MUST compare equivalent endpoint addresses with the same selected model and distinguish changed model, mode, or remoteness.

Artifact-delivery tests MUST cover failure after publication, failure on a cache hit, CLI and call-tool failure reporting, failed job and batch-item status, and successful retry from the saved bundle without rerunning media stages.

Settlement tests MUST cover cache hits without probes, missing required tools on cache hits, lock waits, vanished cache entries, exhausted chains, raising probes, memo expiry and backward clocks, bounded rekeying, and cleanup on exceptions. Existing YouTube URL cases that require metadata resolution before a key is known MUST remain covered.

Tests MUST exercise injected dependencies through real constructors. Removing a facade patch MUST preserve the assertion it enabled. In particular, `test_a_second_walk_that_fails_gives_the_key_back` MUST inject a raising revalidation callable through ProcessingRun, verify that the same error reaches the caller, and prove with a fresh lock descriptor that the held key was released. It MUST also verify that no media stage started. This replaces its patch of `pipeline.revalidate_chain`. The default suite MUST remain offline and hermetic, with live vision and YouTube smoke tests gated as before.

Each landed implementation phase MUST pass `uv run pytest`, `uv run ruff check .`, and `uv run ty check`. The preflight below records existing failures and is not a separately landed phase. The first landing combines characterization with the unused-code removal that resolves the known ty failure; it receives no type-check waiver. The formatting companion task adds `uv run ruff format --check .` to CI after formatting the existing drift. Signature tests MUST pass after each signed change. Baseline counts are not acceptance criteria; regressions cannot be hidden by deleting tests with removed facades.

## Implementation Plan

The implementing maintainer owns each phase. The repository owner approved the three decisions below and requested implementation of this version.

**Preflight, not a separate landing:** Recheck audit claims against the implementation revision and record the baseline failures. Apply the approved decisions below and record any evidence-backed adapters and their removal releases. Confirm the dead-path claims before preparing deletions. If a supported consumer prevents removal, revise the first delivery boundary before proceeding rather than waive its checks.

1. **Establish contracts and remove unused code.** Add missing behavior characterization, then remove unused settlement and stub code in the same landing. Retain one settlement implementation and update signatures. Verify typing, caller migration, cache behavior, and lock cleanup. This phase MUST clear the existing ty failure before landing. Roll back the whole deletion if a live dependency emerges.
2. **Consolidate configuration and option metadata.** Deliver the single snapshot, shared validators, and derived command option sets together. Verify the precedence and identity matrices before removing fallback readers. Any approved compatibility adapters follow Proposed Change 1 and carry no internal production callers.
3. **Move tests to constructor dependencies.** Migrate acquisition, orchestration, and vision tests before deleting facade exports. Pass the revalidation dependency from ProcessingRun into KeySettlement and preserve the lock-release failure test. Verify hermetic integration paths and concrete source contracts.
4. **Consolidate identity and small ownership defects.** Remove duplicate algorithms and unnecessary aliases, retain known key fixtures, and settle logging ownership. Complete documentation owner updates.
5. **Close companion work.** Independently reproducible bug fixes can land before phases 1-4 if they meet the same green-tree checks. Finish CI formatting and instruction repairs, then verify all audit dispositions below. Roll back any behavioral correction independently if its regression tests expose an unresolved contract.

### Response to the prior review

This revision answers the [review of the unversioned draft](01_consolidate-configuration-settlement-and-test-boundaries.review-unversioned.md). The review remains a record of that earlier content, not a review of version 01.

| Finding | Resolution in version 01 |
| --- | --- |
| R1. Missing settlement revalidation dependency | Added the ProcessingRun-to-KeySettlement callable contract, error propagation and lock release requirements, and the constructor-based replacement for the existing failure test. |
| R2. Compatibility conflicts with import prohibition | Added a narrow temporary exception for explicitly recorded configuration adapters, with function-local imports, no internal production callers, and a required removal release. The approved compatibility rule determines which APIs qualify. |
| R3. Phase 1 cannot pass ty | Made baseline discovery a preflight and combined characterization with unused-code removal in the first landing, which must pass ty without a waiver. |

### Audit disposition

This table answers the supplied audit, not a review of an earlier RFC. Numbers match its ranked list. No finding is treated as already fixed.

| Audit item | Disposition in this draft |
| --- | --- |
| 1. False no-endpoint render note | Companion bug fix; persisted attribution requirements in Security Considerations. |
| 2. Configuration folding | Adopted in proposed change 1. |
| 3. Duplicate settlement and ty failure | Adopted in change 3; production policy retained and ADR correction required. |
| 4. Artifact failure visibility | Companion bug fix; coded artifact-delivery failure required while retaining the saved bundle. |
| 5. Facade test seams | Adopted in change 4. |
| 6. Formatting, noqa, red type check | Companion maintenance: format drift, enforce format checks, remove stale suppressions and enable unused-suppression checking after validation; repair type failure without suppressing it. Recheck current CI rather than repeat historical status as current. |
| 7. Documentation claims | Companion documentation: correct transport patch targets, point eval claims to their evidence files, reconcile endpoint scope with ADR-0005/0007, and align traceback docs with the repair. Retain branch policy unless the owner changes it. |
| 8. Empty signed stubs | Adopted in change 6. |
| 9. Source identity duplication | Adopted in change 5; delete the unused wrapper rather than add a use for it. |
| 10. Repeated vision option vocabulary | Adopted in change 2. |
| 11. Traceback conversion | Companion bug fix in Error Handling. |
| 12. Silent scene-detector failure | Companion bug fix in Error Handling. |
| 13. Missing CLAUDE.md | Companion instruction repair: add sibling file with first non-comment instruction `@AGENTS.md`, without copied shared guidance. |
| 14. SECURITY.md scope | Companion documentation in Security Considerations. |
| 15. yt-dlp installation mechanism | Companion documentation: distinguish package installation from subprocess invocation and explain the `uv run` environment. Retain the declared dependency and executable invocation unless verification identifies a separate defect; these mechanisms are not inherently contradictory. |
| 16. Source typed as Any | Adopted in change 5. |
| 17. Small ownership and error defects | Correct stale owner docstrings and timeout source claims; adopt lock log ownership and bounded job failures. Wrapper reduction is conditional; the proposed checked boolean is not adopted without a distinct contract. |

The six extended sections are covered as follows: correctness and error handling by items 1, 4, 11, 12, and 17; test health by items 3, 5, and 6; documentation drift by items 7, 15, and 17; security by item 14 and preserved transport/redaction invariants; dependency and toolchain health by items 6 and 15; instruction-file convention by item 13. Floor-only dependency constraints and absent framework choices do not alone justify upgrades or replacements.

The audit's rejected candidates remain protected: BundleStore's deep interface, the shared ExclusiveLock primitive, the documented warning-record representation, Carrier and its redaction ordering, transport plus endpoint policy in one module, the shared subprocess and durable-write paths, ADR-0006 redaction behavior, and the exempt measurement harness. Removing test facades does not reverse the YouTube/media inspection split.

## Alternatives Considered

- **Keep duplicate paths and synchronize them with tests.** This offers compatibility but leaves precedence and settlement policy independently editable. One owner reduces that failure mode directly.
- **Replace validators and the CLI framework during consolidation.** This could supply broader schema tooling, but changes accepted input and error behavior beyond the audit's ownership problem. Deferred to a separate proposal with compatibility evidence.
- **Introduce a generalized dependency container or selection framework.** This centralizes wiring but adds another interface. Existing constructors cover the dependencies identified by the audit.
- **Split BundleStore now.** Smaller files would not by themselves deepen its already-small interface. Delete the false split stubs and retain its lifecycle rules together.
- **Expose `checked: bool` for vision calls.** This reduces wrapper count but transfers probe-state correctness to callers. Prefer the existing interpreter-owned probe boundary unless a concrete caller needs another contract.

## Decisions

The repository owner approved these recommendations and requested implementation of version 02.

1. **Revalidation:** Keep the existing post-wait endpoint checks and bounded rekey behavior. Correct ADR-0007 to match.
2. **Compatibility:** Remove internal and test-only import names. Retain a temporary adapter only for a known supported consumer or documented API commitment, with a replacement and removal release. Commands and flags remain unchanged. No adapter is justified by hypothetical consumers alone.
3. **Artifact delivery:** Report failure when artifact writing fails, preserve the saved bundle, and allow a normal retry to use that bundle. This replaces version 01's recommendation to report success with a warning.

## Open Questions

None. Compatibility evidence and adapter removal releases, if any adapters qualify, are implementation records rather than unresolved policy decisions.

## References

### Normative

- [CONTEXT.md](../../CONTEXT.md) - product purpose and domain vocabulary.
- [AGENTS.md](../../AGENTS.md) - capability policy, signature rules, tests, and landing policy; endpoint-only wording needs reconciliation with the superseding ADR.
- [ADR-0002](../adr/0002-degradation-over-fail-fast-for-optional-capabilities.md) - structured degradation and required capability failures.
- [ADR-0003](../adr/0003-cache-invalidation-by-pipeline-signature.md) - versioned cache invalidation.
- [ADR-0004](../adr/0004-bundle-identity-excludes-environment-facts.md) - identity exclusions and remoteness amendment.
- [ADR-0005](../adr/0005-any-profile-compliant-vision-endpoint.md) - endpoint profile, remote opt-in, and security boundaries.
- [ADR-0006](../adr/0006-redaction-targets-credential-shape-not-length.md) - redaction invariants.
- [ADR-0007](../adr/0007-an-endpoint-chain-selects-one-reader.md) - reader selection and cache ordering; post-wait wording must be corrected to match the approved production policy.

### Informative

- User-supplied “Code audit - distill (whole repository, deepen + drift)”, provided in the drafting conversation on 2026-09-05 - source of the findings and reported code, traced, ran, live, and estimate evidence. Its ranked findings are preserved in the disposition table; the original probe script and CI logs were not supplied as repository artifacts.
- [README](../../README.md) and [CONTRIBUTING](../../CONTRIBUTING.md) - existing command, traceback, installation, and contributor claims to reconcile.
- [Eval documentation](../../tests/evals/README.md) - canonical route to measured model-selection evidence; this RFC repeats no eval scores.
