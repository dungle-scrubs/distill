# Review of RFC-01: Consolidate configuration, settlement, and test boundaries

## What was reviewed

- RFC: [01_consolidate-configuration-settlement-and-test-boundaries.rfc.md](01_consolidate-configuration-settlement-and-test-boundaries.rfc.md)
- Draft version: **not specified**. `number: 01` identifies the RFC, not a revision. The review filename therefore uses `unversioned`.
- Status: **Draft**; document date: **2026-09-05**.
- Reviewed content SHA-256: `27c150dd14cab69f87f8c8a9064b6b18439d65537d6ea3c6e92a7c924c133b91`.
- Code revision consulted: `a9b2bfa6a558f4d33467c8d85e015bf6df74fadf`.
- Method: one reviewer, whole-document read, targeted source checks, graph discovery and trace, structural validation, local type check, and focused revalidation tests. The reviewer also authored the draft in the preceding turn; this is not an independent-author review.

The RFC is unchanged. Findings below concern the specification, not whether the product work should be undertaken.

## Structural results

Command: `npx tsx /Users/kevin/dev/skills/skills/engineering/draft-rfc/scripts/validate-structure.ts docs/rfc/01_consolidate-configuration-settlement-and-test-boundaries.rfc.md`

Output, verbatim:

```json
{
  "passed": true,
  "errors": [],
  "warnings": []
}
```

## Findings

### R1. High: The replacement test interface omits settlement revalidation

**Lands in:** Proposed Changes 3 and 4, RFC lines 88-100; Testing Strategy, line 166.

The draft requires stage callables on ProcessingRun, acquisition operations on YoutubeDownloader, and the existing FrameInterpreter dependencies. It also prohibits callable-identity checks and requires orchestration tests to use constructor dependencies. It never assigns a replacement dependency for `KeySettlement._divergence`'s revalidation call. Revalidation runs before the media stages, so injecting those stages does not replace it.

**Evidence:** `src/distill/run_orchestrator.py:300` selects `pipeline.revalidate_chain` when a test patches it. `ProcessingRun._settled_after_wait` constructs KeySettlement at `src/distill/run_orchestrator.py:410`, with no revalidation argument. `tests/test_revalidation.py:853` deliberately raises from the patched callback and proves the held key is released at line 889. The dependency is needed for an existing failure-path assertion, not a hypothetical test.

**Why it matters:** An implementation can provide every listed constructor field and still have no specified route for this test after deleting the identity check. It would have to invent an additional interface, keep patching implementation globals, or weaken the assertion. The latter two defeat the stated test-boundary contract.

**Requested resolution:** Specify who owns the revalidation dependency and how ProcessingRun passes it into settlement. Include its failure contract and preserve the test that proves lock release. A single typed callable can suffice; no general dependency container is implied.

**Evidence grade:** Rung 4 for the existing failure-path test: `uv run pytest tests/test_revalidation.py -q` passed all 13 tests, including this case. The claim that the proposed interface is incomplete is rung 2, established by the RFC and source citations; a future implementation failure has not been reproduced.

### R2. Medium: The compatibility option conflicts with the configuration import prohibition

**Lands in:** Proposed Change 1, RFC lines 74-78; Open Question 2, line 222.

The configuration section says lower-level modules MUST NOT import the resolver. Open Question 2 permits direct forwarding adapters for confirmed supported APIs without stating an exception to that prohibition or restricting which APIs can be retained.

**Evidence:** The current forwarding functions are precisely in the prohibited direction: `config.resolve_options` imports configuration at `src/distill/config.py:307`; `DistillOptions.from_args` imports it at `src/distill/options.py:453`; `local_vision.load_local_vision_config` and `local_vision_config_from_args` import it at `src/distill/local_vision.py:891` and line 909. Keeping a direct forwarding adapter at one of these established import locations preserves the dependency the target architecture bans.

**Why it matters:** If the owner chooses compatibility for any of these APIs, a straightforward implementation of Open Question 2 violates Proposed Change 1. This is a conditional specification conflict, not evidence that any external consumer exists. The RFC currently offers that outcome without specifying how it can comply.

**Requested resolution:** Either limit the compatibility question to exports outside the prohibited modules, require removal of these configuration entry points explicitly, or define a temporary import exception with a removal condition. Name the affected surface so the owner can assess the actual compatibility decision.

**Evidence grade:** Rung 2, pointed at the RFC and current adapters; not exercised against external consumers. No claim about external usage is established.

### R3. Medium: Phase 1 cannot satisfy its own type-check gate

**Lands in:** Testing Strategy, RFC line 168; Implementation Plan, lines 174-175.

Every implementation phase must pass ty, but phase 1 only establishes contracts and adds characterization. Phase 2 removes the unused settlement file that causes the existing ty failure. The ordering requires the repair's result before the phase that supplies it.

**Evidence:** Running `uv run ty check` during this review exits 1 with one diagnostic:

```text
error[missing-argument]: No argument provided for required parameter `now` of bound method `AvailabilityMemo.skips`
   --> src/distill/vision_selection.py:269:29
```

The RFC already attributes that failure to this file at line 60 and schedules its deletion in phase 2. It provides no phase-1 exception or earlier repair.

**Why it matters:** A maintainer following the delivery boundaries cannot land phase 1 while meeting the stated green-tree requirement. Repairing code scheduled for deletion or silently waiving the gate are unstated choices.

**Requested resolution:** Make contract establishment a preflight that is not separately landed, combine it with the deletion, or schedule the minimal type repair before the phase-1 gate. Keep the green-tree rule explicit.

**Evidence grade:** Rung 4 for the current ty failure, reproduced with the project command. The phase-order conflict is rung 2, established by the cited requirements. No implementation change was made.

## Cleared

- **All sections were read.** The RFC has substantive scope, error, security, migration, testing, alternatives, and open-question content. Deterministic structural results are reported above rather than re-derived.
- **Audit scope is accounted for.** The disposition table includes all 17 ranked findings and maps all six extended sections. It retains the audit's rejected candidates instead of reopening them as required work. This is document-level verification, not confirmation of every audit claim.
- **Historical evidence is labeled.** Current State expressly attributes test counts and CI observations to the supplied audit. This review does not promote those historical CI observations to current facts.
- **The post-wait policy conflict is real and exposed.** ADR-0007's Consequences says availability is not revalidated after a lock wait. `src/distill/run_orchestrator.py:259` checks whether it is owed and line 267 calls revalidation; `src/distill/vision_chain.py:170` defines the threshold as half the 300-second memo TTL. The focused suite passes all 13 tests. Open Question 1 is therefore justified. The RFC does not silently claim to resolve the conflict. Code and document comparison is rung 2; execution of the focused suite is rung 4.
- **Response-only artifact warnings respect publication order.** `src/distill/run_orchestrator.py:736` commits the manifest before artifact emission at line 747; cache-hit emission also occurs at line 505. Recording a delivery warning in the response without rewriting generation content is consistent with those paths. This is a source trace, rung 3, not a new reproduction of artifact failure.
- **Identity exclusions are preserved explicitly.** The proposed option-table change retains the selected model, mode, and remoteness distinction and excludes address and credential. It calls for effective identity tests rather than trusting metadata declarations alone. This is a specification check, not a rerun of identity tests.
- **Security requirements preserve existing boundaries.** The draft explicitly retains request-time endpoint validation, remote opt-in, transport limits, redaction sinks, and credential exclusion. It does not request live credentials or a new vision client. This clears specification coverage, not transport implementation security.
- **Signature and rollback rules are coherent.** Each changed signed source state receives a signature/version update; old bundles are retained and no destructive data migration is requested.

## Not reviewed

- The original whole-repository audit was not repeated. Source checks were targeted to the RFC's ownership and migration claims. Dead-code and exhaustive caller-count claims remain audit-attributed and conditional on verification before removal.
- External API consumers were not surveyed. A repository search cannot establish their absence; Open Question 2 remains a product compatibility decision.
- The full pytest suite, Ruff checks, live vision/YouTube smokes, and current GitHub Actions status were not rerun. This review ran the structural validator, ty, and the 13-test revalidation suite. Those runs provide the evidence needed for the findings above.
- Eval scores, third-party dependency versions, and transport security were not re-audited. The RFC preserves those policies rather than proposing new implementations.
- No `.env`, credentials, or private configuration values were read. No RFC or production source was edited.

Graph checks used project `Users-kevin-dev-distill`, generation `2026-09-05T14:17:13Z`. Coverage checks for the source and test files relied on reported `no_recorded_issue` with matching metadata. This means no recorded gap, not proof of graph completeness. The untracked RFC reported `not_tracked` and was read directly in full. Graph results were supplemented by direct source reads; no exhaustive absence claim is based on them.
