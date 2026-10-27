# Job status diagnostics

## The silent run after OCR on 2026-10-27 (issue #45)

A YouTube run on a Mac Mini stopped emitting progress after OCR completed on
39 keyframes. `get-job-status` kept answering `running` with no stage and no
progress time, and the caller's 30-minute timeout expired before a bundle
arrived. A later status read reported `failed` with `E_INTERNAL`
(`BrokenPipeError`), after the caller had already timed out. That terminal
record does not establish what caused the earlier silent period.

What is established:

- The build the run used could not have answered the caller's question. Job
  records carried no stage or progress fields, so a run whose stages went
  silent had nothing to tell a poller but `running`.
- Hermetic reproductions of that build confirmed three observability gaps: the
  record could not name a stage, the vision stage announced itself nowhere
  before its first request, and per-frame progress was buffered until the
  whole frame list returned. The fix announces vision before the probe and
  reports completed frames as their results arrive. Each frame milestone is
  persisted without waiting on the note throttle, and batch-parent status
  mirrors the current child.
- The run's parent held a TCP connection to the local vision adapter. An open
  connection is the ordinary state of a run whose vision stage is in flight,
  whether that stage is working or stuck, so the connection alone establishes
  nothing about the delay.

What is not established: what delayed that particular run. The reproductions
prove the record could not have shown the stage, not what the stage was doing
while nothing was shown. No cause is claimed here, and none should be inferred
from a frozen `last_progress_at`: it records the last persisted event, not the
absence of work. See `docs/reference.md` for what the status fields do and do
not promise.
