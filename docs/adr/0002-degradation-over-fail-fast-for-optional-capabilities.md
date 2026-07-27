# Degradation over fail-fast for optional capabilities

Distill depends on several external tools, and any of them can be absent on a
stranger's machine. When an **optional capability** is unavailable — image-text
extraction, vision interpretation — a run continues with reduced output and a
recorded **warning** rather than failing, because a bundle with a transcript and
keyframes but no captions is still worth having, and a user who wanted captions
learns why they are missing from the warning rather than from a stack trace.

The consequence a reader should expect: error paths for optional capabilities
return warnings instead of raising, which looks like swallowed errors unless you
know this was chosen. The obligation that comes with it is that every warning
must survive into the bundle — degradation that produces a thin bundle *silently*
is the failure mode this decision creates, and is worse than the crash it
avoids.

This applies only to optional capabilities. A missing **required capability**
is a fatal error, and which tools are which must be stated rather than left to
be inferred from whichever code path happens to raise.
