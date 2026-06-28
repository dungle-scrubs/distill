---
name: Bug report
description: Report something that's broken
labels: ["bug"]
body:
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear description of the bug. Include the command you ran.
      placeholder: "distill process-local-video ./demo.mp4 ..."
    validations:
      required: true
  - type: textarea
    id: reproduce
    attributes:
      label: Steps to reproduce
      description: Minimal reproduction. Include the exact `distill ...` invocation.
    validations:
      required: true
  - type: input
    id: version
    attributes:
      label: Distill version
      description: Output of `distill --version` or `uv run distill --version`
    validations:
      required: true
  - type: dropdown
    id: os
    attributes:
      label: OS
      options:
        - macOS (Apple Silicon)
        - macOS (Intel)
        - Linux
        - Windows
        - Other
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant output / stack trace
      render: shell
