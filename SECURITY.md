# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Email the maintainer, or use [GitHub's private vulnerability reporting](https://github.com/dungle-scrubs/distill/security/advisories/new).

Include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal input is ideal).

You will get an acknowledgement within a few days.

## Scope

Distill runs local video files through system tools (`ffmpeg`, `tesseract`,
`yt-dlp`, a local Rapid-MLX server) and writes bundles to disk. Out of scope:

- Vulnerabilities in third-party dependencies — report those upstream.
- Issues that require already having code execution on the host.
