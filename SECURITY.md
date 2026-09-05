# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Email the maintainer, or use [GitHub's private vulnerability reporting](https://github.com/dungle-scrubs/distill/security/advisories/new).

Include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal input is ideal).

You will get an acknowledgement within a few days.

## Scope

Distill processes local and YouTube video sources through external tools and
writes bundles and artifacts to disk. Vision endpoints include the default
local Rapid-MLX server and profile-compliant remote servers. Remote access
requires explicit opt-in and HTTPS. Per-request address checks, disabled
redirects and proxies, and response and run budgets remain in force. See
[ADR-0005](docs/adr/0005-any-profile-compliant-vision-endpoint.md).

An `api_key` in `distill.json` is stored as plaintext. Keep configuration files
out of version control and restrict access to them. Use `api_key_env` to refer
to an environment variable when credentials should stay out of the file.
Distill excludes credentials from bundle identity and public configuration
output. Content redaction does not make an artifact safe to publish without review.

Out of scope:

- Vulnerabilities in third-party dependencies - report those upstream.
- Issues that require already having code execution on the host.
