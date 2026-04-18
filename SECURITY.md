# Security policy

## Reporting a vulnerability

Please report security issues privately by email to
**`hinanohart@gmail.com`** with the subject prefix
`[aa-animator security]`.

I aim to acknowledge within 72 hours and to ship a fix (or publish an
advisory explaining the trade-off) within 14 days for high-severity
issues.

## Supported versions

| Version | Supported |
|---|---|
| 0.0.x (pre-release) | Best-effort only |
| 0.1.x (first PyPI release) | Yes, security fixes backported |

## What counts as security-relevant

- **Arbitrary code execution** triggered by a maliciously crafted input
  image or depth map processed by the pipeline
- **Path traversal** in output-path arguments that writes files outside
  the user-specified directory
- **Supply-chain concerns** with the packaging (Trusted Publishing
  configuration, CI token scope, signed releases)
- **Model weight injection** — a scenario where aa-animator downloads
  and executes unexpected code through HuggingFace Hub

## What is not in scope

- Quality or accuracy of the ASCII art output
- Performance regressions or high-memory usage under normal inputs
- Root-level compromise of the host machine; if an attacker can write
  to the user's Python environment they can do far worse than affect
  this package
- Issues in optional extras (`[matte]`, `[i2v]`) that arise solely
  from those third-party packages, not from aa-animator's usage of them

## Embargo policy

Coordinated disclosure preferred. Once a fix is released, a short
advisory is added to [CHANGELOG.md](CHANGELOG.md) crediting the reporter
(unless they prefer to remain anonymous).
