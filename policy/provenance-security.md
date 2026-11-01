# Source provenance and sensitive data

Binding rules from AGENTS.md.

## Source provenance

Third-party source (OpenPI, LIBERO, simulators) is **not** committed. It must still be obtainable: every dependency carries a machine-readable pin in [`configs/repro/manifest.json`](../configs/repro/manifest.json) — an image digest, an archive SHA-256, or at minimum file count plus byte total plus aggregate digest.

A pin without a retrieval path is a blocking defect, not a footnote: record `null` with `status: blocked` and let `repro doctor` report it as unresolved. Human-readable fingerprints stay in [`third_party/PROVENANCE.md`](../third_party/PROVENANCE.md) and must agree with the manifest. Do not claim a git commit for code exported from an image without a verifiable upstream revision.

## Sensitive data protection

Forbidden in tracked files, commit messages, screenshots, videos, logs or reports:

- public IPv4/IPv6 addresses, private domains, internal host names;
- SSH or service ports, port mappings;
- SSH usernames, private absolute paths, workdir names;
- SSH keys, passwords, tokens, cookies, registry credentials or private registry addresses.

Use environment variables and placeholders in `configs/hosts/example.env`. Real values live only in ignored `scripts/deploy/config.local.json` or the shell environment, outside Git.
