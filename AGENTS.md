# Repository policy

These binding rules apply to every agent and human working in this public repository. The machines it runs on are private. Read all linked policy pages before changing, committing, pushing, or releasing files.

1. [Scope and evidence integrity](policy/scope-evidence.md) — what may be added, and what a number is allowed to claim.
2. [Source provenance and sensitive data](policy/provenance-security.md) — how third-party source is pinned, and what must never appear in a tracked file.
3. [Migration and evidence review](policy/migration-review.md) — how files enter this history, and the security gate before every commit.
4. [Licences and release gate](policy/licences-release.md) — licence obligations and the order of the release gate.
5. [Reproducibility](policy/reproducibility.md) — the R1/R2/R3 levels, and the rule that an unpinned dependency is a blocking defect rather than a footnote.

## Minimum state

A clean clone must reach R1 with no private configuration:

```bash
./scripts/run.sh repro doctor
```

That command, not a paragraph, is the definition of "usable". If it fails on a clean clone the repository is broken, however complete the documentation looks.

## Citing a clause

Cite by path (`policy/provenance-security.md`), not by number. The numbers above are navigation only and shift when a page is added; a citation such as "§3" silently changes meaning.
