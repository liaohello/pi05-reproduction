# Licences and release gate

Binding rules from AGENTS.md.

## Licence compliance

Original code is MIT. Keep copyright notices of adapted third-party files, mark modifications, and list component, source and licence in [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md). Check model weights' and datasets' terms before redistributing anything derived from them. This repository does not redistribute weights or datasets.

## Lean development

Implement the smallest thing the current phase needs. Do not build frameworks ahead of results: contracts may be defined (`inference/CONTRACT.md`), implementations may not. Gates are the exception — anything the release gate or [`repro doctor`](../scripts/repro/repro.py) calls is part of the gate, not a framework, and is built when the gate is defined.

## Release gate

A public release requires, in order: `./scripts/run.sh repro doctor` green at R1 → security scan clean → licence check → migrated-file review → current experiment regression green → tag. Until then work stays local.
