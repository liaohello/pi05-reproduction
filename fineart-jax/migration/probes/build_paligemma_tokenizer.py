#!/usr/bin/env python3
"""Materialise a PaliGemma tokenizer directory from the pinned sentencepiece model.

The pinned asset under ASSET_ROOT/tokenizers is a 257,152-piece sentencepiece model
that already contains PaliGemma's 1,152 added pieces (`<loc0000>..<loc1023>`,
`<seg000>..<seg127>`). `transformers` needs a directory with the model plus a
tokenizer config, so this script writes one, outside the candidate tree, and
records the hashes of every byte it produced.

No network access: the only input is the pinned file, and the output is
reproducible from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "GemmaTokenizer",
    "model_max_length": 8192,
    # The reference disables BOS explicitly for the FAST tokenizer instance, which
    # implies the PaliGemma default carries it; text prompts therefore start with <bos>.
    "add_bos_token": True,
    "add_eos_token": False,
    "legacy": True,
    "clean_up_tokenization_spaces": False,
    "sp_model_kwargs": {},
    "do_lower_case": False,
    "remove_space": True,
    "keep_accents": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, out: Path) -> dict:
    if not source.is_file():
        raise SystemExit("pinned sentencepiece model is missing: " + str(source))
    out.mkdir(parents=True, exist_ok=True)
    target = out / "tokenizer.model"
    shutil.copyfile(source, target)

    config = dict(TOKENIZER_CONFIG)
    config.update(SPECIAL_TOKENS)
    (out / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "special_tokens_map.json").write_text(
        json.dumps(SPECIAL_TOKENS, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    produced = {}
    for name in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        path = out / name
        produced[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    record = {
        "schema": "fineart-jax-derived-tokenizer/v1",
        "derived_from": {"file": "paligemma_tokenizer.model", "sha256": sha256_file(source), "bytes": source.stat().st_size},
        "note": (
            "Derived offline from the pinned sentencepiece model. The Hugging Face repo "
            "google/paligemma-3b-pt-224 was not reachable, so its tokenizer_config.json "
            "could not be compared byte-for-byte; the special-token policy here mirrors how "
            "the reference calls the tokenizer: PaliGemma prepends BOS, and the FAST step disables it explicitly."
        ),
        "special_tokens": SPECIAL_TOKENS,
        "files": produced,
    }
    (out / "derivation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    record = build(Path(args.source), Path(args.out))
    print(json.dumps(record["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
