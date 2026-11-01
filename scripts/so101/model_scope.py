"""Resolve a saved SO101 checkpoint's OpenPI model variants without importing JAX.

Documents are ordinary decoded JSON mappings. Pass higher-priority saved run
metadata first and protocol documents afterward. Historical runs that predate
variant metadata use the E2/E4-T LoRA architecture as a conservative fallback.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

_PALIGEMMA = {"gemma_2b", "gemma_2b_lora", "dummy"}
_ACTION = {"gemma_300m", "gemma_300m_lora", "dummy"}
_HISTORICAL = {
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
    "freeze_vision_encoder": True,
}


def _mapping(value: object, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _variant(value: object, key: str, where: str) -> str:
    allowed = _PALIGEMMA if key == "paligemma_variant" else _ACTION
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"invalid {where}.{key}: {value!r}")
    return value


def _freeze_from_filter(value: object, where: str) -> bool | None:
    if not isinstance(value, str):
        raise ValueError(f"{where}.freeze_filter must be a string")
    text = value.lower()
    vision_markers = ("piligemma/img", "paligemma/img", "vision", "siglip")
    if any(marker in text for marker in vision_markers):
        return True
    # OpenPI's known language-only freeze filters name the llm path and omit
    # image-tower paths. Do not eval or repr arbitrary objects from snapshots.
    if "llm" in text and any(token in text for token in ("pathregex", "regex", "pattern")):
        return False
    raise ValueError(f"unrecognized {where}.freeze_filter representation")


def _freeze_from_rule(value: object, where: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}.freeze_rule must be a non-empty string")
    text = value.lower()
    match = re.search(r"(?:siglip|vision)[^.;]{0,70}\b(frozen|freeze|trainable|unfrozen)\b", text)
    if match:
        return match.group(1) in {"frozen", "freeze"}
    raise ValueError(f"unrecognized {where}.freeze_rule")


def _candidate(document: Mapping, index: int, key: str) -> tuple[object, str] | None:
    """Return the first raw candidate for one field without validating fallbacks."""
    label = f"document[{index}]"
    explicit = None
    if "model_scope" in document:
        explicit = _mapping(document["model_scope"], f"{label}.model_scope")
        if key in explicit:
            return explicit[key], f"{label}.model_scope"

    if key == "freeze_vision_encoder":
        config = document.get("config")
        if config is not None:
            config = _mapping(config, f"{label}.config")
            if "freeze_filter" in config:
                return config["freeze_filter"], f"{label}.config.freeze_filter"
        if "freeze_filter" in document:
            return document["freeze_filter"], f"{label}.freeze_filter"
    else:
        config = document.get("config")
        if config is not None:
            config = _mapping(config, f"{label}.config")
            if "model" in config:
                model = _mapping(config["model"], f"{label}.config.model")
                if key in model:
                    return model[key], f"{label}.config.model"
        if "model" in document:
            model = _mapping(document["model"], f"{label}.model")
            if key in model:
                return model[key], f"{label}.model"

    if "initialisation" in document:
        initialisation = _mapping(document["initialisation"], f"{label}.initialisation")
        protocol_key = "freeze_rule" if key == "freeze_vision_encoder" else key
        if protocol_key in initialisation:
            return initialisation[protocol_key], f"{label}.initialisation.{protocol_key}"
    return None


def _resolve_candidate(key: str, candidate: tuple[object, str]) -> object:
    value, where = candidate
    if key == "freeze_vision_encoder":
        if where.endswith(".model_scope"):
            if type(value) is not bool:
                raise ValueError(f"{where}.{key} must be boolean")
            return value
        if where.endswith("freeze_filter"):
            return _freeze_from_filter(value, where)
        return _freeze_from_rule(value, where.rsplit(".", 1)[0])
    return _variant(value, key, where)


def resolve_model_scope(*documents: Mapping) -> dict:
    """Return model variants and vision-freeze setting from saved JSON documents.

    Documents are ordered by priority, highest first. Explicit ``model_scope``
    and snapshot fields are read before protocol ``initialisation`` fallback
    fields within each document. Missing historical metadata falls back to the
    known E2/E4-T dual-LoRA, frozen-vision configuration.
    """
    parsed = [_mapping(raw, f"document[{index}]") for index, raw in enumerate(documents)]
    resolved: dict = {}
    for key, fallback in _HISTORICAL.items():
        for index, document in enumerate(parsed):
            candidate = _candidate(document, index, key)
            if candidate is not None:
                resolved[key] = _resolve_candidate(key, candidate)
                break
        else:
            resolved[key] = fallback
    return {key: resolved[key] for key in _HISTORICAL}
