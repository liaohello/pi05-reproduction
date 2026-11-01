"""Fail-closed FAST tokenization for the Scene1 joint Flow/FAST model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
from transformers import AutoProcessor

PALIGEMMA_VOCAB_SIZE = 257_152
FAST_SKIP_TOKENS = 128
FAST_TOKEN_COUNT = 2_048
FAST_TOKEN_START = 254_976
FAST_TOKEN_STOP = 257_024
PAD_ID = 0
EOS_ID = 1


def action_id_to_paligemma(token_id: int) -> int:
    mapped = PALIGEMMA_VOCAB_SIZE - 1 - FAST_SKIP_TOKENS - int(token_id)
    if not FAST_TOKEN_START <= mapped < FAST_TOKEN_STOP:
        raise ValueError("FAST token mapping left the reserved action range")
    return mapped


def paligemma_to_action_id(token_id: int) -> int:
    mapped = PALIGEMMA_VOCAB_SIZE - 1 - FAST_SKIP_TOKENS - int(token_id)
    if not 0 <= mapped < FAST_TOKEN_COUNT:
        raise ValueError("PaliGemma token is outside the FAST action range")
    return mapped


@dataclass(frozen=True)
class FASTTokenized:
    tokens: np.ndarray
    token_mask: np.ndarray
    ar_mask: np.ndarray
    loss_mask: np.ndarray
    action_region_mask: np.ndarray
    fast_token_count: int


class Scene1FASTTokenizer:
    """Pinned FAST processor plus the PaliGemma sentencepiece model."""

    def __init__(
        self,
        fast_tokenizer_dir: str | Path,
        paligemma_tokenizer_path: str | Path,
        *,
        max_len: int = 128,
    ):
        self.max_len = int(max_len)
        if self.max_len < 2:
            raise ValueError("max_len must be at least two")
        self.fast = AutoProcessor.from_pretrained(
            str(fast_tokenizer_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.language = spm.SentencePieceProcessor(
            model_file=str(paligemma_tokenizer_path)
        )
        if self.fast.vocab_size != FAST_TOKEN_COUNT:
            raise ValueError(f"unexpected FAST vocabulary {self.fast.vocab_size}")
        if self.language.vocab_size() != PALIGEMMA_VOCAB_SIZE:
            raise ValueError(
                f"unexpected PaliGemma vocabulary {self.language.vocab_size()}"
            )

    @staticmethod
    def _validate_normalized(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(
                f"FAST actions must have shape (horizon, 12), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("FAST actions contain non-finite values")
        return actions

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (12,):
            raise ValueError(f"state must have shape (12,), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values")
        cleaned = " ".join(
            str(prompt).lower().strip().replace("_", " ").split())
        discretized = np.digitize(
            state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        prefix = self.language.encode(
            f"Task: {cleaned}, State: "
            + " ".join(map(str, discretized))
            + ";\n",
            add_bos=True,
        )
        if actions is None:
            postfix: list[int] = []
            fast_token_count = 0
        else:
            normalized = self._validate_normalized(actions)
            fast_ids = [
                int(value) for value in self.fast(normalized[None])[0]]
            if not fast_ids:
                raise ValueError("FAST produced no action tokens")
            mapped = [action_id_to_paligemma(value) for value in fast_ids]
            postfix = (
                self.language.encode("Action: ")
                + mapped
                + self.language.encode("|", add_eos=True)
            )
            fast_token_count = len(fast_ids)
            if postfix[-1] != EOS_ID:
                raise ValueError("FAST postfix does not end in EOS")

        tokens = prefix + postfix
        if len(tokens) > self.max_len:
            raise ValueError(
                f"token sequence {len(tokens)} exceeds max_len {self.max_len}; "
                "refusing to truncate the prompt, actions, or EOS")
        padding = self.max_len - len(tokens)
        token_mask = np.asarray([True] * len(tokens) + [False] * padding)
        ar_mask = np.asarray(
            [0] * len(prefix) + [1] * len(postfix) + [0] * padding,
            dtype=np.int32,
        )
        loss_mask = np.asarray(
            [False] * len(prefix) + [True] * len(postfix) + [False] * padding)
        if len(prefix) + len(postfix) != self.max_len - padding:
            raise AssertionError("FAST mask length is inconsistent")
        return (
            np.asarray(tokens + [PAD_ID] * padding, dtype=np.int32),
            token_mask,
            ar_mask,
            loss_mask,
        )

    def extract_actions(
        self,
        tokens: np.ndarray,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        if action_dim != 12 or action_horizon < 1:
            raise ValueError("FAST extraction requires 12 dimensions and positive horizon")
        mapped = [
            paligemma_to_action_id(int(token_id))
            for token_id in np.asarray(tokens).reshape(-1)
            if FAST_TOKEN_START <= int(token_id) < FAST_TOKEN_STOP
        ]
        if not mapped:
            raise ValueError("FAST token sequence contains no action tokens")
        decoded = self.fast.decode(
            [mapped], time_horizon=action_horizon, action_dim=action_dim)[0]
        if decoded.shape != (action_horizon, action_dim):
            raise ValueError(f"unexpected FAST decode shape {decoded.shape}")
        if not np.isfinite(decoded).all():
            raise ValueError("FAST decode returned non-finite actions")
        return np.asarray(decoded, dtype=np.float32)
