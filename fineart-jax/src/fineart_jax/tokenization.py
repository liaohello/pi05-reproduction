"""Text, recipe and FAST action tokenization for the FineART-VLA JAX migration.

The candidate owns every rule here: message layout, supervised spans, state
discretisation, ``<loc>`` conversion, recipe routing, FAST token assembly, the
code-token mask and tokenizer fitting. The only thing shared with the reference
is the *pinned base tokenizer objects* (PaliGemma and FAST), which are assets,
not logic. No ``lerobot`` import appears in this module.

Reference behaviour reproduced, deliberately including its edge cases:

* prompts are ``"{Role}: {content}\\n"`` per turn, and a supervised turn carries
  the tokenizer's EOS before the newline;
* labels are the input ids of every attended token whose characters overlap a
  supervised span, and ``-100`` elsewhere;
* only low-level streams receive the discretised state suffix;
* blend selection is a deterministic blake2b draw keyed on the sample index;
* FAST sequences are ``[BOS] + "Action: " + codes + "|"``, with the codes mapped
  as ``vocab - 1 - skip - token`` and only the code positions marked for CE.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
RESOLVER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")
EMITTED_AT_TOLERANCE_S = 0.1

DEFAULT_BINDINGS = {
    "subtask": "active_at(t, style=subtask)",
    "memory": "active_at(t, style=memory)",
    "plan": "active_at(t, style=plan)",
    "speech": "emitted_at(t, role=assistant, tool_name=say)",
    "interjection": "emitted_at(t, style=interjection)",
    "vqa": "emitted_at(t, style=vqa, role=assistant)",
    "vqa_query": "emitted_at(t, style=vqa, role=user)",
}

VALID_ROLES = ("user", "assistant", "system", "tool")
VALID_STREAMS = ("high_level", "low_level")
# Reference style partition: these names decide which column a resolver reads.
PERSISTENT_STYLES = ("subtask", "plan", "memory", "motion", "task_aug")
EVENT_ONLY_STYLES = ("interjection", "vqa", "trace")


# --------------------------------------------------------------------------- #
# Binding resolution
# --------------------------------------------------------------------------- #
def _timestamp(row: dict) -> float:
    return float(np.asarray(row["timestamp"]).reshape(-1)[0])


def _matching_rows(rows, *, style=None, role=None, tool_name=None, camera=None):
    out = []
    for row in rows:
        if style is not None and row.get("style") != style:
            continue
        if role is not None and row.get("role") != role:
            continue
        if tool_name is not None and not _row_has_tool_name(row, tool_name):
            continue
        if camera is not None and row.get("camera") != camera:
            continue
        out.append(row)
    return out


def _row_has_tool_name(row: dict, tool_name: str) -> bool:
    for call in row.get("tool_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("function"), dict):
            if call["function"].get("name") == tool_name:
                return True
    return False


def _select_one(rows, *, style, role, tool_name, camera):
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous resolver for style={style!r} role={role!r} tool_name={tool_name!r} "
            f"camera={camera!r}: {len(rows)} matching rows. Add a selector that distinguishes them."
        )
    return rows[0]


def active_at(t, *, persistent, style=None, role=None, tool_name=None, camera=None):
    _require_persistent("active_at", style)
    matches = [row for row in _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera)
               if _timestamp(row) <= t]
    if not matches:
        return None
    latest = max(_timestamp(row) for row in matches)
    return _select_one([row for row in matches if _timestamp(row) == latest],
                       style=style, role=role, tool_name=tool_name, camera=camera)


def emitted_at(t, *, persistent, events, style=None, role=None, tool_name=None, camera=None):
    if _column_for_style(style) == "persistent":
        matches = [row for row in _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera)
                   if abs(_timestamp(row) - t) <= EMITTED_AT_TOLERANCE_S]
    else:
        matches = _matching_rows(events, style=style, role=role, tool_name=tool_name, camera=camera)
    return _select_one(matches, style=style, role=role, tool_name=tool_name, camera=camera)


def _nth_relative(name, t, persistent, style, offset, role, tool_name, camera):
    _require_persistent(name, style)
    if abs(offset) < 1:
        raise ValueError(f"{name} offset must be non-zero.")
    rows = sorted(_matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera),
                  key=lambda row: (_timestamp(row), row.get("style") or "", row.get("role") or ""))
    if not rows:
        return None
    anchor = None
    for index, row in enumerate(rows):
        if _timestamp(row) <= t:
            anchor = index
        else:
            break
    target = (offset - 1 if offset > 0 else None) if anchor is None else anchor + offset
    if target is None or target < 0 or target >= len(rows):
        return None
    return rows[target]


def nth_prev(t, *, persistent, style=None, offset=1, role=None, tool_name=None, camera=None):
    return _nth_relative("nth_prev", t, persistent, style, -offset, role, tool_name, camera)


def nth_next(t, *, persistent, style=None, offset=1, role=None, tool_name=None, camera=None):
    return _nth_relative("nth_next", t, persistent, style, offset, role, tool_name, camera)


def _column_for_style(style) -> str:
    """Which column a style is stored in; ``None`` is an event row."""
    if style is None:
        return "events"
    if style in PERSISTENT_STYLES:
        return "persistent"
    if style in EVENT_ONLY_STYLES:
        return "events"
    raise ValueError(f"Unknown language style: {style!r}")


def _require_persistent(name, style):
    if style is None:
        raise ValueError(f"{name} requires a persistent style.")
    if _column_for_style(style) != "persistent":
        raise ValueError(f"{name} cannot be used with event-only style {style!r}.")


def _parse_resolver_args(args: str) -> dict:
    kwargs: dict[str, Any] = {}
    if not args.strip():
        return kwargs
    for part in (item.strip() for item in args.split(",")):
        if not part:
            continue
        if part == "t":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid resolver argument: {part!r}")
        key, value = (item.strip() for item in part.split("=", 1))
        kwargs[key] = int(value) if key == "offset" else value.strip("\"'")
    return kwargs


def resolve_spec(spec: str, *, persistent, events, t):
    match = RESOLVER_RE.match(spec.strip())
    if match is None:
        raise ValueError(f"Invalid resolver expression: {spec!r}")
    name = match.group("name")
    kwargs = _parse_resolver_args(match.group("args"))
    if name == "emitted_at":
        return emitted_at(t, persistent=persistent, events=events, **kwargs)
    if name == "active_at":
        return active_at(t, persistent=persistent, **kwargs)
    if name == "nth_prev":
        return nth_prev(t, persistent=persistent, **kwargs)
    if name == "nth_next":
        return nth_next(t, persistent=persistent, **kwargs)
    raise ValueError(f"Unknown language resolver: {name!r}")


# --------------------------------------------------------------------------- #
# Recipe rendering
# --------------------------------------------------------------------------- #
def referenced_bindings(recipe: dict) -> set:
    names: set = set()
    components = [recipe] if recipe.get("messages") is not None else list((recipe.get("blend") or {}).values())
    for component in components:
        for turn in component.get("messages") or []:
            if turn.get("if_present"):
                names.add(turn["if_present"])
            if turn.get("tool_calls_from"):
                names.add(turn["tool_calls_from"])
            names |= _placeholders(turn.get("content"))
    return names


def _placeholders(content) -> set:
    if content is None:
        return set()
    if isinstance(content, str):
        return set(PLACEHOLDER_RE.findall(content))
    names = set()
    for block in content:
        for value in block.values():
            if isinstance(value, str):
                names |= set(PLACEHOLDER_RE.findall(value))
    return names


def substitute(template: str, bindings: dict) -> str:
    def replace(match):
        name = match.group(1)
        if name not in bindings:
            raise ValueError(f"Unknown template binding: {name!r}")
        value = bindings[name]
        if value is None:
            return ""
        if isinstance(value, dict):
            content = value.get("content")
            return "" if content is None else str(content)
        return str(value)

    return PLACEHOLDER_RE.sub(replace, template)


def render_message_turns(turns: Sequence[dict], bindings: dict) -> dict:
    """Expand recipe turns into messages, streams and supervised indices."""
    messages: list[dict] = []
    streams: list[str] = []
    target_indices: list[int] = []
    for turn in turns:
        if turn.get("if_present") and bindings.get(turn["if_present"]) is None:
            continue
        message = {"role": turn["role"]}
        if turn.get("content") is not None:
            content = turn["content"]
            message["content"] = substitute(content, bindings) if isinstance(content, str) else [
                {key: substitute(value, bindings) if isinstance(value, str) else value for key, value in block.items()}
                for block in content
            ]
        index = len(messages)
        messages.append(message)
        streams.append(turn["stream"])
        if turn.get("target"):
            target_indices.append(index)
    return {
        "messages_rendered": messages,
        "message_streams": streams,
        "target_message_indices": target_indices,
    }


def _blake2b_draw(key: str, total: float) -> float:
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64 * total


def select_recipe(recipe: dict, sample_idx: int) -> dict | None:
    """Deterministic weighted pick among non-VQA blend components."""
    blend = recipe.get("blend")
    if blend is None:
        return recipe
    components = [component for component in blend.values() if component.get("route") is None]
    total = sum(component.get("weight") or 0.0 for component in components)
    if total <= 0:
        return None
    draw = _blake2b_draw(str(sample_idx), total)
    cumulative = 0.0
    last = None
    for component in components:
        last = component
        cumulative += component.get("weight") or 0.0
        if draw < cumulative:
            return component
    return last


def render_vqa_if_present(recipe: dict, *, persistent, events, t, sample_idx, task, default_task):
    """Sparse VQA frames go to a view-specific component before weighted selection."""
    blend = recipe.get("blend")
    if blend is None:
        return None
    renderable = []
    for component in blend.values():
        if component.get("route") != "vqa":
            continue
        bindings = resolve_bindings(component, persistent=persistent, events=events, t=t,
                                    sample_idx=sample_idx, task=task, default_task=default_task)
        rendered = _render_message_recipe(component, bindings)
        if rendered is not None:
            weight = component.get("weight")
            if weight is None:
                raise ValueError("Routed VQA blend components must define a weight.")
            renderable.append((weight, rendered))
    if not renderable:
        return None
    if len(renderable) == 1:
        return renderable[0][1]
    total = sum(weight for weight, _ in renderable)
    draw = _blake2b_draw(f"vqa:{sample_idx}", total)
    cumulative = 0.0
    for weight, rendered in renderable:
        cumulative += weight
        if draw < cumulative:
            return rendered
    return renderable[-1][1]


def _render_message_recipe(recipe: dict, bindings: dict) -> dict | None:
    if recipe.get("messages") is None:
        raise ValueError("Cannot render a blend recipe as a message recipe.")
    rendered = render_message_turns(recipe["messages"], bindings)
    has_low_level = any(stream == "low_level" for stream in rendered["message_streams"])
    if not rendered["target_message_indices"] and not has_low_level:
        return None
    _validate_rendered(rendered)
    return rendered


def _validate_rendered(rendered: dict) -> None:
    if len(rendered["message_streams"]) != len(rendered["messages_rendered"]):
        raise ValueError("message_streams must be aligned with messages.")
    if not rendered["target_message_indices"] and not any(
        stream == "low_level" for stream in rendered["message_streams"]
    ):
        raise ValueError("Rendered samples must contain a target message or a low_level-stream message.")
    for index in rendered["target_message_indices"]:
        if index < 0 or index >= len(rendered["messages_rendered"]):
            raise ValueError(f"Target message index {index} is out of bounds.")


def resolve_bindings(recipe: dict, *, persistent, events, t, sample_idx, task, default_task):
    bindings = {"task": resolve_task(task, persistent=persistent, sample_idx=sample_idx, default_task=default_task)}
    declared = recipe.get("bindings") or {}
    specs = {**DEFAULT_BINDINGS, **declared}
    needed = referenced_bindings(recipe) | set(declared)
    for name, spec in specs.items():
        if name not in needed:
            continue
        bindings[name] = resolve_spec(spec, persistent=persistent, events=events, t=t)
    return bindings


def resolve_task(task, *, persistent, sample_idx, default_task):
    """Explicit task wins; otherwise rotate through ``task_aug`` rows by index."""
    if task is not None:
        return task
    augmented = [row for row in persistent if row.get("style") == "task_aug" and row.get("role") == "user"]
    if augmented:
        digest = hashlib.blake2b(f"task_aug:{sample_idx}".encode(), digest_size=8).digest()
        chosen = augmented[int.from_bytes(digest, "big") % len(augmented)].get("content")
        if chosen:
            return str(chosen)
    return default_task


def render_sample(*, recipe, persistent, events, t, sample_idx, task=None, default_task=None):
    """Render one sample, or ``None`` when it cannot be supervised."""
    routed = render_vqa_if_present(recipe, persistent=list(persistent or []), events=list(events or []), t=t,
                                   sample_idx=sample_idx, task=task, default_task=default_task)
    if routed is not None:
        return routed
    selected = select_recipe(recipe, sample_idx)
    if selected is None:
        return None
    bindings = resolve_bindings(selected, persistent=list(persistent or []), events=list(events or []), t=t,
                                sample_idx=sample_idx, task=task, default_task=default_task)
    return _render_message_recipe(selected, bindings)


def fallback_low_level_render(task):
    """Keep action-only samples trainable when no recipe branch matches."""
    if not isinstance(task, str) or not task:
        return None
    return {
        "messages_rendered": [{"role": "user", "content": task}],
        "message_streams": ["low_level"],
        "target_message_indices": [],
    }


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def format_messages(messages: Sequence[dict], target_indices, eos_token: str | None):
    """Build the flat prompt and each message's supervised character span."""
    targets = set(target_indices or [])
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        content = message.get("content", "") or ""
        header = f"{role.capitalize()}: "
        body = content + eos_token if (eos_token and index in targets) else content
        full = header + body + "\n"
        start = cursor + len(header)
        spans.append((start, start + len(body)))
        parts.append(full)
        cursor += len(full)
    return "".join(parts), spans


def discretize_state_str(state_row) -> str:
    """Format one normalised state row with the 256-bin PI0.5 convention."""
    array = np.asarray(state_row, dtype=np.float32).reshape(-1)
    edges = np.linspace(-1, 1, 256 + 1)[:-1]
    disc = np.digitize(array, bins=edges) - 1
    return " ".join(str(int(value)) for value in disc)


def loc_token(coord: float) -> str:
    """``<locNNNN>`` for a stored annotation in [0, 1] image fractions."""
    index = round(float(coord) * 1023)
    return f"<loc{max(0, min(1023, index)):04d}>"


def vqa_answer_to_loc(answer: dict) -> str | None:
    """Label-first PaliGemma locations for a normalised point or bbox answer."""
    point = answer.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 2 and "point_format" in answer:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        label = str(answer.get("label", "")).strip()
        if not label:
            return None
        return f"{label} {loc_token(y)}{loc_token(x)}"

    detections = answer.get("detections")
    if isinstance(detections, list) and detections:
        parts = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            box = detection.get("bbox")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError):
                continue
            label = str(detection.get("label", "")).strip()
            if not label:
                continue
            parts.append(f"{label} {loc_token(y1)}{loc_token(x1)}{loc_token(y2)}{loc_token(x2)}")
        return " ; ".join(parts) if parts else None
    return None


def messages_vqa_to_loc(messages: Sequence[dict], target_indices) -> list[dict]:
    if not target_indices:
        return list(messages)
    out = list(messages)
    for index in target_indices:
        if not (0 <= index < len(out)):
            continue
        content = out[index].get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            answer = json.loads(content)
        except (ValueError, TypeError):
            continue
        if not isinstance(answer, dict):
            continue
        text = vqa_answer_to_loc(answer)
        if text is not None:
            out[index] = {**out[index], "content": text}
    return out


def strip_blocks(message: dict) -> dict:
    """Flatten text blocks, drop stream/target tags and ignore image blocks."""
    new = dict(message)
    new.pop("stream", None)
    new.pop("target", None)
    content = new.get("content")
    if content is None:
        new["content"] = ""
    elif isinstance(content, str):
        pass
    elif isinstance(content, list):
        parts = [block.get("text", "") for block in content
                 if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)]
        new["content"] = "\n".join(parts)
    else:
        new["content"] = str(content)
    return new


def flatten_say_tool_calls(message: dict) -> dict:
    """Move ``say`` tool calls into text markers PaliGemma can learn."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    say_texts = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if function.get("name") != "say":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                arguments = {}
        text = arguments.get("text", "") if isinstance(arguments, dict) else ""
        if text:
            say_texts.append(str(text))
    new = dict(message)
    new.pop("tool_calls", None)
    if not say_texts:
        return new
    base = str(new.get("content") or "").strip()
    marker = "".join(f"<say>{text}</say>" for text in say_texts)
    new["content"] = f"{base}\n{marker}" if base else marker
    return new


# --------------------------------------------------------------------------- #
# Token-level encoding
# --------------------------------------------------------------------------- #
def encode_prompt(tokenizer, prompt: str, max_length: int, padding_side: str = "right"):
    """Tokenize with offsets, padding to ``max_length`` and right padding."""
    encoded = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        padding_side=padding_side,
    )
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    mask = np.asarray(encoded["attention_mask"], dtype=bool)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    if ids.ndim == 2:
        ids, mask, offsets = ids[0], mask[0], offsets[0]
    return ids, mask, offsets


def text_labels(input_ids: np.ndarray, attention_mask: np.ndarray, offsets: np.ndarray, spans, target_indices) -> np.ndarray:
    """Supervise attended tokens that overlap a target span; ``-100`` elsewhere."""
    labels = np.full_like(input_ids, -100)
    targets = [spans[index] for index in (target_indices or []) if index < len(spans)]
    if not targets:
        return labels
    bounds = np.asarray(targets, dtype=np.int64)
    overlaps = ((offsets[None, :, 1] > bounds[:, None, 0]) & (offsets[None, :, 0] < bounds[:, None, 1])).any(axis=0)
    supervise = attention_mask & overlaps
    labels = np.where(supervise, input_ids, labels)
    return labels


def encode_messages(tokenizer, messages, message_streams, target_indices, *, state_row=None,
                    max_length=200, runtime_tail: bool | None = None):
    """Full text path: VQA rewrite, state suffix, prompt, labels, routing flag."""
    messages = messages_vqa_to_loc(messages, target_indices)
    messages = [strip_blocks(flatten_say_tool_calls(message)) for message in messages]

    if state_row is not None and any(stream == "low_level" for stream in message_streams):
        state_str = discretize_state_str(state_row)
        for message in reversed(messages):
            if message.get("role") == "user":
                base = message.get("content", "") or ""
                message["content"] = f"{base}, State: {state_str};"
                break

    prompt, spans = format_messages(messages, target_indices, getattr(tokenizer, "eos_token", None))
    if runtime_tail is None:
        runtime_tail = not message_streams and not target_indices
    if runtime_tail:
        prompt += "Assistant:"

    ids, mask, offsets = encode_prompt(tokenizer, prompt, max_length)
    labels = text_labels(ids, mask, offsets, spans, target_indices)
    predict_actions = bool(any(stream == "low_level" for stream in message_streams))
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "predict_actions": predict_actions,
        "prompt": prompt,
        "spans": spans,
        "offsets": offsets,
    }


# --------------------------------------------------------------------------- #
# FAST action tokens
# --------------------------------------------------------------------------- #
@dataclass
class FastTokens:
    tokens: np.ndarray
    mask: np.ndarray
    code_mask: np.ndarray
    truncated: bool = False


def fast_action_tokens(action, action_tokenizer, paligemma_tokenizer, *, max_action_tokens: int = 256,
                       fast_skip_tokens: int = 1152, allow_truncation: bool = True) -> FastTokens:
    """Tokenize one action chunk the way the reference processor does.

    The sequence is ``[BOS] + encode("Action: ") + codes + encode("|")``; the codes
    are remapped into PaliGemma's reserved tail and only they are marked for CE.
    """
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"action must be [horizon, dim] or [batch, horizon, dim], got {array.shape}")

    codes = np.asarray(action_tokenizer(array[None, ...]), dtype=np.int64).reshape(-1)
    vocab_size = int(paligemma_tokenizer.vocab_size)
    mapped = vocab_size - 1 - int(fast_skip_tokens) - codes
    bos_id = int(paligemma_tokenizer.bos_token_id)
    prompt_ids = np.asarray(paligemma_tokenizer.encode("Action: ", add_special_tokens=False), dtype=np.int64)
    end_ids = np.asarray(paligemma_tokenizer.encode("|"), dtype=np.int64)

    tokens = np.concatenate([np.asarray([bos_id], dtype=np.int64), prompt_ids, mapped, end_ids])
    code_start = 1 + len(prompt_ids)
    code_end = code_start + len(mapped)
    code_mask = np.zeros(len(tokens), dtype=bool)
    code_mask[code_start:code_end] = True

    truncated = False
    if len(tokens) > max_action_tokens:
        if not allow_truncation:
            raise ValueError(
                f"FAST action sequence has {len(tokens)} tokens, exceeding "
                f"max_action_tokens={max_action_tokens}."
            )
        truncated = True
        tokens = tokens[:max_action_tokens]
        code_mask = code_mask[:max_action_tokens]
        mask = np.ones(max_action_tokens, dtype=bool)
    else:
        pad = max_action_tokens - len(tokens)
        mask = np.concatenate([np.ones(len(tokens), dtype=bool), np.zeros(pad, dtype=bool)])
        code_mask = np.concatenate([code_mask, np.zeros(pad, dtype=bool)])
        tokens = np.concatenate([tokens, np.zeros(pad, dtype=np.int64)])
    return FastTokens(tokens=tokens, mask=mask, code_mask=code_mask, truncated=truncated)


# --------------------------------------------------------------------------- #
# FAST tokenizer fitting
# --------------------------------------------------------------------------- #
def dataset_signature(*, dataset_repo_id, base_tokenizer_name, n_samples, chunk_size, normalization_mode,
                      dataset_revision=None, episodes=None, exclude_episodes=None, action_stats=None,
                      use_relative_actions=False, relative_action_mask=None, validation_samples=256,
                      max_reconstruction_rmse=0.10, max_dim_rmse=0.20) -> str:
    """Hash every input that changes the fitted action distribution."""
    payload = {
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": dataset_revision,
        "base_tokenizer_name": base_tokenizer_name,
        "n_samples": n_samples,
        "chunk_size": chunk_size,
        "normalization_mode": normalization_mode,
        "episodes": episodes,
        "exclude_episodes": exclude_episodes,
        "action_stats": _jsonable(action_stats),
        "use_relative_actions": use_relative_actions,
        "relative_action_mask": relative_action_mask,
        "validation_samples": validation_samples,
        "max_reconstruction_rmse": max_reconstruction_rmse,
        "max_dim_rmse": max_dim_rmse,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def select_episode_indices(available, episodes, exclude_episodes):
    allowed = set(episodes) if episodes is not None else set(available)
    excluded = set(exclude_episodes or [])
    return [episode for episode in available if episode in allowed and episode not in excluded]


def sample_action_chunks(*, actions, episode_index, n_samples, validation_samples, chunk_size, seed=42,
                         episodes=None, exclude_episodes=None):
    """Collect chunks exactly as the reference does, including its RNG order."""
    actions = np.asarray(actions, dtype=np.float32)
    episode_index = np.asarray(episode_index, dtype=np.int64)
    order = np.argsort(episode_index, kind="stable")
    eps_sorted = episode_index[order]
    boundaries = np.asarray(np.searchsorted(eps_sorted, np.arange(int(eps_sorted.max()) + 2)))
    ep_to_slice = {
        int(ep): (int(boundaries[ep]), int(boundaries[ep + 1]))
        for ep in range(len(boundaries) - 1)
        if boundaries[ep] < boundaries[ep + 1]
    }
    actions = actions[order]

    selected = select_episode_indices(list(ep_to_slice), episodes, exclude_episodes)
    if not selected:
        raise RuntimeError("FAST fit: episode selection is empty after applying exclusions.")

    total = n_samples + validation_samples
    per_episode = max(1, (total + len(selected) - 1) // len(selected))
    rng = np.random.default_rng(seed)
    chunks = []
    rows = []
    for ep in rng.permutation(selected):
        if len(chunks) >= total:
            break
        start, stop = ep_to_slice[int(ep)]
        episode_actions = actions[start:stop]
        if episode_actions.shape[0] < chunk_size:
            continue
        starts = rng.integers(0, episode_actions.shape[0] - chunk_size + 1, size=per_episode)
        for offset in starts:
            chunks.append(episode_actions[int(offset): int(offset) + chunk_size])
            rows.append(start + int(offset))
            if len(chunks) >= total:
                break
    if not chunks:
        raise RuntimeError("FAST fit collected zero action chunks.")
    return np.stack(chunks, axis=0), np.asarray(rows, dtype=np.int64)


def reconstruction_report(tokenizer, actions, max_reconstruction_rmse=0.10, max_dim_rmse=0.20):
    """Decode held-out chunks and reject a tokenizer with excessive error."""
    decoded = np.asarray(tokenizer.decode(tokenizer(actions)), dtype=np.float32)
    if decoded.shape != actions.shape:
        raise RuntimeError(
            f"FAST tokenizer reconstruction shape mismatch: expected {actions.shape}, got {decoded.shape}."
        )
    if not np.isfinite(decoded).all():
        raise RuntimeError("FAST tokenizer reconstruction contains non-finite values.")
    squared = np.square(decoded - actions)
    rmse = float(np.sqrt(squared.mean()))
    dim_rmse = np.sqrt(squared.mean(axis=(0, 1)))
    nonconstant = np.ptp(actions, axis=(0, 1)) > 1e-8
    worst = float(dim_rmse[nonconstant].max(initial=0.0))
    report = {
        "num_validation_chunks": int(actions.shape[0]),
        "reconstruction_rmse": rmse,
        "max_dim_rmse": worst,
        "dim_rmse": dim_rmse.tolist(),
        "max_reconstruction_rmse": max_reconstruction_rmse,
        "max_allowed_dim_rmse": max_dim_rmse,
    }
    if rmse > max_reconstruction_rmse or worst > max_dim_rmse:
        raise RuntimeError(
            "FAST tokenizer reconstruction error exceeds the configured limit: "
            f"rmse={rmse:.4f} (max {max_reconstruction_rmse:.4f}), "
            f"max_dim_rmse={worst:.4f} (max {max_dim_rmse:.4f})."
        )
    return report, decoded
