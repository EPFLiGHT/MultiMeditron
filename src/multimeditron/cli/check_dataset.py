import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import click

from multimeditron.cli import EPILOG, main_cli
from multimeditron.model.constants import CONVERSATIONS_KEY, MODALITIES_KEY, TEXT_KEY

from datasets import disable_caching
disable_caching()

logger = logging.getLogger(__name__)

DEFAULT_ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"


def _is_dataset_folder(path: str) -> bool:
    try:
        from datasets import config as datasets_config
    except Exception:
        return False
    return os.path.exists(os.path.join(path, datasets_config.DATASET_INFO_FILENAME)) and \
        os.path.exists(os.path.join(path, datasets_config.DATASET_STATE_JSON_FILENAME))


def _load_hf_dataset(path: str) -> Tuple[Any, str]:
    from datasets import load_dataset, load_from_disk

    if _is_dataset_folder(path):
        dataset = load_from_disk(path)
    else:
        _, ext = os.path.splitext(path)
        if ext == ".parquet":
            dataset = load_dataset("parquet", data_files=path)
        elif ext == ".arrow":
            dataset = load_dataset("arrow", data_files=path)
        elif ext == ".jsonl":
            dataset = load_dataset("json", data_files=path)
        else:
            dataset = load_dataset(path)

    if isinstance(dataset, dict):
        dataset = dataset.get("train") or next(iter(dataset.values()))

    return dataset, path


def _count_attachment_tokens(text: str, attachment_token: str) -> int:
    return text.count(attachment_token)


def _count_tokens_in_conversations(conversations: List[Dict[str, Any]], attachment_token: str,
                                   errors: List[str], sample_id: str) -> int:
    count = 0
    for msg_idx, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            errors.append(f"{sample_id}: conversations[{msg_idx}] must be an object with 'role' and 'content'.")
            continue
        if "role" not in msg or "content" not in msg:
            errors.append(f"{sample_id}: conversations[{msg_idx}] must include 'role' and 'content'.")
            continue
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            errors.append(f"{sample_id}: conversations[{msg_idx}] 'role' and 'content' must be strings.")
            continue
        count += _count_attachment_tokens(msg["content"], attachment_token)
    return count


def _value_is_bytes(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def _validate_image_value(value: Any, base_path: str, errors: List[str], sample_id: str, mod_idx: int) -> None:
    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None and isinstance(value, Image.Image):
        return
    if _value_is_bytes(value):
        return
    if isinstance(value, dict):
        if "bytes" in value and _value_is_bytes(value["bytes"]):
            return
        if "path" in value and isinstance(value["path"], str):
            candidate = value["path"]
            if os.path.isabs(candidate):
                exists = os.path.exists(candidate)
            else:
                exists = os.path.exists(os.path.join(base_path, candidate))
            if not exists:
                errors.append(f"{sample_id}: modalities[{mod_idx}].value path not found: {candidate}")
            return
        errors.append(f"{sample_id}: modalities[{mod_idx}].value must include 'bytes' or 'path' for image modality.")
        return
    if isinstance(value, str):
        candidate = value
        if os.path.isabs(candidate):
            exists = os.path.exists(candidate)
        else:
            exists = os.path.exists(os.path.join(base_path, candidate))
        if not exists:
            errors.append(f"{sample_id}: modalities[{mod_idx}].value path not found: {candidate}")
        return
    errors.append(f"{sample_id}: modalities[{mod_idx}].value has unsupported type {type(value).__name__}.")


def _validate_sample(sample: Dict[str, Any], sample_id: str, modality_type: str,
                     attachment_token: str, base_path: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(sample, dict):
        if not (hasattr(sample, "keys") and hasattr(sample, "__getitem__")):
            errors.append(f"{sample_id}: sample must be a JSON object.")
            return errors

    has_text = TEXT_KEY in sample
    has_conversations = CONVERSATIONS_KEY in sample
    if has_text == has_conversations:
        errors.append(f"{sample_id}: sample must include exactly one of '{TEXT_KEY}' or '{CONVERSATIONS_KEY}'.")
        return errors

    attachment_count = 0
    if has_text:
        if not isinstance(sample[TEXT_KEY], str):
            errors.append(f"{sample_id}: '{TEXT_KEY}' must be a string.")
        else:
            attachment_count = _count_attachment_tokens(sample[TEXT_KEY], attachment_token)
    else:
        if not isinstance(sample[CONVERSATIONS_KEY], list):
            errors.append(f"{sample_id}: '{CONVERSATIONS_KEY}' must be a list of messages.")
        else:
            attachment_count = _count_tokens_in_conversations(
                sample[CONVERSATIONS_KEY], attachment_token, errors, sample_id
            )

    if MODALITIES_KEY not in sample:
        errors.append(f"{sample_id}: missing '{MODALITIES_KEY}' field required for multimodal samples.")
        return errors

    modalities = sample[MODALITIES_KEY]
    if not isinstance(modalities, list):
        errors.append(f"{sample_id}: '{MODALITIES_KEY}' must be a list.")
        return errors
    if len(modalities) == 0:
        errors.append(f"{sample_id}: '{MODALITIES_KEY}' must contain at least one modality.")
        return errors

    if attachment_count != len(modalities):
        errors.append(
            f"{sample_id}: attachment token count ({attachment_count}) does not match number of modalities ({len(modalities)})."
        )

    found_modality = False
    for mod_idx, modality in enumerate(modalities):
        if not isinstance(modality, dict):
            errors.append(f"{sample_id}: modalities[{mod_idx}] must be an object.")
            continue
        if "type" not in modality or "value" not in modality:
            errors.append(f"{sample_id}: modalities[{mod_idx}] must include 'type' and 'value'.")
            continue

        if modality["type"] == modality_type:
            found_modality = True
            _validate_image_value(modality["value"], base_path, errors, sample_id, mod_idx)

    if not found_modality:
        errors.append(f"{sample_id}: no '{modality_type}' modality found in '{MODALITIES_KEY}'.")
    return errors


def _validate_row(sample: Dict[str, Any], modality_type: str, attachment_token: str, base_path: str) -> Dict[str, Any]:
    sample_id = f"sample {sample.get('__index__', 'unknown')}"
    errors = _validate_sample(sample, sample_id, modality_type, attachment_token, base_path)
    return {"__errors__": errors}


@main_cli.command(epilog=EPILOG)
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--modality", "-m", "modality_type", type=click.Choice(["image"], case_sensitive=False),
              required=True, help="Modality type to validate (currently only 'image').")
@click.option("--attachment-token", default=DEFAULT_ATTACHMENT_TOKEN, show_default=True,
              help="Token used in text/conversations to indicate modality placement.")
@click.option("--max-samples", type=int, default=None,
              help="Maximum number of samples to validate (default: all).")
@click.option("--num-proc", type=int, default=None,
              help="Number of processes to use for validation (default: CPU count).")
def check_dataset(dataset_path: str, modality_type: str, attachment_token: str,
                  max_samples: Optional[int], num_proc: Optional[int]):
    """
    Validate dataset format for a given modality.
    """
    modality_type = modality_type.lower()
    base_path = dataset_path
    if os.path.isfile(dataset_path):
        base_path = os.path.dirname(dataset_path) or "."

    try:
        dataset, source_desc = _load_hf_dataset(dataset_path)
    except Exception as exc:
        click.echo(f"Failed to load dataset: {exc}")
        raise SystemExit(1)

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    if num_proc is None or num_proc < 1:
        num_proc = os.cpu_count() or 1

    dataset = dataset.add_column("__index__", list(range(1, len(dataset) + 1)))
    dataset = dataset.map(
        _validate_row,
        num_proc=num_proc,
        fn_kwargs={
            "modality_type": modality_type,
            "attachment_token": attachment_token,
            "base_path": base_path,
        },
        desc=f"Validating dataset {source_desc}",
    )

    error_ds = dataset.filter(lambda x: len(x["__errors__"]) > 0, num_proc=num_proc)

    if len(error_ds) > 0:
        click.echo(click.style("❌ Dataset format check failed:", fg="red", bold=True))
        row = error_ds[0]
        for err in row["__errors__"]:
            click.echo(click.style(f"- {err}", fg="red", bold=True))
        raise SystemExit(1)

    click.echo(click.style("✅ Dataset format check passed.", fg="green", bold=True))
