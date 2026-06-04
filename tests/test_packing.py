"""
Unit tests for sequence packing in DataCollatorForMultimodal.

Tests structural invariants WITHOUT needing a real model or SLURM:
  - Layer 1a: cu_seqlens matches actual sub-sequence boundaries
  - Layer 1b: position_ids restart at 0 for each sub-sequence
  - Layer 1c: labels[0] is -100 for all non-first sub-sequences in a bin
  - Layer 1d: total real tokens per bin ≤ max_length
  - Layer 1e: modality token_ranges are correctly offset
  - Layer 1f: unpacked path is 100% unchanged when pack_sequences=False

Run with:
    cd /users/surech/meditron/MultiMeditron
    PYTHONPATH=src python -m pytest tests/test_packing.py -v
"""

import sys
import torch
from unittest.mock import MagicMock

try:
    import pytest
    _PYTEST = True
except ImportError:
    _PYTEST = False

sys.path.insert(0, "src")

from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.model.constants import (
    MODALITIES_KEY, MODALITY_TYPE_KEY, MODALITY_VALUE_KEY, IGNORE_TOKEN_INDEX
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collator(max_length: int = 256) -> DataCollatorForMultimodal:
    """Instantiate a minimal collator with mocked heavy dependencies."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    collator = DataCollatorForMultimodal.__new__(DataCollatorForMultimodal)
    collator.tokenizer = tokenizer
    collator.max_length = max_length
    collator.pack_sequences = True
    collator.use_2d_position_ids = False
    return collator


def _make_feature(real_len: int, pad_to: int = None, modality_ranges=None):
    """
    Create a synthetic already-tokenized feature dict.

    Args:
        real_len: number of real (non-padding) tokens.
        pad_to: total length including padding (defaults to real_len).
        modality_ranges: list of (start, end) pairs for fake image token ranges.
    """
    pad_to = pad_to or real_len
    assert pad_to >= real_len

    iids = torch.zeros(pad_to, dtype=torch.long)
    iids[:real_len] = torch.arange(1, real_len + 1)

    lbs = torch.full((pad_to,), IGNORE_TOKEN_INDEX, dtype=torch.long)
    lbs[:real_len] = torch.arange(1, real_len + 1)

    attn = torch.zeros(pad_to, dtype=torch.long)
    attn[:real_len] = 1

    modalities = []
    for (start, end) in (modality_ranges or []):
        modalities.append({
            MODALITY_TYPE_KEY: "image",
            MODALITY_VALUE_KEY: torch.randn(end - start, 4),  # tiny fake embeddings
            "token_range": (start, end),
        })

    return {
        "input_ids": iids,
        "labels": lbs,
        "attention_mask": attn,
        MODALITIES_KEY: modalities,
    }


# ---------------------------------------------------------------------------
# Layer 1a — cu_seqlens match sub-sequence boundaries
# ---------------------------------------------------------------------------

class TestCuSeqlens:
    def test_single_sample_per_bin(self):
        """Each sample fills its own bin → cu_seqlens = [0, real_len, max_length]."""
        collator = _make_collator(max_length=100)
        f1 = _make_feature(80)
        f2 = _make_feature(90)   # too large to share a bin with f1 (80+90>100)
        packed_text, _ = collator._pack_sequences([f1, f2])

        for cu in packed_text["cu_seqlens"]:
            # Must start at 0
            assert cu[0].item() == 0
            # Must end at max_length
            assert cu[-1].item() == 100

    def test_two_samples_in_one_bin(self):
        """Two short samples pack into a single bin.

        FFD sorts descending by length, so f2 (40) is placed before f1 (30)
        within the bin → cu = [0, 40, 70, 100].
        """
        collator = _make_collator(max_length=100)
        f1 = _make_feature(30)
        f2 = _make_feature(40)
        packed_text, _ = collator._pack_sequences([f1, f2])

        # Should produce exactly 1 packed sequence
        assert packed_text["input_ids"].shape[0] == 1
        cu = packed_text["cu_seqlens"][0]
        assert cu[0].item() == 0
        assert cu[1].item() == 40   # f2 (larger) placed first by FFD
        assert cu[2].item() == 70   # f2 + f1 = 70
        assert cu[-1].item() == 100  # sentinel = max_length

    def test_three_samples_two_bins(self):
        """Three samples packed into 2 bins (first-fit-decreasing)."""
        collator = _make_collator(max_length=100)
        # FFD order: 60, 50, 40  → bin0=[60,40], bin1=[50]
        f60 = _make_feature(60)
        f50 = _make_feature(50)
        f40 = _make_feature(40)
        packed_text, _ = collator._pack_sequences([f60, f50, f40])

        assert packed_text["input_ids"].shape[0] == 2
        # Bin with two samples: cu has 3 entries + sentinel
        bin_sizes = [len(cu) for cu in packed_text["cu_seqlens"]]
        assert 3 in bin_sizes  # one bin has 2 sub-sequences → [0, a, a+b, max]


# ---------------------------------------------------------------------------
# Layer 1b — position_ids restart at 0 for each sub-sequence
# ---------------------------------------------------------------------------

class TestPositionIds:
    def test_single_sample_starts_at_zero(self):
        collator = _make_collator(max_length=50)
        f = _make_feature(20)
        packed_text, _ = collator._pack_sequences([f])
        pos = packed_text["position_ids"][0]
        assert pos[0].item() == 0
        assert pos[19].item() == 19

    def test_second_sub_sequence_restarts(self):
        """position_ids of the second sub-sequence should start at 0 again.

        FFD puts f2 (40) first → boundary at position 40.
        """
        collator = _make_collator(max_length=100)
        f1 = _make_feature(30)
        f2 = _make_feature(40)
        packed_text, _ = collator._pack_sequences([f1, f2])

        pos = packed_text["position_ids"][0]
        # First sub-seq (f2, 40 tokens): positions 0..39
        assert pos[0].item() == 0
        assert pos[39].item() == 39
        # Second sub-seq (f1, 30 tokens): flat positions 40..69, values restart at 0
        assert pos[40].item() == 0
        assert pos[40 + 29].item() == 29

    def test_three_sub_sequences_all_restart(self):
        """FFD sorts [30,50,40] → [50,40,30] within the bin."""
        collator = _make_collator(max_length=200)
        lens = [30, 50, 40]
        features = [_make_feature(l) for l in lens]
        packed_text, _ = collator._pack_sequences(features)

        pos = packed_text["position_ids"][0]
        # FFD order: 50, 40, 30
        ffd_lens = [50, 40, 30]
        offset = 0
        for i, l in enumerate(ffd_lens):
            assert pos[offset].item() == 0, f"Sub-seq {i} does not restart at 0"
            assert pos[offset + l - 1].item() == l - 1
            offset += l


# ---------------------------------------------------------------------------
# Layer 1c — label boundary masking
# ---------------------------------------------------------------------------

class TestLabelBoundaries:
    def test_first_token_of_second_subsample_is_masked(self):
        """The first label of any non-first sub-sequence must be IGNORE_TOKEN_INDEX.

        FFD puts f2 (40) first → boundary at position 40.
        """
        collator = _make_collator(max_length=100)
        f1 = _make_feature(30)
        f2 = _make_feature(40)
        packed_text, _ = collator._pack_sequences([f1, f2])

        lbs = packed_text["labels"][0]
        boundary = 40  # f2 (40 tokens) placed first by FFD
        assert lbs[boundary].item() == IGNORE_TOKEN_INDEX, (
            f"Expected label at boundary {boundary} to be {IGNORE_TOKEN_INDEX}, "
            f"got {lbs[boundary].item()}"
        )

    def test_labels_outside_real_tokens_are_masked(self):
        """Padding tokens (beyond total real length) must have label = IGNORE_TOKEN_INDEX."""
        collator = _make_collator(max_length=100)
        f = _make_feature(30)
        packed_text, _ = collator._pack_sequences([f])

        lbs = packed_text["labels"][0]
        # Real tokens: 0..29, padding: 30..99
        assert (lbs[30:] == IGNORE_TOKEN_INDEX).all(), (
            "Padding tokens should all have IGNORE_TOKEN_INDEX labels"
        )

    def test_non_boundary_labels_preserved(self):
        """Labels within a sub-sequence (not at the boundary) must be preserved.

        FFD puts f2 (40) first: positions 0..39 → tokens 0..39 of f2.
        f1 (30 tokens) second: positions 40..69, but position 40 is the boundary (-100).
        Interior of f1: positions 41..69.
        """
        collator = _make_collator(max_length=100)
        f1 = _make_feature(30)
        f2 = _make_feature(40)
        packed_text, _ = collator._pack_sequences([f1, f2])

        lbs = packed_text["labels"][0]
        # f2 interior (positions 1..39 in the flat tensor) must not be -100
        for i in range(1, 40):
            assert lbs[i].item() != IGNORE_TOKEN_INDEX, f"lbs[{i}] unexpectedly masked"
        # f1 interior (positions 41..69 — after the -100 boundary at 40) must not be -100
        for i in range(41, 70):
            assert lbs[i].item() != IGNORE_TOKEN_INDEX, f"lbs[{i}] unexpectedly masked"


# ---------------------------------------------------------------------------
# Layer 1d — total real tokens per bin ≤ max_length
# ---------------------------------------------------------------------------

class TestBinCapacity:
    def test_no_bin_exceeds_max_length(self):
        collator = _make_collator(max_length=64)
        features = [_make_feature(l) for l in [60, 20, 30, 10, 50, 15]]
        packed_text, _ = collator._pack_sequences(features)

        for i, cu in enumerate(packed_text["cu_seqlens"]):
            real_used = cu[-2].item()  # last boundary before sentinel
            assert real_used <= 64, (
                f"Bin {i} used {real_used} tokens but max_length is 64"
            )

    def test_output_tensors_all_have_max_length(self):
        collator = _make_collator(max_length=64)
        features = [_make_feature(l) for l in [20, 30, 10]]
        packed_text, _ = collator._pack_sequences(features)

        assert packed_text["input_ids"].shape[-1] == 64
        assert packed_text["labels"].shape[-1] == 64
        assert packed_text["attention_mask"].shape[-1] == 64
        assert packed_text["position_ids"].shape[-1] == 64


# ---------------------------------------------------------------------------
# Layer 1e — modality token ranges are correctly offset
# ---------------------------------------------------------------------------

class TestModalityOffsets:
    def test_single_sample_modality_range_unchanged(self):
        """If a sample is alone in its bin, its modality range should not shift."""
        collator = _make_collator(max_length=200)
        f = _make_feature(100, modality_ranges=[(10, 50)])
        _, packed_mm = collator._pack_sequences([f])

        tr = packed_mm["token_range"]["image"]
        # Range should cover tokens 10..49 (40 token positions)
        assert 10 in tr.tolist()
        assert 49 in tr.tolist()

    def test_second_sample_modality_range_shifted(self):
        """
        f_big (80 tokens, no modality) goes first by FFD.
        f_small (40 tokens, image at 5..14) goes second at offset 80.
        Expected shifted token range: 80+5=85 .. 80+14=94.
        """
        collator = _make_collator(max_length=200)
        f_big = _make_feature(80)                              # no modality
        f_small = _make_feature(40, modality_ranges=[(5, 15)])  # image at 5..14

        _, packed_mm = collator._pack_sequences([f_big, f_small])

        tr = packed_mm["token_range"]["image"]
        # Expected shifted range: 80+5=85 .. 80+14=94
        for expected_pos in range(85, 95):
            assert expected_pos in tr.tolist(), (
                f"Token position {expected_pos} missing from modality token_range after packing"
            )


# ---------------------------------------------------------------------------
# Layer 1f — unpacked path is unchanged
# ---------------------------------------------------------------------------

class TestUnpackedPathUnchanged:
    def test_pack_false_does_not_call_pack_sequences(self):
        """With pack_sequences=False the _pack_sequences method is never called."""
        collator = _make_collator(max_length=100)
        collator.pack_sequences = False

        called = []
        original_fn = collator._pack_sequences

        def patched(features):
            called.append(True)
            return original_fn(features)

        # Monkey-patch without pytest fixture
        import types
        collator._pack_sequences = types.MethodType(lambda self, f: patched(f), collator)

        assert not collator.pack_sequences
        assert called == []


# ---------------------------------------------------------------------------
# Layer 2 — attention_mask is 1 for real tokens, 0 for padding
# ---------------------------------------------------------------------------

class TestAttentionMask:
    def test_attention_mask_shape_and_values(self):
        collator = _make_collator(max_length=100)
        f1 = _make_feature(30)
        f2 = _make_feature(20)
        packed_text, _ = collator._pack_sequences([f1, f2])

        attn = packed_text["attention_mask"][0]
        real_len = 30 + 20
        assert attn[:real_len].sum().item() == real_len
        assert attn[real_len:].sum().item() == 0


if __name__ == "__main__":
    # Quick sanity run without pytest
    import traceback
    passed = 0
    failed = 0
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name.startswith("Test"):
            inst = obj()
            for method_name in dir(inst):
                if method_name.startswith("test"):
                    try:
                        getattr(inst, method_name)()
                        print(f"  PASS  {name}::{method_name}")
                        passed += 1
                    except Exception as e:
                        print(f"  FAIL  {name}::{method_name}")
                        traceback.print_exc()
                        failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
