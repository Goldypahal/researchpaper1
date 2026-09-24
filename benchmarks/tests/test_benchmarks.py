"""
Unit Tests for Benchmark Suite: Dyck-k, Hidden FSM, and Legacy Parity.
Verifies mathematical correctness, depth isolation, split disjointness, and deterministic reproducibility.
"""

import unittest
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from benchmarks.dyck import (
    get_dyck_benchmark_splits,
    generate_dyck_k_samples,
    verify_sequence_validity,
    is_open_bracket,
    is_close_bracket,
    OPEN_BRACKETS,
    CLOSE_BRACKETS,
    BRACKET_MATCH
)
from benchmarks.fsm import (
    HiddenFSM,
    get_fsm_benchmark_splits,
    generate_fsm_samples
)
from benchmarks.parity_legacy import (
    get_parity_benchmark_splits
)


class TestDyckBenchmark(unittest.TestCase):
    def test_single_sequence_grammar_validity(self):
        """Verifies that generated sequences pass stack pop/push invariants."""
        data, masks, _ = generate_dyck_k_samples(
            num_samples=50, seq_len=64, vocab_size=512, k=4,
            min_depth=1, max_depth=5, seed=123
        )
        for i in range(len(data)):
            seq = data[i].tolist()
            mask = masks[i].tolist()
            valid, max_d, err = verify_sequence_validity(seq, mask)
            self.assertTrue(valid, f"Sequence {i} failed validity check: {err}")
            self.assertLessEqual(max_d, 5, f"Sequence {i} exceeded max depth 5: reached {max_d}")

    def test_depth_stratification_and_ood(self):
        """Checks that ID depth is in [1, 6] and OOD depth is in [7, 12]."""
        splits = get_dyck_benchmark_splits(
            num_train=50, num_val=20, num_test=20, num_ood=30,
            seq_len=64, min_depth_id=1, max_depth_id=6,
            min_depth_ood=7, max_depth_ood=12, seed=42
        )
        
        # Test ID train
        train_data, train_masks = splits["train"]
        for i in range(len(train_data)):
            valid, max_d, _ = verify_sequence_validity(train_data[i].tolist())
            self.assertTrue(valid)
            self.assertLessEqual(max_d, 6)
            
        # Test OOD
        ood_data, ood_masks = splits["ood"]
        for i in range(len(ood_data)):
            valid, max_d, _ = verify_sequence_validity(ood_data[i].tolist())
            self.assertTrue(valid)
            self.assertGreaterEqual(max_d, 7, f"OOD sequence {i} had depth {max_d} < 7")
            self.assertLessEqual(max_d, 12, f"OOD sequence {i} had depth {max_d} > 12")

    def test_split_disjointness_zero_leakage(self):
        """Proves D_train, D_val, and D_test have zero sample leakage."""
        splits = get_dyck_benchmark_splits(
            num_train=100, num_val=50, num_test=50, num_ood=50,
            seq_len=64, seed=99
        )
        train_set = {tuple(x.tolist()) for x in splits["train"][0]}
        val_set = {tuple(x.tolist()) for x in splits["val"][0]}
        test_set = {tuple(x.tolist()) for x in splits["test"][0]}
        ood_set = {tuple(x.tolist()) for x in splits["ood"][0]}

        self.assertEqual(len(train_set.intersection(val_set)), 0, "Train and Val overlap detected!")
        self.assertEqual(len(train_set.intersection(test_set)), 0, "Train and Test overlap detected!")
        self.assertEqual(len(val_set.intersection(test_set)), 0, "Val and Test overlap detected!")
        self.assertEqual(len(train_set.intersection(ood_set)), 0, "Train and OOD overlap detected!")

    def test_deterministic_reproducibility(self):
        """Ensures identical seed produces bitwise identical tensors."""
        splits_a = get_dyck_benchmark_splits(num_train=20, num_val=10, num_test=10, num_ood=10, seq_len=32, seed=777)
        splits_b = get_dyck_benchmark_splits(num_train=20, num_val=10, num_test=10, num_ood=10, seq_len=32, seed=777)

        self.assertTrue(torch.equal(splits_a["train"][0], splits_b["train"][0]))
        self.assertTrue(torch.equal(splits_a["val"][0], splits_b["val"][0]))
        self.assertTrue(torch.equal(splits_a["ood"][0], splits_b["ood"][0]))


class TestFSMBenchmark(unittest.TestCase):
    def test_fsm_transition_consistency(self):
        """Verifies state transition and emission table shapes and dynamics."""
        fsm = HiddenFSM(num_states=8, alphabet_size=16, seed=42)
        self.assertEqual(fsm.T.shape, (8, 16))
        self.assertEqual(fsm.E.shape, (8, 16))

        gen = torch.Generator().manual_seed(42)
        tokens, states = fsm.generate_sequence(seq_len=50, gen=gen)
        self.assertEqual(len(tokens), 50)
        self.assertEqual(len(states), 51)
        # All tokens should be within emission token range
        for tok in tokens:
            self.assertGreaterEqual(tok, 10)
            self.assertLess(tok, 26)

    def test_fsm_split_disjointness_and_ood(self):
        splits = get_fsm_benchmark_splits(
            num_train=50, num_val=20, num_test=20, num_ood=30,
            seq_len=48, num_states_id=8, num_states_ood=16, seed=1234
        )
        train_set = {tuple(x.tolist()) for x in splits["train"][0]}
        val_set = {tuple(x.tolist()) for x in splits["val"][0]}
        test_set = {tuple(x.tolist()) for x in splits["test"][0]}

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_fsm_deterministic_reproducibility(self):
        splits_a = get_fsm_benchmark_splits(num_train=20, num_val=10, num_test=10, num_ood=10, seq_len=32, seed=888)
        splits_b = get_fsm_benchmark_splits(num_train=20, num_val=10, num_test=10, num_ood=10, seq_len=32, seed=888)

        self.assertTrue(torch.equal(splits_a["train"][0], splits_b["train"][0]))
        self.assertTrue(torch.equal(splits_a["ood"][0], splits_b["ood"][0]))


if __name__ == "__main__":
    unittest.main()
