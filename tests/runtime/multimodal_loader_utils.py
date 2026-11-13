from __future__ import annotations

import unittest

import torch

from hcu_sglang.multimodal_gen.loader import utils
from sglang.multimodal_gen.runtime.loader import utils as upstream


class StreamingLoaderUtilsTests(unittest.TestCase):
    def test_official_interfaces_keep_identity(self):
        for name in (
            "initialize_model",
            "finalize_loaded_model",
            "get_param_names_mapping",
            "hf_to_custom_state_dict",
            "_list_safetensors_files",
            "_load_safetensors_file",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(utils, name), getattr(upstream, name))
                self.assertIn(name, dir(utils))

    def test_checkpoint_loader_follows_installed_official_version(self):
        if hasattr(upstream, "load_safetensors_state_dict"):
            self.assertIs(
                utils.load_safetensors_state_dict,
                upstream.load_safetensors_state_dict,
            )
        else:
            with self.assertRaises(AttributeError):
                getattr(utils, "load_safetensors_state_dict")

    def test_missing_official_interface_is_not_synthesized(self):
        with self.assertRaises(AttributeError):
            getattr(utils, "_not_an_official_loader_interface")

    def test_identity_mapping_preserves_tensor_and_input(self):
        tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        checkpoint = {"weight": tensor}
        result = list(
            utils.iter_hf_to_custom_state_dict(
                checkpoint, upstream.get_param_names_mapping({})
            )
        )
        self.assertEqual(len(result), 1)
        name, value, reverse = result[0]
        self.assertEqual(name, "weight")
        self.assertIs(value, tensor)
        self.assertEqual(reverse, ("weight", None, None))
        self.assertEqual(list(checkpoint), ["weight"])
        self.assertIs(checkpoint["weight"], tensor)

    def test_iteration_does_not_consume_later_weights(self):
        consumed = []
        tensors = [torch.tensor([n]) for n in range(3)]

        def checkpoint():
            for index, tensor in enumerate(tensors):
                consumed.append(index)
                yield f"weight_{index}", tensor

        result = utils.iter_hf_to_custom_state_dict(
            checkpoint(), upstream.get_param_names_mapping({})
        )
        self.assertEqual(consumed, [])
        self.assertIs(next(result)[1], tensors[0])
        self.assertEqual(consumed, [0])
        self.assertIs(next(result)[1], tensors[1])
        self.assertEqual(consumed, [0, 1])

    def test_out_of_order_slices_merge_in_declared_order(self):
        tensors = {
            "v": torch.tensor([[30, 31]], dtype=torch.float32),
            "q": torch.tensor([[10, 11]], dtype=torch.float32),
            "k": torch.tensor([[20, 21]], dtype=torch.float32),
        }
        indices = {"q": 0, "k": 1, "v": 2}
        result = list(
            utils.iter_hf_to_custom_state_dict(
                tensors, lambda name: ("qkv", indices[name], 3)
            )
        )
        self.assertEqual(len(result), 1)
        name, tensor, reverse = result[0]
        self.assertEqual(name, "qkv")
        self.assertTrue(
            torch.equal(tensor, torch.tensor([[10, 11], [20, 21], [30, 31]]))
        )
        self.assertEqual(reverse, ("k", 1, 3))

    def test_existing_target_name_takes_precedence_over_rename(self):
        tensor = torch.tensor([4.0])
        result = list(
            utils.iter_hf_to_custom_state_dict(
                {"weight": tensor},
                lambda name: ("renamed", 0, 2),
                valid_target_names={"weight"},
            )
        )
        self.assertEqual(result[0][0], "weight")
        self.assertIs(result[0][1], tensor)
        self.assertEqual(result[0][2], ("weight", None, None))

    def test_ignored_keys_and_incomplete_groups_preserve_source_behavior(self):
        mapping = {
            "ignored": (None, None, None),
            "empty": ("", None, None),
            "partial": ("merged", 0, 2),
        }
        result = list(
            utils.iter_hf_to_custom_state_dict(
                {name: torch.ones(1) for name in mapping}, mapping.__getitem__
            )
        )
        self.assertEqual(result, [])

    def test_duplicate_precision_variants_reach_the_consumer(self):
        quantized = torch.tensor([2], dtype=torch.int8)
        unquantized = torch.tensor([2.0], dtype=torch.bfloat16)
        result = list(
            utils.iter_hf_to_custom_state_dict(
                iter((("int8", quantized), ("bf16", unquantized))),
                lambda name: ("weight", None, None),
            )
        )
        self.assertEqual([entry[0] for entry in result], ["weight", "weight"])
        self.assertIs(result[0][1], quantized)
        self.assertIs(result[1][1], unquantized)

    def test_unambiguous_conversion_matches_official_full_state_loader(self):
        checkpoint = {
            "second": torch.tensor([[3.0, 4.0]]),
            "bias": torch.tensor([5.0]),
            "first": torch.tensor([[1.0, 2.0]]),
        }

        def mapping(name):
            if name == "bias":
                return "bias", None, None
            return "weight", {"first": 0, "second": 1}[name], 2

        expected, reverse = upstream.hf_to_custom_state_dict(checkpoint, mapping)
        actual = list(utils.iter_hf_to_custom_state_dict(checkpoint, mapping))
        self.assertEqual({entry[0] for entry in actual}, set(expected))
        for name, tensor, origin in actual:
            self.assertTrue(torch.equal(tensor, expected[name]))
            self.assertEqual(origin, reverse[name])


if __name__ == "__main__":
    unittest.main()
