"""CPU contracts for the independent GLP-based conditional mask denoiser."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditional_diffusion.diffusion_seg_model import ConditionalMaskDenoiser


def small_model(state_channels=3):
    return ConditionalMaskDenoiser(
        state_channels=state_channels, timesteps=20,
        widths=(8, 16, 24, 32), token_dim=32, heads=4, na_kernel=3,
    )


def open_attention_gates(model):
    # adaLN-Zero initially suppresses residual attention. Open its learned gates
    # to inspect gradient routing after that initial training state.
    with torch.no_grad():
        for block in model.joint:
            for stream in block.streams:
                stream["modulation"].bias.fill_(0.1)
                stream["modulation"].weight.normal_(std=0.01)


class ConditionalMaskDenoiserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(73)

    def test_cached_forward_and_both_state_representations(self):
        for state_channels in (3, 4):
            model = small_model(state_channels).eval()
            images = torch.randn(2, 3, 32, 48)
            states = torch.randn(2, state_channels, 32, 48)
            boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 1.0, 1.0]])
            times = torch.tensor([0, 19])
            with torch.no_grad():
                contexts = model.encode_global([torch.randn(3, 64, 64), torch.randn(3, 64, 96)])
                with patch.object(model.local, "forward", wraps=model.local.forward) as local:
                    condition = model.encode_condition(images, boxes, contexts)
                    cached = model.denoise(condition, states, times)
                    model.denoise(condition, states, times.flip(0))
                    self.assertEqual(local.call_count, 1)
                direct = model(images, states, times, boxes, contexts)
            self.assertEqual(tuple(cached.shape), (2, 4, 32, 48))
            torch.testing.assert_close(cached, direct, rtol=0, atol=0)
            self.assertEqual(condition["valid"].sum(1).tolist(), [4, 6])
            self.assertTrue(torch.isfinite(cached).all())

    def test_odd_size_padding_preserves_real_pixel_coordinates(self):
        model = small_model().eval()
        image = torch.randn(1, 3, 31, 47)
        boxes = torch.tensor([[0.1, 0.2, 0.6, 0.8]])
        with torch.no_grad():
            contexts = model.encode_global([torch.randn(3, 64, 96)])
            condition = model.encode_condition(image, boxes, contexts)
            prediction = model.denoise(condition, torch.randn(1, 3, 31, 47), torch.tensor([10]))
        self.assertEqual(prediction.shape, (1, 4, 31, 47))
        self.assertEqual(condition["features"][0].shape[-2:], (32, 48))
        expected_end = boxes[:, :2] + (boxes[:, 2:] - boxes[:, :2]) * torch.tensor([32 / 31, 48 / 47])
        torch.testing.assert_close(condition["boxes"][:, 2:], expected_end)
        torch.testing.assert_close(condition["prior"][:, :, -1, :], condition["prior"][:, :, -2, :])
        torch.testing.assert_close(condition["prior"][:, :, :, -1], condition["prior"][:, :, :, -2])

    def test_local_global_and_state_gradients_with_detached_prior(self):
        model = small_model().train()
        open_attention_gates(model)
        contexts = model.encode_global([torch.randn(3, 64, 64)])
        contexts[0][0].retain_grad()
        contexts[0][1].retain_grad()
        condition = model.encode_condition(torch.randn(1, 3, 32, 32), torch.tensor([[0.0, 0.0, 1.0, 1.0]]), contexts)
        self.assertTrue(condition["features"][0].requires_grad)
        self.assertTrue(condition["tokens"].requires_grad)
        self.assertFalse(condition["prior"].requires_grad)
        self.assertFalse(condition["uncertainty"].requires_grad)
        states = torch.randn(1, 3, 32, 32, requires_grad=True)
        model.denoise(condition, states, torch.tensor([5])).square().mean().backward()
        self.assertGreater(float(states.grad.abs().sum()), 0)
        self.assertGreater(float(contexts[0][0].grad.abs().sum()), 0)
        self.assertIsNone(contexts[0][1].grad)
        for module in (model.local, model.time_mlp, model.state_stem, model.head):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        model.zero_grad(set_to_none=True)
        contexts = model.encode_global([torch.randn(3, 64, 64)])
        contexts[0][1].square().mean().backward()
        self.assertGreater(float(model.global_encoder.prior_head[-1].weight.grad.abs().sum()), 0)

    def test_padding_global_tokens_does_not_change_single_window_result(self):
        model = small_model().eval()
        open_attention_gates(model)
        images = torch.randn(2, 3, 32, 32)
        states = torch.randn(2, 3, 32, 32)
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(2, 1)
        times = torch.tensor([5, 5])
        with torch.no_grad():
            contexts = model.encode_global([torch.randn(3, 64, 64), torch.randn(3, 96, 96)])
            batched = model(images, states, times, boxes, contexts)
            single = model(images[:1], states[:1], times[:1], boxes[:1], contexts[:1])
        torch.testing.assert_close(batched[:1], single, rtol=2e-5, atol=2e-6)

    def test_invalid_state_time_and_configuration_are_rejected(self):
        for arguments in ({"state_channels": 2}, {"timesteps": 1}, {"widths": (8, 16, 24)},
                          {"token_dim": 30}, {"heads": 3}, {"na_kernel": 2}):
            with self.assertRaises(ValueError):
                ConditionalMaskDenoiser(**arguments)
        model = small_model().eval()
        with torch.no_grad():
            contexts = model.encode_global([torch.randn(3, 64, 64)])
            condition = model.encode_condition(torch.randn(1, 3, 32, 32), torch.tensor([[0.0, 0.0, 1.0, 1.0]]), contexts)
            for times in (torch.tensor([-1]), torch.tensor([20]), torch.tensor([1.5]), torch.tensor([1, 2])):
                with self.assertRaises(ValueError):
                    model.denoise(condition, torch.randn(1, 3, 32, 32), times)
            with self.assertRaises(ValueError):
                model.denoise(condition, torch.randn(1, 4, 32, 32), torch.tensor([1]))
            with self.assertRaises(ValueError):
                model.encode_condition(torch.randn(1, 3, 32, 32), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), contexts)


if __name__ == "__main__":
    unittest.main()
