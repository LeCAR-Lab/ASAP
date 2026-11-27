import os
import sys
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoidverse.agents.modules.ppo_modules import PPOActor  # noqa: E402
from humanoidverse.utils import device as device_utils  # noqa: E402


def fake_hosts(xpu=False, cuda=False):
    return mock.patch.object(
        device_utils, "is_available", side_effect=lambda backend: {"xpu": xpu, "cuda": cuda}[backend]
    )


class DeviceSelectionTest(unittest.TestCase):
    def test_xpu_host(self):
        with fake_hosts(xpu=True):
            self.assertEqual(device_utils.default_device_type(), "xpu")
            self.assertEqual(device_utils.resolve_device(), "xpu:0")

    def test_cuda_is_kept_when_both_are_present(self):
        with fake_hosts(xpu=True, cuda=True):
            self.assertEqual(device_utils.default_device_type(), "cuda")
            self.assertEqual(device_utils.resolve_device(), "cuda:0")

    def test_cuda_host_is_unchanged(self):
        with fake_hosts(cuda=True):
            self.assertEqual(device_utils.default_device_type(), "cuda")
            self.assertEqual(device_utils.resolve_device(), "cuda:0")

    def test_cpu_host(self):
        with fake_hosts():
            self.assertEqual(device_utils.default_device_type(), "cpu")
            self.assertEqual(device_utils.resolve_device(), "cpu")

    def test_explicit_device_wins(self):
        with fake_hosts(xpu=True, cuda=True):
            self.assertEqual(device_utils.resolve_device("cpu"), "cpu")
            self.assertEqual(device_utils.resolve_device("cuda:1"), "cuda:1")
            self.assertEqual(device_utils.resolve_device("xpu:1"), "xpu:1")
            self.assertEqual(device_utils.resolve_device("xpu"), "xpu:0")
            self.assertEqual(device_utils.resolve_device("cuda"), "cuda:0")


def make_actor(obs_dim=64, num_actions=7):
    config = OmegaConf.create(
        {
            "input_dim": ["actor_obs"],
            "output_dim": ["robot_action_dim"],
            "layer_config": {"type": "MLP", "hidden_dims": [32, 16], "activation": "ELU"},
        }
    )
    actor = PPOActor({"actor_obs": obs_dim}, config, num_actions, 1.0)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in actor.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.3)
    return actor.eval()


@unittest.skipUnless(device_utils.is_available("xpu"), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_actor_matches_cpu(self):
        actor = make_actor()
        obs = torch.randn(32, 64, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            expected = actor.act_inference(obs)
            actual = actor.to("xpu").act_inference(obs.to("xpu")).cpu()
        self.assertGreater(expected.std().item(), 1e-3)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
