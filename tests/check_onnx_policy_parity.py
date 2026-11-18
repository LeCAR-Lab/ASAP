"""Compare the released ONNX policies with the same weights run as PyTorch modules.

The weights of every ONNX policy under sim2real/models are loaded into the
repo's own PPOActor. The actor runs on the CPU (float32) and on the chosen
device, and the outputs are compared with onnxruntime on the CPU.

    python tests/check_onnx_policy_parity.py --device xpu
"""

import argparse
import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from omegaconf import OmegaConf
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoidverse.agents.modules.ppo_modules import PPOActor  # noqa: E402
from humanoidverse.utils.device import resolve_device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_actor(onnx_path):
    """Build a PPOActor with the weights of an exported ONNX policy."""
    weights = [numpy_helper.to_array(t) for t in onnx.load(onnx_path).graph.initializer]
    mats, biases = weights[0::2], weights[1::2]
    config = OmegaConf.create(
        {
            "input_dim": ["actor_obs"],
            "output_dim": ["robot_action_dim"],
            "layer_config": {
                "type": "MLP",
                "hidden_dims": [int(m.shape[0]) for m in mats[:-1]],
                "activation": "ELU",
            },
        }
    )
    actor = PPOActor({"actor_obs": int(mats[0].shape[1])}, config, int(mats[-1].shape[0]), 1.0)
    linears = [m for m in actor.actor.module if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, w, b in zip(linears, mats, biases):
            lin.weight.copy_(torch.from_numpy(w.copy()))
            lin.bias.copy_(torch.from_numpy(b.copy()))
    return actor.eval()


def metrics(test, ref, atol, rtol):
    test, ref = test.double(), ref.double()
    diff = test - ref
    mse = diff.pow(2).mean().item()
    inside = ((diff.abs() <= atol + rtol * ref.abs()).float().mean().item()) * 100
    return {
        "mse": mse,
        "rel_mse": mse / ref.pow(2).mean().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(test.flatten(), ref.flatten(), dim=0).item(),
        "inside_pct": inside,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="default: xpu, then cuda, then cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--obs-scale", type=float, default=0.1, help="standard deviation of the random observations"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = resolve_device(args.device)

    paths = sorted(glob.glob(os.path.join(ROOT, "sim2real/models/**/*.onnx"), recursive=True))
    print(f"device={device} torch={torch.__version__} batch={args.batch} obs_scale={args.obs_scale}")
    print("rows: name | comparison | mse | rel_mse | max_abs | cosine | % inside (atol 2e-2, rtol 1e-2 for bf16, 1e-4/1e-4 for fp32)")
    for path in paths:
        name = os.path.relpath(path, os.path.join(ROOT, "sim2real/models"))
        actor = build_actor(path)
        gen = torch.Generator().manual_seed(args.seed)
        obs_dim = actor.actor.input_dim
        obs = torch.randn(args.batch, obs_dim, generator=gen) * args.obs_scale

        session = onnxruntime.InferenceSession(path)
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        ort = torch.from_numpy(
            np.concatenate([session.run([out_name], {in_name: o[None].numpy()})[0] for o in obs])
        )
        with torch.no_grad():
            cpu = actor.act_inference(obs)
            fp32_dev = actor.to(device).act_inference(obs.to(device)).cpu()
            actor.to("cpu")
            bf16_dev = actor.to(device, torch.bfloat16).act_inference(obs.to(device, torch.bfloat16)).float().cpu()
            actor.to("cpu", torch.float32)
            bf16_cpu = actor.to(torch.bfloat16).act_inference(obs.to(torch.bfloat16)).float()
        rows = {
            "torch cpu fp32 vs onnxruntime": metrics(cpu, ort, 1e-4, 1e-4),
            f"{device} fp32 vs onnxruntime": metrics(fp32_dev, ort, 1e-4, 1e-4),
            f"{device} fp32 vs cpu fp32": metrics(fp32_dev, cpu, 1e-4, 1e-4),
            f"{device} bf16 vs cpu fp32": metrics(bf16_dev, cpu, 2e-2, 1e-2),
            "cpu bf16 vs cpu fp32": metrics(bf16_cpu, cpu, 2e-2, 1e-2),
        }
        for label, m in rows.items():
            print(
                f"{name} | {label} | {m['mse']:.3e} | {m['rel_mse']:.3e} | {m['max_abs']:.3e} | "
                f"{m['cosine']:.8f} | {m['inside_pct']:.2f}"
            )


if __name__ == "__main__":
    main()
