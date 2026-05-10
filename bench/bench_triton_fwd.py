"""Forward-only bench for the Triton scan kernel.

The Triton kernel doesn't have a backward yet, so this is forward-only.
If the forward win here is real, the backward is worth writing; if not,
we stop and tell the user.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS, QuantizerConfig, QuantRecipe
from gru_qat.triton_kernels.scan import gru_scan_forward


def _sync() -> None:
    torch.cuda.synchronize()


def _median_ms(fn, n_warmup: int, n_iter: int) -> float:
    for _ in range(n_warmup):
        fn()
    _sync()
    samples = []
    for _ in range(n_iter):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["32,16,256", "64,32,512"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iter", type=int, default=30)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# warmup={args.warmup} iter={args.iter}")
    print()
    print(f"{'variant':32s} {'shape':22s} {'fwd ms':>10s}  {'vs cudnn':>10s}")
    print("-" * 80)

    for shape_str in args.shapes:
        seq, batch, hid = (int(x) for x in shape_str.split(","))
        in_dim = hid

        x = torch.randn(seq, batch, in_dim, device=device)
        h0 = torch.randn(batch, hid, device=device)

        # cuDNN
        cudnn = nn.GRU(in_dim, hid).to(device).eval()

        def cudnn_fwd() -> None:
            with torch.no_grad():
                cudnn(x, h0.unsqueeze(0))

        # Ours fused+prebatch (no compile)
        rec_fp = QuantRecipe(
            weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
            input_act=QuantizerConfig(bits=32, name="x_id"),
            hidden=QuantizerConfig(bits=32, name="h_id"),
        )
        ours = (
            GRULayer(in_dim, hid, recipe=rec_fp,
                     gate_layout="fused", pre_batch_input=True)
            .to(device).eval()
        )
        ours_compiled = (
            GRULayer(in_dim, hid, recipe=rec_fp,
                     gate_layout="fused", pre_batch_input=True,
                     compile_step=True)
            .to(device).eval()
        )
        ours_compiled.load_state_dict(ours.state_dict())

        def ours_fwd() -> None:
            with torch.no_grad():
                ours(x, h0)

        def ours_compiled_fwd() -> None:
            with torch.no_grad():
                ours_compiled(x, h0)

        # Triton-scan fp32 forward — uses ours's quantize_weights/input_projection
        # to set up gi exactly the same way as the PyTorch path.
        with torch.no_grad():
            w = ours.cell.quantize_weights()

        def triton_fwd() -> None:
            with torch.no_grad():
                gi = ours.cell.input_projection(x, w)
                gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

        ms_cudnn = _median_ms(cudnn_fwd, args.warmup, args.iter)
        ms_ours = _median_ms(ours_fwd, args.warmup, args.iter)
        ms_ours_c = _median_ms(ours_compiled_fwd, args.warmup, args.iter)
        ms_triton = _median_ms(triton_fwd, args.warmup, args.iter)

        shape_fmt = f"({seq},{batch},{hid},{hid})"
        for name, ms in [
            ("cudnn_gru_fp32", ms_cudnn),
            ("ours_fp32_fused_prebatch", ms_ours),
            ("ours_fp32_fused_prebatch_compiled", ms_ours_c),
            ("triton_scan_fp32", ms_triton),
        ]:
            ratio = f"{ms / ms_cudnn:5.2f}x"
            print(f"{name:32s} {shape_fmt:22s} {ms:10.3f}  {ratio:>10s}")
        print()


if __name__ == "__main__":
    main()
