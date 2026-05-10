"""Train-step bench: Triton scan (hybrid backward) vs the PyTorch best path.

The Triton kernel currently has a Triton forward and a PyTorch reference
backward. This bench tells us how much of train-step time is forward —
i.e. how much we'd gain from also writing a Triton backward kernel.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import QuantizerConfig, QuantRecipe
from gru_qat.triton_kernels.scan import gru_scan


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


def _make_recipe() -> QuantRecipe:
    return QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )


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
    print(f"{'variant':40s} {'shape':22s} {'train ms':>10s}  {'vs cudnn':>10s}")
    print("-" * 90)

    for shape_str in args.shapes:
        seq, batch, hid = (int(x) for x in shape_str.split(","))
        in_dim = hid

        # cuDNN baseline
        cudnn = nn.GRU(in_dim, hid).to(device)
        x = torch.randn(seq, batch, in_dim, device=device)
        h0 = torch.randn(batch, hid, device=device)

        def cudnn_train() -> None:
            cudnn.zero_grad(set_to_none=True)
            out, _ = cudnn(x, h0.unsqueeze(0))
            loss = out.float().pow(2).sum()
            loss.backward()

        # Compiled PyTorch best
        rec = _make_recipe()
        ours_compiled = (
            GRULayer(in_dim, hid, recipe=rec,
                     gate_layout="fused", pre_batch_input=True,
                     compile_step=True)
            .to(device)
        )

        def ours_compiled_train() -> None:
            ours_compiled.zero_grad(set_to_none=True)
            out, _ = ours_compiled(x, h0)
            loss = out.float().pow(2).sum()
            loss.backward()

        # Triton hybrid: same parameters as ours_compiled (so apples-to-apples)
        Wi_cat = ours_compiled.cell.quant_W_ir(ours_compiled.cell.W_ir).detach()
        # Actually grab the concatenated weights properly
        with torch.no_grad():
            w = ours_compiled.cell.quantize_weights()
        Wi_cat_param = nn.Parameter(w.Wi_cat.detach().clone())
        bi_cat_param = nn.Parameter(w.bi_cat.detach().clone())
        Wh_cat_param = nn.Parameter(w.Wh_cat.detach().clone())
        bh_cat_param = nn.Parameter(w.bh_cat.detach().clone())
        tri_params = [Wi_cat_param, bi_cat_param, Wh_cat_param, bh_cat_param]

        def triton_hybrid_train() -> None:
            for pp in tri_params:
                if pp.grad is not None:
                    pp.grad = None
            gi = F.linear(x, Wi_cat_param, bi_cat_param)
            out = gru_scan(gi, h0, Wh_cat_param, bh_cat_param)
            loss = out.float().pow(2).sum()
            loss.backward()

        ms_cudnn = _median_ms(cudnn_train, args.warmup, args.iter)
        ms_compiled = _median_ms(ours_compiled_train, args.warmup, args.iter)
        ms_triton = _median_ms(triton_hybrid_train, args.warmup, args.iter)

        shape_fmt = f"({seq},{batch},{hid},{hid})"
        for name, ms in [
            ("cudnn_gru_fp32", ms_cudnn),
            ("ours_fp32_fused_prebatch_compiled", ms_compiled),
            ("triton_scan_fwd + pytorch_bwd", ms_triton),
        ]:
            ratio = f"{ms / ms_cudnn:5.2f}x"
            print(f"{name:40s} {shape_fmt:22s} {ms:10.3f}  {ratio:>10s}")
        print()


if __name__ == "__main__":
    main()
