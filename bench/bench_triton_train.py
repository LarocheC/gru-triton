"""Train-step bench: Triton scan (forward + backward) vs the PyTorch best path.

Two regimes:
- ``--mode fp32``: identity-quantizers reference (no fake-quant in either path).
- ``--mode qat``:  int8 hidden quant frozen, both paths use it. Tests the
  in-kernel fake-quant win.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import (
    FakeQuantizePerTensor,
    QuantizerConfig,
    QuantRecipe,
)
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
    p.add_argument("--mode", choices=["fp32", "qat"], default="fp32")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# warmup={args.warmup} iter={args.iter}")
    print()
    print(f"{'variant':40s} {'shape':22s} {'train ms':>10s}  {'vs cudnn':>10s}")
    print("-" * 90)

    qat_mode = args.mode == "qat"
    print(f"# mode: {args.mode}")
    print()

    for shape_str in args.shapes:
        seq, batch, hid = (int(x) for x in shape_str.split(","))
        in_dim = hid

        # cuDNN baseline (always fp32, no quant — it's the speed ceiling)
        cudnn = nn.GRU(in_dim, hid).to(device)
        x = torch.randn(seq, batch, in_dim, device=device) * 0.1
        h0 = torch.randn(batch, hid, device=device) * 0.1

        def cudnn_train() -> None:
            cudnn.zero_grad(set_to_none=True)
            out, _ = cudnn(x, h0.unsqueeze(0))
            loss = out.float().pow(2).sum()
            loss.backward()

        # PyTorch best path. In QAT mode, build int8 frozen hidden quant.
        if qat_mode:
            bits = 8
            qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
            h_scale = 0.02
            rec = QuantRecipe(
                weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
                input_act=QuantizerConfig(bits=32, name="x_id"),
                hidden=QuantizerConfig(bits=bits, mode="frozen", name="h_q"),
            )
        else:
            rec = _make_recipe()
            h_scale = qmin = qmax = None  # unused
        ours_compiled = (
            GRULayer(in_dim, hid, recipe=rec,
                     gate_layout="fused", pre_batch_input=True,
                     compile_step=True)
            .to(device)
        )
        if qat_mode:
            for q in (ours_compiled.cell.quant_h_in,
                      ours_compiled.cell.quant_h_out):
                assert isinstance(q, FakeQuantizePerTensor)
                q.scale = torch.tensor(h_scale, device=device)
                q.zero_point = torch.tensor(0.0, device=device)

        def ours_compiled_train() -> None:
            ours_compiled.zero_grad(set_to_none=True)
            out, _ = ours_compiled(x, h0)
            loss = out.float().pow(2).sum()
            loss.backward()

        # Triton path: same parameters as ours_compiled.
        with torch.no_grad():
            w = ours_compiled.cell.quantize_weights()
        Wi_cat_param = nn.Parameter(w.Wi_cat.detach().clone())
        bi_cat_param = nn.Parameter(w.bi_cat.detach().clone())
        Wh_cat_param = nn.Parameter(w.Wh_cat.detach().clone())
        bh_cat_param = nn.Parameter(w.bh_cat.detach().clone())
        tri_params = [Wi_cat_param, bi_cat_param, Wh_cat_param, bh_cat_param]

        h_in_q = (h_scale, qmin, qmax) if qat_mode else None
        h_out_q = (h_scale, qmin, qmax) if qat_mode else None

        def triton_train() -> None:
            for pp in tri_params:
                if pp.grad is not None:
                    pp.grad = None
            gi = F.linear(x, Wi_cat_param, bi_cat_param)
            out = gru_scan(
                gi, h0, Wh_cat_param, bh_cat_param,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
            loss = out.float().pow(2).sum()
            loss.backward()

        ms_cudnn = _median_ms(cudnn_train, args.warmup, args.iter)
        ms_compiled = _median_ms(ours_compiled_train, args.warmup, args.iter)
        ms_triton = _median_ms(triton_train, args.warmup, args.iter)

        shape_fmt = f"({seq},{batch},{hid},{hid})"
        pt_label = (
            "ours_qat_compiled" if qat_mode else "ours_fp32_compiled"
        )
        tri_label = "triton_scan_qat" if qat_mode else "triton_scan_fp32"
        for name, ms in [
            ("cudnn_gru_fp32", ms_cudnn),
            (pt_label, ms_compiled),
            (tri_label, ms_triton),
        ]:
            ratio = f"{ms / ms_cudnn:5.2f}x"
            print(f"{name:40s} {shape_fmt:22s} {ms:10.3f}  {ratio:>10s}")
        print()


if __name__ == "__main__":
    main()
