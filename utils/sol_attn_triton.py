"""Sol-Attn Triton kernels (portable reference implementation).

Vendored from NVlabs/Sana, `sol-engine` branch
(techniques/sparse_backends/sol_attn: preprocess.py + triton_ref/fwd.py),
Apache-2.0.  Paper: "Sol-Attn: Accelerating Video Generation Inference via
On-the-Fly Attention Sparsification" (arXiv:2607.24027).

Local changes:
* removed the SM90/SM100-only architecture gate so the portable Triton
  reference can run on any TMA-capable GPU (tested on Blackwell sm_120);
* single-file layout with the "diag" threshold routing path only;
* validation raises for unsupported inputs so callers can fall back to
  dense attention.

Contract: q, k, v are contiguous bfloat16 CUDA tensors in BTHD layout
[batch, tokens, heads, 128].  Returns attention output in the same layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK_SIZE = 64
GROUP = 32
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != HEAD_DIM:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous BTHD tensors")
    arch = torch.cuda.get_device_capability(q.device)
    if arch < (9, 0):
        raise RuntimeError(
            "the Sol-Attn Triton reference needs a TMA-capable GPU (sm90+)"
        )


# ---------------------------------------------------------------------------
# Preprocess: block summaries and per-(query-block, head) routing thresholds
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["T"],
)
@triton.jit
def _reduce_kc_kernel(
    k_desc,
    kc,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    values = k_desc.load(
        [batch, block * BLOCK, head, d_tile * TILE_D]
    ).reshape([BLOCK, TILE_D])
    summary = tl.sum(values, axis=0) / block_len
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    tl.store(
        kc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["T"],
)
@triton.jit
def _reduce_vc_kernel(
    v_desc,
    vc,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    values = v_desc.load(
        [batch, block * BLOCK, head, d_tile * TILE_D]
    ).reshape([BLOCK, TILE_D])
    summary = tl.sum(values, axis=0)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    tl.store(
        vc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["N"],
)
@triton.jit
def _reduce_kc_stats_kernel(
    kc_desc,
    kc_mean,
    kc_var_diag,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    TILE_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    d_tile, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    block_offsets = tl.arange(0, GROUP_SIZE)
    block_offsets = tl.max_contiguous(block_offsets, GROUP_SIZE)
    d_offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    total = tl.zeros((TILE_D,), dtype=tl.float32)
    total_sq = tl.zeros((TILE_D,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, N, GROUP_SIZE):
        valid = start + block_offsets < N
        values = kc_desc.load(
            [batch, start, head, d_tile * TILE_D]
        ).reshape([GROUP_SIZE, TILE_D]).to(tl.float32)
        values = tl.where(valid[:, None], values, 0.0)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        count += tl.sum(valid.to(tl.float32), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    valid_d = d_offsets < D
    tl.store(
        kc_mean + batch_head * D + d_offsets,
        mean,
        mask=valid_d,
    )
    tl.store(
        kc_var_diag + batch_head * D + d_offsets,
        variance,
        mask=valid_d,
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["T"],
)
@triton.jit
def _diag_threshold_kernel(
    q_desc,
    kc_mean,
    kc_var_diag,
    global_threshold,
    softmax_scale,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
    TAU: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    d_offsets = tl.arange(0, TILE_D)
    valid_d = d_offsets < D
    q_values = q_desc.load(
        [batch, q_start, head, 0]
    ).reshape([BLOCK, TILE_D])
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(
        kc_mean + batch_head * D + d_offsets,
        mask=valid_d,
        other=0.0,
    )
    var_kc = tl.load(
        kc_var_diag + batch_head * D + d_offsets,
        mask=valid_d,
        other=0.0,
    )
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc, axis=0
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(
        global_threshold + (batch * N + q_block) * H + head,
        mean + TAU * std,
    )


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, head_dim = k.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    kc = torch.empty(
        (batch, blocks, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_like(kc)
    k_desc = TensorDescriptor.from_tensor(
        k,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    v_desc = TensorDescriptor.from_tensor(
        v,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    grid = (triton.cdiv(head_dim, tile_d), blocks, batch * heads)
    _reduce_kc_kernel[grid](
        k_desc,
        kc,
        tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    _reduce_vc_kernel[grid](
        v_desc,
        vc,
        tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> torch.Tensor:
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    kc_mean = torch.empty(
        (batch, heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    kc_var_diag = torch.empty_like(kc_mean)
    global_threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    q_desc = TensorDescriptor.from_tensor(
        q,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    kc_desc = TensorDescriptor.from_tensor(
        kc,
        [1, THRESHOLD_GROUP_SIZE, 1, tile_d],
    )
    _reduce_kc_stats_kernel[
        (triton.cdiv(head_dim, tile_d), batch * heads)
    ](
        kc_desc,
        kc_mean,
        kc_var_diag,
        heads,
        blocks,
        head_dim,
        tile_d,
        THRESHOLD_GROUP_SIZE,
    )
    _diag_threshold_kernel[(blocks, batch * heads)](
        q_desc,
        kc_mean,
        kc_var_diag,
        global_threshold,
        scale,
        tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        tau,
    )
    return global_threshold


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kc, vc = _reduce_kv(k, v)
    threshold = _compute_diag_threshold(q, kc, tau=tau, scale=scale)
    return kc, vc, threshold


# ---------------------------------------------------------------------------
# Forward: single online-softmax pass with on-the-fly block routing
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["T"],
)
@triton.jit
def _forward(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
    vc_desc,
    threshold,
    o_desc,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)
    q_start = q_block * BLOCK
    q = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK, D])
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK,), dtype=tl.float32)
    row_max = tl.full((BLOCK,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK
    route_threshold = tl.load(
        threshold + (batch * NT + q_block) * H + head
    )

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = kc_desc.load(
            [batch, group_start, head, 0]
        ).reshape([GROUP_SIZE, D])
        vc = vc_desc.load(
            [batch, group_start, head, v_tile * BV]
        ).reshape([GROUP_SIZE, BV])
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        exact = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
        ) & valid

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype), vc
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, GROUP_SIZE, exact_offsets
            )
            kv_start = block * BLOCK
            k = k_desc.load(
                [batch, kv_start, head, 0]
            ).reshape([BLOCK, D])
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                (kv_start + token_offsets)[None, :] < T,
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = v_desc.load(
                [batch, kv_start, head, v_tile * BV]
            ).reshape([BLOCK, BV])
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype), v
            )
            row_max = new_max

    o_desc.store(
        [batch, q_start, head, v_tile * BV],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Run the Sol-Attn Triton reference on contiguous BTHD bf16 inputs."""

    _validate(q, k, v)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    kc, vc, threshold = prepare(q, k, v, scale=scale, tau=tau)
    output = torch.empty_like(v)
    block_shape = [1, BLOCK_SIZE, 1, head_dim]
    summary_shape = [1, GROUP, 1, head_dim]
    _forward[(1, blocks, batch * heads)](
        TensorDescriptor.from_tensor(q, block_shape),
        TensorDescriptor.from_tensor(k, block_shape),
        TensorDescriptor.from_tensor(v, block_shape),
        TensorDescriptor.from_tensor(kc, summary_shape),
        TensorDescriptor.from_tensor(vc, summary_shape),
        threshold,
        TensorDescriptor.from_tensor(output, block_shape),
        scale,
        tokens,
        heads,
        head_dim,
        blocks,
        head_dim,
        BLOCK_SIZE,
        GROUP,
    )
    return output


__all__ = ["sol_attn", "BLOCK_SIZE", "HEAD_DIM"]
