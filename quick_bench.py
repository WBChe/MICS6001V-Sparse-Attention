import torch
from block_sparse_attn import block_sparse_attn_func
from block_sparse_attn.utils.benchmark import benchmark_forward

def time_fwd(func, *args, **kwargs):
    result = benchmark_forward(func, *args, repeats=10, verbose=False, **kwargs)
    return result[1].mean

def flops(batch, seqlen, headdim, nheads):
    return 4 * batch * seqlen**2 * nheads * headdim // 2

BLOCK_SIZE = 128
batch, nheads, headdim = 8, 32, 128
device = "cuda"

print(f"{'seqlen':>8} {'sparsity':>10} {'time_ms':>10} {'TFLOP/s':>10}")
print("-" * 45)

for seqlen in [1024, 2048, 4096, 8192]:
    for sparsity in [0.0, 0.5, 0.9]:
        nblocks = seqlen // BLOCK_SIZE

        # shape: (1, nblocks, nblocks) -> repeat to (batch, nheads, nblocks, nblocks)
        base_mask = torch.zeros(1, nblocks, nblocks, device=device, dtype=torch.bool)
        if sparsity < 1.0:
            n_active = max(1, int(nblocks * nblocks * (1 - sparsity)))
            idx = torch.randperm(nblocks * nblocks)[:n_active]
            base_mask[0].view(-1)[idx] = True
        base_mask = base_mask.unsqueeze(0).repeat(batch, nheads, 1, 1)

        q = torch.randn(batch * seqlen, nheads, headdim, device=device, dtype=torch.float16)
        k = torch.randn(batch * seqlen, nheads, headdim, device=device, dtype=torch.float16)
        v = torch.randn(batch * seqlen, nheads, headdim, device=device, dtype=torch.float16)
        cu_seqlens = torch.arange(0, (batch+1)*seqlen, step=seqlen, dtype=torch.int32, device=device)
        head_mask_type = torch.ones(nheads, dtype=torch.int32, device=device)

        t = time_fwd(
            block_sparse_attn_func,
            q, k, v, cu_seqlens, cu_seqlens,
            head_mask_type, None, base_mask,
            seqlen, seqlen, 0.0,
            is_causal=True,
        )
        f = flops(batch, seqlen, headdim, nheads)
        tflops = f / t / 1e12
        print(f"{seqlen:>8} {sparsity:>10.0%} {t*1000:>10.3f} {tflops:>10.2f}")
