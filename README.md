# MICS6001V: LLM Serving on GPUs — Efficient Sparse Attention

复现并优化 [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
优化后Prototype代码 [Repo](https://github.com/henrylin46/MICS6001V-BSA) [Result](https://github.com/henrylin46/MICS6001V-BSA/blob/main/BSR_BENCHMARK_RESULTS.md)

## 环境
- GPU: NVIDIA RTX 6000 Ada Generation (49GB × 4)
- CUDA: 12.8 / PyTorch: 2.7.0+cu128
- 集群: fpga04 (192.168.200.207)，通过entry node (10.92.254.111) 跳转

## 复现结果

### Correctness
| 测试 | 结果 |
|------|------|
| fwd correctness | 28800/28800 passed ✅ |
| fwd+bwd correctness | 14400/14400 passed ✅ |

### Performance — Forward only (batch=8, nheads=32, headdim=128, causal)

| seqlen | sparsity | speedup vs FlashAttn2 |
|--------|----------|-----------------------|
| 4096   | 90%      | 4.8x                  |
| 8192   | 90%      | 6.4x                  |
| 16384  | 90%      | 6.9x                  |
| 32768  | 90%      | 7.3x                  |
| 65536  | 90%      | 8.3x                  |

### Performance — Forward+Backward (batch=1, nheads=32, headdim=128, causal)

| seqlen | sparsity | speedup vs FlashAttn2 |
|--------|----------|-----------------------|
| 8192   | 90%      | 3.7x                  |
| 16384  | 90%      | 7.0x                  |
| 16384  | 95%      | 10.4x                 |
| 32768  | 90%      | 7.6x                  |
| 32768  | 95%      | 13.3x                 |

## 目录结构
```
reproduction/
├── benchmarks/
│   └── quick_bench.py              # 快速benchmark脚本（不依赖flash_attn）
└── results/
    ├── benchmark_rtx6000ada.md     # 早期benchmark记录
    ├── all_results_*_fwd_causal.json       # fwd完整数据
    ├── all_results_*_fwd_bwd_causal.json   # fwd+bwd完整数据
    └── *.xlsx                      # Excel格式结果

docs/
└── code_structure.md               # 代码结构详细分析
```

## 代码结构

见 [docs/code_structure.md](docs/code_structure.md)

## 优化方向（进行中）

1. **Ada架构(sm_89)调优** — 针对RTX 6000 Ada调整kernel tile参数
2. **Block mask存储优化** — CSR格式压缩，降低显存占用
3. **GQA支持** — 支持Grouped Query Attention
