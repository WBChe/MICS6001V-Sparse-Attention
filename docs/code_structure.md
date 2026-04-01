# 代码结构分析

## 整体层次
```
Python层（用户调用）
├── block_sparse_attn_func()        # blocksparse主入口
├── block_streaming_attn_func()     # block级streaming
└── token_streaming_attn_func()     # token级streaming

     ↓ mask转换
BlockSparseAttnFunc（autograd Function）
├── forward:  convert_blockmask_row_reverse() → CUDA fwd
└── backward: convert_blockmask_col_reverse() → CUDA bwd

     ↓ C++绑定
flash_api.cpp

     ↓ CUDA kernel
kernel_traits.h          # kernel模板参数
flash_fwd_kernel.h       # forward kernel实现
flash_bwd_kernel.h       # backward kernel实现
flash_fwd_launch_template.h  # 根据参数选择kernel配置
```

## mask的三种形态
```
用户输入:  base_blockmask  (batch, nheads, nblocks_q, nblocks_k)  bool
    ↓ convert_blockmask_row_reverse()
fwd输入:  row_blockmask   (batch, nheads, nblocks_q, nblocks_k)  int32
    ↓ convert_blockmask_col_reverse()
bwd输入:  col_blockmask   (batch, nheads, nblocks_k, nblocks_q)  int32
```

转换作用：bool矩阵 → 每行只存非零列索引，-1表示跳过。
kernel靠这个索引决定跳过哪些block。

## 支持的四种Attention模式

| 模式 | head_mask_type | 说明 |
|------|---------------|------|
| Dense | 0 | 完整attention，等价于FlashAttention |
| Block Sparse | 1 | 传入base_blockmask，按block跳过 |
| Block Streaming | -1 | sink blocks + local blocks |
| Token Streaming | -1 + exact_streaming=True | token级sink + local |

## 不同head可以用不同模式
```python
# 例：8个head，混合模式
head_mask_type = [1, 1, 0, 0, 0, -1, 0, -1]
# head0,1: blocksparse
# head2,3,4,6: dense
# head5,7: streaming
```

## kernel参数选择（_get_block_size）

根据GPU架构和head_dim自动选择tile大小：
- sm80 (A100): 针对HBM带宽优化
- sm8x (sm86/sm89): 目前和sm86共用同一配置
- sm90 (H100): 有专门优化

## 优化切入点

| 优化方向 | 文件位置 |
|---------|---------|
| Ada(sm89)架构调优 | `block_sparse_attn_interface.py: _get_block_size()` + `setup.py` |
| mask存储压缩 | `block_sparse_attn_interface.py: convert_blockmask_row_reverse()` |
| GQA支持 | `block_sparse_attn_interface.py: block_sparse_attn_func()` |
