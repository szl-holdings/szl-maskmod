# szl-maskmod

Canonical GitHub source for `SZLHOLDINGS/szl-maskmod`.

```python
from szl_maskmod import maskmod_attn, ReceiptChain, selfcheck
import torch
q = k = v = torch.randn(1, 2, 8, 16)
y = maskmod_attn(q, k, v, causal=True)
print(selfcheck())
```

Not a FlexAttention rehost. No CUDA benches. Λ = Conjecture 1.
