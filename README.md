# szl-maskmod

Canonical GitHub source for `SZLHOLDINGS/szl-maskmod`.
GitHub bytes are the artifact. Hugging Face Hub is a publish mirror.

```python
from kernels import get_kernel
km = get_kernel("SZLHOLDINGS/szl-maskmod", revision="main", trust_remote_code=True)
```

```python
from szl_maskmod import maskmod_attn, ReceiptChain, selfcheck
import torch
q = k = v = torch.randn(1, 2, 8, 16)
y = maskmod_attn(q, k, v, causal=True)
print(selfcheck())
```

Not a FlexAttention rehost. No CUDA benches. No speedup claim.
Doctrine v11. Λ = Conjecture 1 OPEN.
