---
library_name: kernels
license: apache-2.0
tags:
  - kernel
  - attention
  - flexattention
  - provenance
  - szl-holdings
---

# szl-maskmod

Original SZL score_mod + block-mask attention (FlexAttention silhouette).
Inspired by Dong et al. https://arxiv.org/abs/2412.05496.
Not PyTorch flex_attention.py. No CuTeDSL. No speedup claim.
GitHub = source of truth. Hub = publish mirror. ATELIER owns cards.
Doctrine v11. Λ = Conjecture 1 OPEN. Apache-2.0. Copyright 2026 SZL Holdings.

```python
from kernels import get_kernel
km = get_kernel("SZLHOLDINGS/szl-maskmod", revision="main", trust_remote_code=True)
```

