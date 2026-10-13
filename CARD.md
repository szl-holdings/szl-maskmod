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

<p><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="40" height="40" /></p>


# szl-maskmod

Original SZL score_mod + block-mask attention (FlexAttention silhouette).
Inspired by Dong et al. https://arxiv.org/abs/2412.05496.
Not PyTorch flex_attention.py. No CuTeDSL. No speedup claim.
Doctrine v11. Λ = Conjecture 1 OPEN. Apache-2.0. Copyright 2026 SZL Holdings.

Pure-Python (noarch) kernel published from GitHub
[szl-holdings/szl-maskmod](https://github.com/szl-holdings/szl-maskmod) by its
`hf-kernel-publish` workflow. GitHub is the source of truth; this kernel repository
is written only by that workflow, and this card is published from `CARD.md` there.

## Load it

```python
# pip install "kernels>=0.17.2" torch
from kernels import get_kernel

km = get_kernel("SZLHOLDINGS/szl-maskmod", version=1, trust_remote_code=True)
```

- `version=1` resolves the `v1` branch, the stable API channel. It is republished on
  every release. The `kernels` library refuses a call that names neither a version nor
  a revision.
- `trust_remote_code` is required: SZLHOLDINGS is not a trusted kernel publisher on the Hub.
  The narrowest form is `trust_remote_code=["SZLHOLDINGS/szl-maskmod"]`.
- CPU hosts load `build/torch-cpu`; CUDA and other hosts load `build/torch-universal`.
  Both variants carry identical code, built from the same GitHub commit.

### Pin an exact build (production)

```python
from huggingface_hub import HfApi
from kernels import get_kernel

refs = HfApi().list_repo_refs("SZLHOLDINGS/szl-maskmod", repo_type="kernel")
pin = next(b.target_commit for b in refs.branches if b.name == "v1")
km = get_kernel("SZLHOLDINGS/szl-maskmod", revision=pin, trust_remote_code=["SZLHOLDINGS/szl-maskmod"])
```

Record `pin` (a commit of this kernel repository) in your lockfile.
`SZL_KERNEL_SOURCE.json` in this repository names the GitHub commit each build came from.
Commits of the model repository
[huggingface.co/SZLHOLDINGS/szl-maskmod](https://huggingface.co/SZLHOLDINGS/szl-maskmod)
are not kernel revisions and do not resolve here.

## Use

```python
import torch

q = k = v = torch.randn(1, 2, 8, 16)
chain = km.ReceiptChain()
keep = km.create_dense_mask(8, 8, causal=True, sliding_window=4)
y = km.maskmod_attn(q, k, v, block_mask=keep, chain=chain)
print(km.selfcheck())
```

Each receipt binds the block-mask topology (shape and sha256 of the keep-mask), the
score_mod name and the causal flag into a SHA3-256 hash chain. Correctness is checked
against SDPA on CPU; no CUDA benchmark, tokens/s or energy figure is claimed.

Artwork: SZL Orbit · © 2026 SZL Holdings / Stephen Lutar · [CC BY 4.0](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/LICENSE) · [Trademark notice](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/NOTICE).
