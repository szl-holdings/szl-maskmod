# szl-maskmod

Canonical GitHub source for `SZLHOLDINGS/szl-maskmod`.
GitHub bytes are the artifact. Hugging Face Hub is a publish mirror.

Set `SZL_MASKMOD_HF_REVISION` to the immutable **first-class Kernel Hub** commit
from a verified publication of [`kernels/SZLHOLDINGS/szl-maskmod`](https://huggingface.co/kernels/SZLHOLDINGS/szl-maskmod). Use the `kernels`
client version qualified with that publication. The GitHub source commit,
model-type mirror commit, and Kernel Hub commit are separate identities.
An observed head, a branch name, or a successful import does not qualify a release.

`trust_remote_code=True` permits execution of the selected repository's Python.
Review that exact revision, its provenance and publication evidence before enabling it.
The format check below only rejects missing or mutable revision inputs; it does not
verify hashes, publisher authorization or compatibility. If that evidence is unavailable,
stop the Hub load and use separately reviewed local source for development.

```python
import os
import re

hf_revision = os.environ.get("SZL_MASKMOD_HF_REVISION", "")
if re.fullmatch(r"[0-9a-f]{40}", hf_revision) is None:
    raise ValueError("A verified immutable Kernel Hub revision is required")

from kernels import get_kernel

km = get_kernel("SZLHOLDINGS/szl-maskmod", revision=hf_revision, trust_remote_code=True)
```

## Source-only development

Review [`torch-ext/szl_maskmod/`](https://github.com/szl-holdings/szl-maskmod/tree/5ad7c3842b15f77486e3a9bfef62c7824543cac8/torch-ext/szl_maskmod)
at that immutable GitHub source revision, separately from any Hub release.
With the source's dependencies already available, run from the reviewed checkout root:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("torch-ext").resolve()))
import szl_maskmod as local_kernel
```

This selects local Python source rather than calling the Hub loader. Importing local
source also executes Python. This documentation check does not run that import,
install dependencies, qualify a runtime or establish a Hub publication.

Local API example:

```python
from szl_maskmod import maskmod_attn, ReceiptChain, selfcheck
import torch
q = k = v = torch.randn(1, 2, 8, 16)
y = maskmod_attn(q, k, v, causal=True)
print(selfcheck())
```

Not a FlexAttention rehost. No CUDA benches. No speedup claim.
Doctrine v11. Λ = Conjecture 1 OPEN.
