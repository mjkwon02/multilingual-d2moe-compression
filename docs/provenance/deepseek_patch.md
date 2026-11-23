# DeepSeek modeling patch provenance

**File**: `patches/deepseek_sdpa_fallback.patch`

- **Base**: Hugging Face model `deepseek-ai/deepseek-moe-16b-base`,
  revision `521d2bc4fb69a3f3ae565310fcc3b65f97af2580`, file
  `modeling_deepseek.py`.
- **Base file SHA-256**:
  `c21ae68b8466d020cdfaa3a27acbd51acfe6664d8fc1d112902540b964816158`
- **Purpose**: the upstream file imports `flash_attn` unconditionally when
  `is_flash_attn_2_available()` is true. On the experiment hosts flash-attn
  was not installable, so the import is wrapped in `try/except ImportError`,
  letting transformers fall back to SDPA attention. No numerical or
  architectural change; attention math is transformers' SDPA path.
- **How to apply**:
  ```bash
  huggingface-cli download deepseek-ai/deepseek-moe-16b-base \
      --revision 521d2bc4fb69a3f3ae565310fcc3b65f97af2580
  patch <path-to>/modeling_deepseek.py < patches/deepseek_sdpa_fallback.patch
  # If trust_remote_code has already cached the module, patch the copy under
  # $HF_HOME/modules/transformers_modules/deepseek-ai/deepseek-moe-16b-base/<rev>/ too.
  ```
- **Used in every DeepSeek run of the paper.** The two on-disk patched copies
  used during the experiments were byte-identical; the patch was verified
  (2026-09-28) to apply cleanly to the pinned base revision and to reproduce
  the exact file that ran (checksums in the private provenance archive).
