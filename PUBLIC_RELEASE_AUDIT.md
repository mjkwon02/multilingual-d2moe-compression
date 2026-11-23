# Public-release audit — 2026-11-23

Pre-publication audit of this repository (security, personal data, licenses,
README reproducibility) performed before switching the GitHub repository
`mjkwon02/multilingual-d2moe-compression` from Private to Public.
No push, no visibility change, and no remote-configuration change were made
during this audit.

## A. Executive summary

**Verdict: PUBLIC READY** (within the audited scope) — no valid secrets, no
license violation, and no blocking disclosure risk remain. The initial
audit returned FIX REQUIRED with four items; all four were remediated the
same day (§H). The one outstanding mechanical step is outside this audit's
mandate: the amended root commit must be force-pushed to `origin/main`
(replacing `b8f1678`, which still contains the pre-fix `device_map.json`
blob) before the repository is switched to Public.

Scope limitation: the live GitHub remote could not be contacted from the
audit machine (no credentials here), so "pushed history" was audited via the
local remote-tracking ref `origin/main` = `b8f1678` recorded at push time.
Local history is a single root commit; if the remote has not been changed
outside this clone, remote and local trees are identical.

## B. Git security audit

**Examined:** branch `main` (sole branch), remote-tracking `origin/main`,
no tags, no stash. Entire history = 1 commit
(`b8f16787207403b57944b70354d2e38a3dae8db5`, 2026-11-23, 1,064 files), so
"history" and "current tree" coincide except for uncommitted working-tree
edits (README rewrite + this audit's two fixes).

- **Secrets:** pattern scan (HF/GitHub/OpenAI/AWS/Google/Slack tokens, JWT,
  private-key blocks) and credential-keyword scan (`password`, `api_key`,
  `auth_token`, …) over the committed tree and working tree: **0 hits**.
  No `.env`, key, certificate, netrc, or notebook files are tracked.
  `gitleaks`/`trufflehog` were not available on the audit machine; scanning
  was regex + manual context review (noted as a residual risk, §F).
- **Personal data:** no emails, no local absolute paths, no phone numbers.
  IP-looking strings are all `nvidia-*-cu12` package versions in
  `requirements/`. **One finding:** `configs/deepseek/device_map.json`
  contained two lab members' account usernames in a GPU-sharing note.
  Fixed in the working tree (usernames removed, technical content kept).
  The old text **remains in the blob of commit `b8f1678`**, which was
  pushed to the private remote — see §G item 1.
- **Large/third-party blobs in history:** largest blobs are
  `agg_covariance_spectra.npz` (24 MB) and
  `massive_revised/predictions.jsonl` (13 MB), both first-party result
  data; nothing above GitHub limits; no checkpoints, no model weights, no
  deleted-then-hidden files (single commit).
- **Not examined:** any state on the GitHub server beyond the local
  tracking ref (branches/objects created outside this clone, if any).

## C. README validation

The current root `README.md` (working-tree version, uncommitted) was the
reference. All checks run from the repository root on Linux,
Python 3.13.9.

| README item | Actual target | Check method | Result |
|---|---|---|---|
| Install: `conda env create -f requirements/deepseek_conda_env.yml` | file exists, tracked | YAML parse (env not created) | PASS (static) |
| `bash tests/smoke_test.sh` | 30 scripts + numeric spot-check | executed (37 s, CPU-only) | **PASS** — "SMOKE TEST PASSED", headline own-gain spot-check OK |
| Regenerate: `phase1_41_headline_gate.py` | exists | executed via smoke test | PASS |
| Regenerate: `qwen15_table1_aggregate.py --seeds 0 1 2` | exists | executed | PASS |
| Regenerate: `_make_table2_accuracy.py` (SIB-200) | exists | executed | PASS |
| Regenerate: `qwen15_sib200_clean_aggregate.py` | exists | executed | PASS |
| Regenerate: `_make_belebele_clean.py` | exists | executed | PASS |
| Regenerate: `massive_revised_aggregate.py` | exists | executed | PASS |
| Documentation links (10 relative links) | all exist, all git-tracked | script check | PASS |
| Repository-map directories | all 14 entries exist | listing | PASS |
| All tracked `*.py` | 100% | `py_compile` | PASS |
| All tracked `*.sh` | 100% | `bash -n` | PASS |
| `CITATION.cff` | valid YAML, 4 authors, no placeholders/TODO/DOI stubs | parse + grep | PASS |

Regeneration fidelity: after the smoke test, **every tracked table/JSON/CSV
/MD/PNG output was byte-identical** to the committed version; only the 5
PDF figures differed (embedded creation timestamps) and were restored with
`git checkout`. Gitignored timestamped snapshot triplets produced by
`qwen15_table1_aggregate.py` were removed after confirming the CSV/MD
copies byte-identical to the canonical files (the JSON copy differs only
in its generation-timestamp field).

GPU-path (B-type) reproduction — data preparation, calibration, Fisher,
whitening/SVD, merging, model evaluation — was **not executed** (per audit
rules). README correctly separates it from saved-result regeneration and
correctly states that checkpoints/intermediate tensors are not included
and that an external D²-MoE checkout (`D2MOE_DIR`) is required.

## D. Paper–repository consistency

The camera-ready `.tex` is not in this tree; consistency was checked
against the repository's own frozen records
(`docs/PAPER_FIGURE_TABLE_MAPPING.md`, `docs/EXPERIMENT_MAPPING.md`,
`docs/limitations.md`, `docs/data_exposure_audit.md`) and the stored
result files — noted as a limitation in §F.

- **Key finding 01** (own-language first, Balanced second, both models):
  verified numerically — DeepSeek `phase1_41_headline_gate_result.json`
  own-gains positive on all 5 languages (5.963/60.784/10.759/35.551/67.850)
  and the Qwen Table 1 grid shows the own-language cell smallest and
  Balanced second-smallest in **every** column. PASS.
- **Key finding 02** (whitening > Fisher, but below the preregistered
  dominance threshold): wording matches the repository's prereg records;
  the README deliberately does **not** claim 3× dominance, consistent with
  the documented ratios. PASS.
- **SIB-200:** "98 calibration-unexposed items" = 98 of 204 in
  `data_exposure_audit.md`; "6 of 14 demonstrations overlap" confirmed in
  both `limitations.md` and `data_exposure_audit.md`; README correctly
  calls it *test-item-filtered*, not exposure-free. PASS.
- **MASSIVE:** README "zero-shot, 60-way, English-only vs Balanced"
  matches `protocol.json`/`summary.json` (prompt P0 zero-shot, 60-intent
  label set, 200 items/language, 35 cells = baseline + 2 conditions × 3
  seeds × 5 languages). `data/splits/massive_demo_ids.json` serves the
  protocol's P8 *fallback* path only; the reported run is P0. PASS.
- **Qwen scope:** README limits Qwen to the FLORES grid + exposure-filtered
  SIB-200 and explicitly disclaims mechanism analysis on Qwen — matches
  the result tree. PASS.
- No venue/acceptance claims are made anywhere; none were added. PASS.

## E. Changes made (this audit)

1. `configs/deepseek/device_map.json` — removed two third-party account
   usernames from a free-text note; technical meaning preserved.
2. `docs/LICENSES.md` — corrected two statements to disclose the one piece
   of third-party text in `data/`: the 8 MASSIVE English demonstration
   utterances in `data/splits/massive_demo_ids.json` (MASSIVE is CC BY 4.0;
   redistribution with attribution is permitted — this was a documentation
   inaccuracy, not a license violation).
3. Removed 3 regenerated, gitignored snapshot files; restored 5 regenerated
   PDF figures to their committed bytes. No result file was altered.

No history rewrite, no push, no remote change, no license change.

## F. Remaining risks

- Secret scan was regex-based; no entropy-based scanner (`gitleaks`,
  `trufflehog`) was available. Running one before the switch is cheap
  insurance.
- Paper consistency was verified against the repository's frozen mapping
  documents, not the camera-ready PDF/`.tex` itself.
- The live GitHub remote state was not directly inspected from this
  machine (no credentials); verify on github.com that `main` is at the
  final commit before switching visibility.
- `manifests/verification_report.md` intentionally references
  `ARR_8/PUBLIC_REPOSITORY_CLEANUP_REPORT.md` and
  `ARR_8/private_release_archive/` — paths **outside** this repository
  (private provenance). Readers of the public repo cannot resolve them;
  this is disclosed in that document and is informational only.
- Natural-language sweep covered all tracked JSON/JSONL string fields
  (>300 chars) and sampled JSONL records; a byte-level review of every
  CSV cell was not performed.

## G. Final verdict

**PUBLIC READY** within the audited scope. The four FIX REQUIRED items
from the initial pass are resolved (§H). Before switching visibility,
the repository owner must still (outside this audit's mandate):

1. Force-push the amended root commit to `origin/main` once, replacing
   `b8f1678` (whose `device_map.json` blob contains the removed
   usernames). Until then, the private remote retains the old blob.
2. Verify on github.com that `main` shows the amended commit and the
   expected file count, then switch visibility.

## H. Remediation addendum (2026-11-23, after the initial audit)

All four §G items from the initial FIX REQUIRED verdict were remediated
with user approval; the approval explicitly covered rewriting the root
commit locally and explicitly deferred the force-push.

1. **History cleanup.** Preconditions re-verified before the rewrite:
   history is a single root commit; the only refs are `main` and the
   remote-tracking `origin/main`; no tags, no stash, one author. (The
   GitHub-side collaborator list could not be queried from this machine —
   the repository is private, created and pushed by the sole author the
   same day.) The root commit was amended to the fixed tree. Post-rewrite
   scan: no ref that will be pushed reaches the old blob; the usernames
   appear in **zero** objects reachable from `main`. The old objects
   remain locally only via the `origin/main` tracking ref and reflog, and
   on the remote, until the force-push (deliberately not performed).
2. **Working-tree changes committed** into the amended root commit: the
   rewritten `README.md`, the `device_map.json` username removal, the
   `docs/LICENSES.md` MASSIVE-demonstration disclosure, this report, and
   the `.gitignore` exception below. No experiment code or result file
   was modified.
3. **Manifest refreshed.** `manifests/public_tree_checksums.sha256`
   entries updated only for the files changed by this remediation
   (`README.md`, `configs/deepseek/device_map.json`, `docs/LICENSES.md`)
   plus a new entry for `PUBLIC_RELEASE_AUDIT.md`; the other verified
   entries are byte-for-byte untouched. Full `sha256sum -c` re-run: all
   entries pass, and the manifest's file list now equals the git-tracked
   file list exactly (sole exception: the manifest cannot list itself).
4. **`_layer_meta` included.** The 432
   `results/qwen/phase1/_layer_meta/**/svd_scale_layers/layer_*.json`
   files (~18 MB) were verified safe before inclusion: uniform schema
   (`min_eig`, `max_eig`, `clamped`, `n_eig_raised`, `rows`, `trace_raw`),
   values exclusively bool/float/int — **no string values at all**, hence
   no possible PII, credentials, or dataset text. No release script
   consumes them; they are included as the only preserved per-layer
   provenance of the Qwen whitening stage's numerical health, and they
   were already counted in the checksum manifest's public tree. The
   `.gitignore` blanket `svd_scale_layers/` rule (which continues to
   block the large SVD tensors everywhere else — re-verified) gained one
   scoped exception for this metadata path.

Final re-audit after the rewrite: secrets/PII scans clean on the new
HEAD; README links/commands re-verified; CPU-only smoke test re-run and
passed; manifest verification 100%. The new root commit hash and final
file count are recorded in the git history itself.

## I. Publication-wording finalization (2026-11-24)

Before the Public switch, all unconfirmed publication-status wording was
removed, per the owner's instruction:

- `README.md` intro set to "Code and experimental results for …" (no
  acceptance/camera-ready-status claims; none existed elsewhere in it).
- `docs/PAPER_FIGURE_TABLE_MAPPING.md` header: the unconfirmed venue tag
  was removed; the factual audit date against the final PDF is kept.
- `CITATION.cff`: unchanged — it contains only verified author/title
  metadata and no venue information.
- The only remaining "EACL" string in the tree is the SIB-200 dataset
  citation *(Adelani et al., EACL 2024)* in
  `scripts/evaluation/phase1_sib200_eval.py`, which is a third-party
  reference, not a claim about this paper.

Manifest entries for the files touched above were refreshed and the full
manifest re-verified; links and the smoke test re-checked; the root
commit re-amended (history remains a single root commit).
