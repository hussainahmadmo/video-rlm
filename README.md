# Conductor — technical contribution evidence for counsel

Prepared for review on October 29, 2026. Contributor recorded in the original local Git history: **Hussain Ahmad Mohammad**, hussainahmad@heart.cs.uchicago.edu.

Conductor investigates service-aware preparation scheduling for multimodal language-model serving. Preparation work occurs before a request becomes ready for model inference. The selected work covers tenant scheduling, CPU/GPU preparation placement, and resource accounting.

## Start here

- `docs/joint_preparation_allocation.md`: technical design, accounting units, and limitations.
- `docs/cpu_gpu_placement_l40s_multiseed_20260920.md`: recorded three-seed placement experiment, configuration, results, and claim boundaries.
- `conductor/experiments/scripts/run/joint_preparation_allocation.py`: implementation snapshot.
- `analysis/test_joint_preparation_allocation.py`: 42 allocator tests. From this repository root: `python -m unittest discover -s analysis -p test_joint_preparation_allocation.py`.

## Original contribution records

Original commit metadata and email-format patches are in `evidence/`. These preserve the original author, date, message, and changes as recorded in the local repository:

| Original commit | Contribution |
|---|---|
| a019e51 | Bounded-lag efficient fairness policy, tests, and evaluation support |
| 26f15be | Work-conserving GPU preparation lane borrowing and tests |
| 19099d9 | Learned resource-aware configuration selector and training support |
| b7b2185 | Repeated CPU/GPU placement evaluation and analysis |

The learned selector is a separate research component; its patch does not establish that it was used in the placement experiment.

## Provenance and limits

This is a small review snapshot on a separate branch with independent history. Its preparation date is not the original development date. `evidence/original-commits.txt` records original local commit identifiers and dates; `source-manifest.json` records the source snapshot and file hashes. Git author metadata is self-reported and is not independent verification of authorship. Original full history remains in the working repository.

The experiment report is an existing project record, not a new benchmark or independent reproduction. Full raw measurements, datasets, model weights, manuscript, and experimental archives are omitted. Source patches and example code are evidence excerpts, not a complete runnable serving system.

Claims are limited to preparation-stage behavior. Do not infer full-system GPU fairness, a formal fairness guarantee, a publication acceptance, external adoption, independent recognition, or an adaptive-routing advantage over the calibrated static split from this packet. Counsel can request underlying experiment artifacts, collaborator confirmation, publications, citations, or usage evidence separately.
