# Representation Analysis Plan: SHViT-S1 vs DoubleHead SHViT-S1

## Goal

The goal is to explain why SHViT-S1 performs better than DoubleHead SHViT-S1 (DHViT-S1), especially on ImageNet-R. The thesis claim should not be only "DHViT has lower accuracy"; it should show that DHViT changes or damages the learned representation in a way that reduces class separability and robustness under distribution shift.

The analysis should answer four questions:

1. Are the early representations actually preserved in DHViT?
2. Do the two DHViT branches learn meaningfully different representations?
3. Is the DHViT final representation more or less class-separable than SHViT?
4. Does the representation difference explain the ImageNet-R drop?

## Main Hypothesis

SHViT-S1 is better because its final representation remains more class-discriminative and more stable under distribution shift. DHViT-S1 duplicates later layers, but the training objective pushes branches apart without a useful specialization signal. The result is representational divergence without improved class structure.

There is also a possible early-layer issue: `freeze_stage_1()` freezes parameters but does not freeze BatchNorm running statistics. During `train_monster()`, `model.train()` can still update BatchNorm buffers in the shared stem and `blocks1`. So DHViT may not preserve early SHViT features as cleanly as intended.

## Models To Compare

Use exactly these models for the core comparison:

- Original pretrained/evaluated `SHViT-S1`.
- Trained `DHViT-S1` checkpoint, for example `122_shvit_s1_doublehead_1805_100epochs.pth`.

For DHViT, analyze:

- branch A alone,
- branch B alone,
- ensemble average,
- branch A vs branch B similarity.

Do not only analyze the ensemble output. A weak branch can be hidden by averaging, and the thesis needs to know whether each branch learned a valid representation.

## Datasets

Use three dataset subsets with the same preprocessing (`Resize(256)`, `CenterCrop(224)`, ImageNet normalization):

1. ImageNet-1K validation sample:
   - 50 to 100 images per selected class.
   - Used to test in-distribution class structure.

2. ImageNet-200 subset:
   - The ImageNet validation subset corresponding to ImageNet-R classes.
   - Used as the in-distribution counterpart for ImageNet-R.

3. ImageNet-R:
   - Same class set as ImageNet-200.
   - Used to test robustness under rendition/style shift.

For visualization, start with 16 classes to keep plots readable. For quantitative metrics, use all 200 ImageNet-R classes if compute allows, or at least 50 well-sampled classes.

## Layers To Extract

Use matched layers where possible.

### SHViT-S1 Layers

Extract activations from:

- `patch_embed`: earliest convolutional representation.
- `blocks1`: final shared early representation.
- `blocks2`: middle representation.
- `blocks3`: final spatial representation.
- pooled pre-head feature: `adaptive_avg_pool2d(blocks3_output, 1).flatten(1)`.
- logits: classifier output.

### DHViT-S1 Layers

Extract activations from:

- `patch_embed`: shared early representation.
- `blocks1`: shared early representation.
- Branch A:
  - `blocks2_A`
  - `blocks3_A`
  - pooled `feat_A`
  - `logits_A`
- Branch B:
  - `blocks2_B`
  - `blocks3_B`
  - pooled `feat_B`
  - `logits_B`
- ensemble logits:
  - `(logits_A + logits_B) / 2`

For early-layer comparison, `patch_embed` and `blocks1` should be very close between SHViT and DHViT if the frozen trunk was preserved. If they differ substantially, inspect BatchNorm running statistics.

## Analysis Metrics

### 1. Activation Similarity Between SHViT And DHViT

For the same images, compute:

- Mean squared error between normalized activations.
- Cosine similarity of pooled vectors.
- Linear CKA between activation matrices.

Run this for:

- SHViT `patch_embed` vs DHViT `patch_embed`.
- SHViT `blocks1` vs DHViT `blocks1`.
- SHViT `blocks3` pooled feature vs DHViT `feat_A`.
- SHViT `blocks3` pooled feature vs DHViT `feat_B`.
- SHViT `blocks3` pooled feature vs DHViT ensemble feature if an average feature is computed.

Expected useful thesis result:

- Early layers should be nearly identical if freezing worked.
- Last-layer DHViT features should diverge from SHViT.
- If the divergence is stronger on ImageNet-R than ImageNet-200, it supports the robustness-failure explanation.

### 2. Branch Similarity Inside DHViT

For branch A and branch B, compute:

- `CKA(feat_A, feat_B)`.
- Mean cosine similarity between `feat_A` and `feat_B`.
- Prediction agreement rate.
- Disagreement accuracy:
  - when branches agree, how often are they correct?
  - when branches disagree, how often is either branch correct?
  - how often does the ensemble fix a branch error?

Expected useful thesis result:

- If branches have high similarity and high prediction agreement, DHViT did not learn meaningful specialization.
- If branches are dissimilar but both are less class-separable, diversity hurt useful semantics.
- If one branch is consistently worse, the model learned an unbalanced ensemble.

### 3. Class Separability

For each layer/model/dataset, compute:

- Silhouette score using true labels.
- Nearest-centroid classification accuracy in feature space.
- k-NN accuracy, for example k=5 or k=20.
- Linear probe accuracy with frozen features.
- Fisher ratio:
  - between-class centroid variance divided by within-class variance.
- Mean intra-class distance.
- Mean inter-class centroid distance.

Run this for:

- SHViT pooled final features.
- DHViT `feat_A`.
- DHViT `feat_B`.
- Optional DHViT averaged feature `(feat_A + feat_B) / 2`.

Expected useful thesis result:

- SHViT should show better class compactness and/or larger inter-class separation than DHViT, especially on ImageNet-R.
- If DHViT has worse silhouette, worse k-NN, and worse linear probe accuracy, then the final representation is objectively less useful.

### 4. Distribution Shift Stability

For each class that exists in both ImageNet-200 and ImageNet-R:

- Compute the class centroid in ImageNet-200.
- Compute the class centroid in ImageNet-R.
- Measure centroid shift distance.
- Measure whether ImageNet-R samples remain closer to their own ImageNet-200 centroid than to other class centroids.

Metrics:

- Mean class centroid shift.
- Class centroid retrieval accuracy.
- Cross-domain nearest-centroid accuracy:
  - train centroids on ImageNet-200,
  - classify ImageNet-R features by nearest centroid.

Expected useful thesis result:

- SHViT should have smaller cross-domain centroid shift or better cross-domain nearest-centroid accuracy.
- DHViT may overfit ImageNet-like features while failing to keep the same class geometry under style/rendition shift.

### 5. Visualization

Use visualizations only as support, not as the main proof.

Generate:

- PCA plots for first-layer and final-layer features.
- UMAP plots for final features.
- Per-class centroid plots for ImageNet-200 vs ImageNet-R.
- Heatmap of pairwise class centroid distances.
- CKA matrix:
  - SHViT layers vs DHViT layers/branches.

Recommended plots for thesis:

1. Early-layer CKA bar chart:
   - shows whether early features are preserved.
2. Final-layer UMAP/PCA:
   - shows SHViT has cleaner clusters than DHViT.
3. Class separability table:
   - silhouette, k-NN, linear probe, Fisher ratio.
4. Cross-domain centroid shift plot:
   - shows SHViT representation is more stable from ImageNet-200 to ImageNet-R.
5. Branch disagreement plot:
   - shows whether DHViT branches specialize usefully or not.

## Execution Plan

Implemented files:

- `representation_extract.py`: loads SHViT or DHViT, runs eval-mode feature extraction, and saves `.npy` vectors.
- `analyze_representations.py`: computes CKA, branch similarity, separability, centroid shift, prediction agreement, and PCA plots.
- `imagenet_r_16_classes.txt`: default pilot class list.
- `run_representation_analysis.sh`: Slurm execution file that runs the full pilot pipeline.

### Phase 1: Make Activation Extraction Reliable

Use `representation_extract.py` instead of relying only on the current global VCC hooks.

The script should:

- Load SHViT-S1 and DHViT-S1 explicitly.
- Register forward hooks on named modules.
- Run both models in `eval()` mode.
- Save activations per model/layer/dataset.
- Save labels, file paths, and predictions.
- For DHViT, save branch A logits, branch B logits, ensemble logits, `feat_A`, and `feat_B`.

Output format:

```text
analysis/representation/shvit_s1/imagenet200/
  patch_embed.npy
  blocks1.npy
  blocks2.npy
  blocks3.npy
  pooled.npy
  logits.npy
  labels.npy
  paths.txt

analysis/representation/dhvit_s1/imagenet200/
  patch_embed.npy
  blocks1.npy
  blocks2_A.npy
  blocks2_B.npy
  blocks3_A.npy
  blocks3_B.npy
  feat_A.npy
  feat_B.npy
  logits_A.npy
  logits_B.npy
  logits_ensemble.npy
  labels.npy
  paths.txt
```

Use pooled vectors for large-scale metrics. Save raw spatial tensors only for small batches because they are large.

### Phase 2: Sanity Checks

Before running full analysis:

1. Run on one batch of 32 images.
2. Verify tensor shapes.
3. Verify top-1 accuracy from saved logits matches direct model evaluation.
4. Verify DHViT ensemble logits equal `(logits_A + logits_B) / 2`.
5. Compare early SHViT and DHViT activations:
   - if checkpoint loading is correct and BN buffers are close, early representations should be very similar.

If early layers differ unexpectedly, inspect:

- `patch_embed.*.bn.running_mean`
- `patch_embed.*.bn.running_var`
- `blocks1.*.bn.running_mean`
- `blocks1.*.bn.running_var`

This would support the argument that "frozen" layers were not fully frozen because BatchNorm buffers continued changing.

### Phase 3: Quantitative Representation Metrics

Use `analyze_representations.py` to load the saved `.npy` files and produce:

- CKA results.
- Branch similarity results.
- Class separability metrics.
- Cross-domain centroid shift metrics.
- Prediction disagreement metrics.

Save results as:

```text
analysis/representation/results/
  cka_shvit_vs_dhvit.csv
  branch_similarity.csv
  separability_imagenet200.csv
  separability_imagenetr.csv
  centroid_shift.csv
  prediction_disagreement.csv
```

### Phase 4: Plots

Generate publication/thesis-ready figures:

```text
analysis/representation/figures/
  early_layer_cka.png
  final_feature_umap_imagenet200.png
  final_feature_umap_imagenetr.png
  separability_metrics_barplot.png
  centroid_shift_by_model.png
  dhvit_branch_disagreement.png
  cka_matrix.png
```

Keep the visuals consistent:

- same selected classes,
- same colors across SHViT and DHViT,
- same random seed for PCA/UMAP,
- same number of samples per class.

### Phase 5: Thesis Interpretation

The final written argument should be structured like this:

1. Accuracy establishes the phenomenon:
   - DHViT is slightly worse on ImageNet-200 but much worse on ImageNet-R.

2. Early-layer analysis checks whether the shared trunk was preserved:
   - if preserved, the failure comes from later branch representation learning;
   - if not preserved, BatchNorm drift is part of the failure.

3. Branch analysis checks whether duplication created useful specialization:
   - high similarity means no specialization;
   - low similarity plus low separability means harmful specialization.

4. Final-layer separability explains performance:
   - SHViT has cleaner class geometry;
   - DHViT has weaker clustering or lower linear-probe/k-NN accuracy.

5. Cross-domain shift explains ImageNet-R:
   - SHViT class centroids remain more stable from ImageNet-200 to ImageNet-R;
   - DHViT representation shifts more, causing the robustness drop.

## Minimal HPC Job Strategy

Run activation extraction separately from training. It should be much cheaper than training.

Recommended first job:

- 16 classes,
- 50 ImageNet-200 images per class,
- 50 ImageNet-R images per class,
- layers: `patch_embed`, `blocks1`, final pooled features,
- both SHViT-S1 and DHViT-S1.

If that works, scale to:

- all 200 ImageNet-R classes,
- 50 to 100 images per class,
- all planned layers.

## Success Criteria For The Analysis

The analysis is thesis-ready if it can support at least one of these concrete conclusions:

1. DHViT did not learn new representations:
   - branches have high CKA/cosine similarity and high prediction agreement.

2. DHViT learned different but worse representations:
   - branches differ, but separability/k-NN/linear probe are worse than SHViT.

3. DHViT harmed robustness geometry:
   - cross-domain centroid shift is larger from ImageNet-200 to ImageNet-R.

4. DHViT altered supposedly frozen early features:
   - early-layer activation/BN-buffer differences are measurable.

The strongest thesis result would be:

> DHViT's additional branch capacity does create representational divergence, but the divergence is not aligned with class-discriminative or distribution-stable features. SHViT-S1 retains a more coherent final representation, which explains its stronger ImageNet-R performance.
