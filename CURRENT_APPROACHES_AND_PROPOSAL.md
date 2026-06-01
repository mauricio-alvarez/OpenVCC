# Current SHViT Training Approaches and Proposed Next Technique

## Scope

This note documents the current project state around `main.py`, `SHViT.py`, `train_model.py`, and `train_model.sh`, then proposes a new training technique for the SHViT branch architecture. The goal is to improve ImageNet-1K/ImageNet-R performance without requiring local heavy compute; full runs should still happen on the HPC cluster.

## Current Entry Point

`train_model.sh` submits a single-GPU Slurm job and runs:

```bash
python main.py
```

At the moment, `main.py` is configured primarily for evaluation, not training:

- It hard-codes cluster paths for ImageNet, SHViT checkpoints, and trained monster checkpoints.
- It sets `NUM_CLASS = 1000`.
- It loads a double-head checkpoint through `load_finetuned_monster(...)`.
- It runs `test_imagenet_r(model)` and `test_imagenet_1k(model, set_name='val')`.
- The active `data_loader(...)` and `train_monster_improved(...)` call are commented/triple-quoted, so the Slurm script will not train unless `main.py` is edited.

## Base SHViT

`SHViT.py` contains the base SHViT family:

- A convolutional patch embedding stem.
- Three sequential stages: `blocks1`, `blocks2`, `blocks3`.
- `BasicBlock` uses depthwise convolution, optional single-head self-attention (`SHSA`), and a feed-forward block.
- `SHViT_s1`, `SHViT_s2`, `SHViT_s3`, and `SHViT_s4` define progressively larger embedding/depth configurations.
- The standard classifier is `BN_Linear(embed_dim[-1], num_classes)`.

## Current Architecture Variants

### DoubleHeadSHViT

The double-head architecture shares the early feature extractor and duplicates the later network:

- Shared: `patch_embed` and `blocks1`.
- Branch A: own stage-1-to-2 downsample, `blocks2_A`, stage-2-to-3 downsample, `blocks3_A`, `head_A`.
- Branch B: own stage-1-to-2 downsample, `blocks2_B`, stage-2-to-3 downsample, `blocks3_B`, `head_B`.
- Training forward returns `(logits_A, logits_B, feat_A, feat_B)`.
- Evaluation forward returns the average logits: `(logits_A + logits_B) / 2`.

Initialization copies the original pretrained SHViT stem/stage 1 into the shared trunk, copies stage 2/3 into both branches, copies the original classifier into both heads, and adds Gaussian noise to branch B stage 2/3 weights.

### TripleHeadSHViT

The triple-head architecture also shares `patch_embed` and `blocks1`, then creates three later-stage branches:

- Branches A/B/C each have independent downsampling, stage 2, and stage 3 modules.
- A router reads the shared stage-1 feature map and produces three softmax weights.
- The branch feature maps are fused as a weighted sum.
- A single classifier head predicts from the fused feature.

The current `TripleHeadSHViT.forward()` returns only logits. It computes `routing_weights` internally but does not return them to the training loop.

## Current Training Approaches

### Standard Head-Only Fine-Tuning

`train_model_timm(...)` and `train_model_base(...)` provide conventional head-only fine-tuning:

- Freeze the backbone.
- Train only the classifier head.
- Use cross-entropy.
- Use AdamW.
- Save the best validation checkpoint.

This is a conservative baseline but does not train the new branch capacity.

### `train_monster(...)`: Diversified Ensemble

This is the cleaner current double-head training path.

Behavior:

- Builds or loads a double/triple monster model.
- Freezes the shared stem and `blocks1`.
- Trains all remaining branch parameters.
- Uses AdamW with `lr=1e-4`, `weight_decay=0.05`.
- Uses cosine annealing over `NUM_EPOCHS`.
- Uses AMP via `NativeScaler`.

For double-head output, the loss is:

```text
CE(logits_A, y) + CE(logits_B, y) + 0.5 * mean(cosine(feat_A, feat_B)^2)
```

Inference averages the two branch logits.

Main limitation:

The diversity term forces pooled branch features toward orthogonality for every sample. That can create diversity, but it can also punish useful shared class information. The two branches start from nearly identical pretrained weights, and the objective asks them to both classify the same image while also globally disagreeing in representation space. This tension can easily waste capacity instead of creating complementary experts.

### `train_monster_improved(...)`: Progressive Unfreezing Attempt

This function tries to improve stability by:

- Freezing stage 2/3 and downsample modules for the first 3 epochs.
- Training only still-trainable parts first.
- Unfreezing stage 2/3 at epoch 3 with a lower learning rate.
- Reducing AdamW weight decay from `0.05` to `0.01`.
- Attempting a load-balancing loss for router-style outputs.

Current issue:

- For `DoubleHeadSHViT`, training forward returns four values, but `train_monster_improved(...)` treats any tuple as `(output, routing_weights)`. With the current code, double-head improved training should fail with a tuple-unpack error.
- For `TripleHeadSHViT`, the router balance loss is unreachable because `TripleHeadSHViT.forward()` returns only logits, not `(logits, routing_weights)`.
- So the intended load-balancing improvement is not active for the current triple-head code, and the current double-head path is incompatible with the improved loop.

## Additional Risks Found

- `main.py` is not parameterized, so training/eval mode, checkpoint paths, dataset paths, model size, epochs, and output names are controlled by manual edits.
- `train_model.sh` says "Starting image training", but the current `main.py` evaluates a loaded checkpoint.
- The checkpoint save condition in `train_monster(...)` and `train_monster_improved(...)` updates `best_acc` even on periodic saves. At epoch multiples of 20, a worse accuracy can overwrite the best accuracy tracker.
- Older router-format checkpoints loaded with `load_finetuned_monster(..., strict=False)` may leave `head_A`/`head_B` randomly initialized if their keys are missing.
- Augmentation is currently modest: `RandomResizedCrop` and horizontal flip only. There is no mixup, cutmix, RandAugment, repeated augmentation, random erasing, label smoothing, or teacher regularization in the monster path.
- Validation only reports top-1 for `validate(...)`; ImageNet-1K top-5 is reported only by `test_imagenet_1k(...)`.
- `test_imagenet_r(...)` writes confusion matrices during evaluation and assumes fixed cluster paths.
- `test_imagenet_r(...)` also forces CUDA with `.cuda()`, so it is not CPU-safe for local smoke tests.
- `get_imagenet_labels(...)` references `requests` and `json`, but those imports are not present in `train_model.py`; the helper will fail if called.

## Observed S1 Failure From Logs

The provided SHViT-S1 and double-head SHViT-S1 logs show that the double-head model did not simply fail to improve; it became worse on both in-distribution and robustness-oriented evaluation.

| Metric | SHViT-S1 | DoubleHead SHViT-S1 | Change |
| --- | ---: | ---: | ---: |
| ImageNet-200 top-1 | 89.56 | 88.37 | -1.19 |
| ImageNet-R top-1 | 35.39 | 30.11 | -5.28 |
| Robustness gap | 54.17 | 58.26 | +4.09 worse |
| ImageNet-R MSP AURRA | 63.53 | 57.11 | -6.42 |

This pattern suggests the double-head model mostly preserved familiar ImageNet-like recognition but damaged the representation needed for distribution shift. The larger ImageNet-R drop is the key signal: the extra branch capacity did not become complementary robustness capacity.

Likely causes:

- The two branches were initialized almost identically from the same pretrained stage 2/3 weights, then branch B received small random noise. That is symmetry breaking, but not a meaningful specialization signal.
- The diversity loss penalizes feature cosine similarity on every image. This can push branches away from shared class semantics that both branches actually need, especially for difficult or stylized images.
- The loss optimizes each branch independently but inference averages logits. The ensemble prediction itself is not directly optimized in `train_monster(...)`.
- There is no frozen teacher anchoring the branches to the original SHViT decision function, so fine-tuning can drift away from the pretrained model.
- The current augmentation is too weak to create robust complementary experts.
- `train_monster_improved(...)` is not a valid double-head fix in the current code because the double-head forward returns four values while the loop expects a two-value router output.

## Fast Go/No-Go Tests Before Full Training

Do not wait 90 to 100 epochs to decide if the next technique is promising. The next HPC workflow should use staged gates:

1. Zero-epoch check:
   - Build the double-head model from SHViT-S1.
   - Evaluate before training.
   - Expected result: ensemble accuracy should be close to original SHViT-S1. If it is already far below SHViT-S1, initialization/loading is wrong.

2. One-epoch overfit check on a small stratified subset:
   - Use 1 percent to 5 percent of ImageNet train.
   - Train only heads.
   - Track branch A, branch B, and ensemble accuracy on a small validation subset.
   - Expected result: ensemble should not be worse than both branches, and both branches should learn.

3. Five-epoch smoke run:
   - Use the full validation set but only 5 training epochs.
   - Freeze stem and `blocks1`.
   - Train heads plus optionally stage 3.
   - Expected result: ImageNet-200 should stay within about 1 point of the teacher/original, and ImageNet-R should not drop more than about 1 to 2 points.

4. Ten-to-fifteen-epoch proxy run:
   - Use stronger augmentation and teacher distillation.
   - Evaluate ImageNet-200, ImageNet-R, branch disagreement, and feature cosine each epoch.
   - Continue to a full 90-epoch run only if ImageNet-R is trending upward or at least not degrading while ImageNet-200 remains stable.

Recommended early kill criteria:

- Stop if ImageNet-R is more than 3 points below SHViT-S1 after 5 to 10 epochs.
- Stop if ImageNet-200 drops more than 2 points and does not recover by epoch 10.
- Stop if branch A and branch B predictions agree on almost everything and feature cosine remains high; the branches are not specializing.
- Stop if one branch is much weaker than the other; the ensemble is hiding a failed branch.

## Proposed Next Technique: Teacher-Anchored Branch Specialization

The next experiment should not push branches apart blindly. Instead, it should keep the original SHViT as a frozen teacher and train the duplicated branches to become complementary while staying anchored to the pretrained decision function.

### Core Idea

Use the original pretrained SHViT as a frozen teacher. Train the double-head model with:

1. Ensemble cross-entropy, so the averaged prediction is directly optimized.
2. Per-branch cross-entropy, so each branch remains independently useful.
3. Teacher distillation, so branch specialization does not drift away from the pretrained model.
4. Confidence-gated diversity, so branches diversify mainly on hard/ambiguous examples instead of being forced orthogonal everywhere.
5. Branch dropout, so inference averaging is not hiding weak branches.

### Suggested Loss

For input `x`, label `y`, frozen teacher logits `t`, branch logits `a` and `b`, and features `feat_A`, `feat_B`:

```text
ensemble_logits = (a + b) / 2

loss =
    CE(ensemble_logits, y)
  + 0.5 * (CE(a, y) + CE(b, y))
  + kd_weight * T^2 * 0.5 * (KL(a/T, t/T) + KL(b/T, t/T))
  + div_weight * hard_gate * margin_feature_diversity(feat_A, feat_B)
```

Where:

- `T = 2` or `3`.
- `kd_weight` starts around `0.5` and decays after warm-up.
- `hard_gate = 1 - max_softmax(teacher_logits)` or a thresholded version of it.
- `margin_feature_diversity = relu(cosine(feat_A, feat_B) - margin)^2`, with `margin` around `0.2` to `0.4`.

This changes the diversity objective from "always be orthogonal" to "avoid collapsing into identical branches, especially when the teacher is uncertain."

### Training Schedule

Recommended HPC schedule for `shvit_s1` first:

1. Baseline eval:
   - Evaluate original SHViT checkpoint on ImageNet-1K val and ImageNet-R.
   - Evaluate the existing double-head checkpoint the same way.

2. Warm-up, 5 epochs:
   - Freeze shared stem and `blocks1`.
   - Freeze branch stage 2/3.
   - Train `head_A` and `head_B`.
   - Use `lr_head=3e-4`, `weight_decay=0.01`, label smoothing `0.1`.

3. Stage-3 unfreeze, 20 to 40 epochs:
   - Unfreeze `blocks3_A/B` and `ds_2_to_3_A/B`.
   - Use discriminative LR: head `1e-4`, stage 3 `3e-5`.
   - Enable teacher distillation and branch dropout.

4. Stage-2 unfreeze, 50 to 100 epochs:
   - Unfreeze `blocks2_A/B` and `ds_1_to_2_A/B`.
   - Use LR: head `5e-5`, stage 3 `1e-5`, stage 2 `5e-6`.
   - Keep the stem and `blocks1` frozen for the first run.

5. Optional final calibration, 5 epochs:
   - Disable strong augmentation.
   - Lower LR by 10x.
   - Train with CE + KD only.

### Augmentation

Use stronger but standard ImageNet fine-tuning augmentation:

- RandAugment or TrivialAugment.
- Mixup `0.1` to `0.2`.
- CutMix `0.5` to `1.0`.
- Random erasing `0.1`.
- Label smoothing `0.1`.

For branch specialization, branch A and branch B can receive different stochastic augmentations of the same image, while the ensemble loss can be computed on aligned logits. This encourages robustness without requiring the branches to abandon the teacher.

### Code Changes Needed

Minimal implementation plan:

- Add a `return_features` or `return_branch_outputs` argument to `DoubleHeadSHViT.forward(...)` so the training loop does not depend on `model.training`.
- Add a `return_router` option to `TripleHeadSHViT.forward(...)` if the router path is kept.
- Add `train_monster_teacher_anchored(...)` rather than patching `train_monster_improved(...)` in place.
- Fix checkpoint best-accuracy tracking so periodic saves do not overwrite `best_acc`.
- Parameterize `main.py` with argparse: mode, model size, checkpoint path, dataset path, epochs, batch size, output file, resume path.
- Update `train_model.sh` to pass arguments explicitly.

### Metrics to Track

Each HPC run should log:

- ImageNet-1K top-1/top-5.
- ImageNet-R top-1.
- ImageNet-200 subset accuracy used inside `test_imagenet_r(...)`.
- Robustness gap: ImageNet-200 minus ImageNet-R.
- Branch A top-1, Branch B top-1, ensemble top-1.
- Mean branch disagreement.
- Mean cosine similarity between branch features.
- ECE/RMSCE from MSP.

The branch metrics are important. If both branches are weak but the ensemble looks okay, the architecture is fragile. If both branches are strong and disagree selectively, the new technique is doing what we want.

## Recommendation

Start with `shvit_s1` double-head only. The triple-head router path needs a cleaner forward contract before it is worth more compute. A teacher-anchored double-head run is the best next use of HPC time because it directly addresses the main failure mode in the current approach: unconstrained diversification from two nearly identical pretrained branches.
