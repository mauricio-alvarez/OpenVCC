# Typhon Method Review

Date: 2026-06-09

Reviewed files:

- `SHViT.py`
- `train_model.py`
- `main.py`
- `test_typhon.py`
- `experiment_overfit.txt`
- `experiment_pretrain.txt`

## Executive Summary

Replacing the duplicated DHViT stage-2/stage-3 branches with Typhon's two attention mixers is a better architectural direction. It keeps one backbone path, avoids duplicating full late-stage blocks, and tries to introduce multi-head-like capacity inside the SHViT attention mixer itself.

The current implementation, however, is not equivalent to vanilla multi-head attention. It is closer to an ensemble of multiple single-head SHSA mixers that all attend over the same active channel subset, then sum their outputs. This is more efficient and cleaner than DHViT, but it does not yet guarantee useful head specialization.

The ImageNet-1K experiment supports this interpretation:

| Model | ImageNet-1K Top-1 | ImageNet-200 | ImageNet-R | Delta |
|---|---:|---:|---:|---:|
| SHViT-S1 baseline | not in this log | 89.56 | 35.39 | 54.17 |
| DHViT-S1 previous | 70.71 | 88.37 | 30.11 | 58.26 |
| Typhon-S1 current | 71.78 | 89.07 | 33.95 | 55.12 |

Typhon improves substantially over DHViT, especially on ImageNet-R, but it is still below SHViT-S1. That means the new idea is directionally better than duplicating whole stages, but the current block design and training recipe are not yet strong enough to beat the original model.

## Implemented Update

The code now implements the main recommendations from this review.

Changed files:

- `SHViT.py`
- `train_model.py`
- `main.py`

New Typhon options:

| Option | Meaning | Recommended use |
|---|---|---|
| `mixer_fusion="gated_sum"` | Every mixer receives the full active channel subset, but learnable gates start as `[1, 0, ...]`. | Default. Use this for the next serious ImageNet-1K run because it preserves the SHViT path at initialization. |
| `mixer_fusion="split_concat"` | The active channel subset is split across attention mixers, outputs are concatenated, and then projected. | Experimental. This is closer to MHA-style head separation, but it is not function-preserving. |
| `use_merge_norm=False` | Disables the extra post-sum/post-concat `GroupNorm`. | Default. Keeps the copied SHViT projection input distribution closer to the original. |
| `distill_weight` | Adds optional KL distillation from the original SHViT teacher during ImageNet-1K Typhon training. | Use `0.5` for the next run if memory allows. |

The current `main.py` run configuration now starts from the original SHViT checkpoint instead of resuming the old Typhon checkpoint:

```python
train_typhon_imagenet_1k(
    use_shvit=True,
    model_location="/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth",
    NUM_EPOCHS=12,
    output_file_name="typhon_s1_imagenet1k.pth",
    batch_size=256,
    num_mixers=2,
    resume_checkpoint=None,
    mixer_fusion="gated_sum",
    use_merge_norm=False,
    distill_weight=0.5,
    distill_temperature=2.0,
)
```

This is the safer default because the new architecture has different semantics from the old checkpoint. Loading the old checkpoint is still possible, but it should be treated as a compatibility path, not the main experiment.

## Current Typhon Architecture

Original SHViT uses `BasicBlock` with:

1. depthwise convolution residual
2. one `SHSA` mixer residual
3. FFN residual

The original `SHSA` does this:

1. split channels into active `x1` and passive `x2`
2. apply single-head self-attention only to `x1`
3. concatenate attended `x1` with unchanged `x2`
4. project back through `ReLU + Conv2d_BN`
5. return the projection to the residual wrapper

Typhon changes only the mixer part for `"s"` blocks:

1. split channels into active `x1` and passive `x2`
2. send the same `x1` into `num_mixers` independent `TyphonAttention` modules
3. sum all mixer outputs
4. apply `GroupNorm` to the summed active output
5. concatenate with the same passive `x2`
6. project once through `ReLU + Conv2d_BN`
7. add this projection back to the block input
8. apply the FFN residual

Code locations:

- `TyphonAttention`: `SHViT.py:188`
- `TyphonBasicBlock`: `SHViT.py:212`
- `Typhon`: `SHViT.py:318`
- `typhon_s1` to `typhon_s4`: `SHViT.py:491`

## Architectural Assessment

The decision to avoid full stage duplication is correct. DHViT duplicated stage 2, stage 3, and classifier heads, which created a large amount of extra capacity without a strong reason for semantic complementarity. The previous representation analysis showed that the two branches became geometrically different but not meaningfully complementary.

Typhon is more targeted: it adds diversity inside the attention mixer, where vanilla ViTs normally have multiple heads. This is the right place to experiment if the thesis argument is that SHViT's single-head attention is efficient but possibly under-expressive.

The original Typhon block had three important limitations.

First, every mixer saw the same active channel tensor `x1`. Vanilla multi-head attention usually partitions or projects into separate head subspaces. The implementation now supports `split_concat`, where each mixer receives a disjoint slice of `x1`. The default remains `gated_sum` because it is function-preserving.

Second, mixer outputs were summed unconditionally. The implementation now has two fusion choices: gated summation for safe SHViT initialization, and split concatenation for explicit head separation.

Third, initialization was not function-preserving. `build_and_load_typhon()` copied the original SHSA mixer into every Typhon mixer and added only `1e-4` noise to the later mixers. Then `TyphonBasicBlock` summed those outputs and applied a new `GroupNorm`. This has been changed for `gated_sum`: mixer 0 is copied from SHViT, additional mixers are copied with tiny noise, gates initialize as `[1, 0, ...]`, and merge normalization is disabled by default.

This matters because the goal is to improve a pretrained architecture, not train a new architecture from scratch. If the modified block does not initially behave like SHViT, ImageNet fine-tuning has to recover baseline behavior before it can improve it.

## Code-Level Findings

### 1. Stage-1 Freeze Still Allows BatchNorm Drift

`freeze_stage_1()` only sets `requires_grad=False` for `patch_embed` and `blocks1`. It does not keep their BatchNorm modules in eval mode. Since `train_typhon()` and `train_typhon_imagenet_1k()` call `model.train()` every epoch, frozen BatchNorm running statistics can still change.

This is the same class of issue found earlier with DHViT.

Implemented fix:

```python
def set_frozen_stage_eval(model):
    model.patch_embed.eval()
    model.blocks1.eval()
    for module in list(model.patch_embed.modules()) + list(model.blocks1.modules()):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
```

The training loops now call this immediately after every `model.train()` during Typhon fine-tuning.

### 2. Checkpoint Saving Can Overwrite the Best Accuracy

In `train_typhon_imagenet_1k()`, the save condition is:

```python
if acc > best_acc or epoch % 10 == 0:
    best_acc = acc
```

This incorrectly replaces `best_acc` even when the checkpoint is saved only because `epoch % 10 == 0`. In the log, epoch 5 reached 72.27%, but epoch 10 saved 71.62% and lowered `best_acc`. Epoch 11 then saved 71.77% as if it were a new best.

Implemented fix: Typhon training now updates `best_acc` only when `acc > best_acc`. Periodic checkpoints no longer lower the stored best score.

### 3. `typhon_s*` Ignores `pretrained=True`

The registered `typhon_s1` to `typhon_s4` constructors accept `pretrained`, but do not load any weights. This is acceptable if Typhon is always built through `build_and_load_typhon()`, but misleading if using `timm.create_model("typhon_s1", pretrained=True)`.

### 4. Mixer Symmetry Is Only Weakly Broken

The second mixer is initialized with the same weights as the first mixer plus `1e-4` noise. Since both mixers see the same input and their outputs are summed, gradients can remain highly correlated. This may produce two almost-identical mixers rather than two useful heads.

This should be measured directly with:

- cosine similarity between mixer outputs
- CKA between mixer outputs
- attention-map similarity
- per-mixer ablation accuracy

### 5. FashionMNIST Is a Weak Architecture Test

The FashionMNIST test confirms that the model can optimize and generalize somewhat, but it does not validate the ImageNet-R robustness hypothesis. FashionMNIST is grayscale, resized to 224, and much simpler than ImageNet-R. The full model is trained, so the test mostly checks implementation sanity.

## Experiment Analysis

### FashionMNIST Overfit Experiment

Typhon:

- train accuracy: 98.73%
- test accuracy: 92.96%
- ECE: 3.77%

DHViT:

- train accuracy: 99.46%
- test accuracy: 94.15%
- ECE: 3.39%

Interpretation:

Typhon learns, so the implementation is not broken. However, DHViT fits and generalizes better on this small benchmark. That does not mean DHViT is better for ImageNet-R; it only means FashionMNIST does not provide evidence that Typhon is stronger.

### ImageNet-1K Adaptation Experiment

The run resumed from epoch 4 and trained epochs 5 through 11.

Validation accuracy:

- epoch 5: 72.27%
- epoch 6: 72.18%
- epoch 7: 71.96%
- epoch 8: 71.76%
- epoch 9: 71.73%
- epoch 10: 71.62%
- epoch 11: 71.77%

The model peaked immediately at epoch 5, then gradually degraded. That pattern usually means the learning rate or trainable scope is too aggressive for a pretrained model adaptation. The model is not undertrained; it is drifting away from a useful pretrained solution.

Final evaluation:

- ImageNet-200: 89.07%
- ImageNet-R: 33.95%
- ImageNet-1K val top-1: 71.78%
- ImageNet-1K val top-5: 90.53%

Interpretation:

Typhon is much better than DHViT on ImageNet-R, but still below SHViT-S1. This is a positive result for abandoning full block duplication, but not yet evidence that Typhon improves SHViT.

## Recommended Architecture Revision

The next Typhon version should be evaluated first with the function-preserving `gated_sum` mode. That is the most important change.

Implemented default block design:

```python
head_outputs = [mixer_i(x1) for mixer_i in self.mixers]
x1_merged = sum(gate_i * head_i for gate_i, head_i in zip(self.mixer_gates, head_outputs))
```

Initialize:

- mixer 0 from SHViT
- mixer 1 from SHViT or randomized small
- gates as `[1.0, 0.0]`
- no post-sum `GroupNorm` at first

This starts exactly or nearly exactly as SHViT, then lets the second mixer gradually learn useful residual behavior.

Implemented experimental alternative closer to vanilla MHA:

```python
chunks = torch.split(x1, self.head_dims, dim=1)
head_outputs = [mixer_i(chunk_i) for mixer_i, chunk_i in zip(self.mixers, chunks)]
x1_merged = torch.cat(head_outputs, dim=1)
```

This preserves head identity until the copied SHViT projection. It is closer to multi-head attention, but it is not function-preserving because the original SHViT mixer used the full active channel subset for q/k/v.

Avoid a strong diversity loss on final features. That already failed in DHViT. If diversity is used, apply it weakly to attention maps or mixer outputs, and only after the model has recovered SHViT baseline performance.

## Recommended Training Strategy

### Phase 0: Fix Training Mechanics

Before another long HPC run:

1. freeze stage-1 BatchNorm running stats every epoch
2. fix best-checkpoint tracking
3. log trainable parameter groups
4. log mixer-output cosine similarity
5. evaluate ImageNet-R every epoch or every 2 epochs on a fixed 10k-image proxy subset

### Phase 1: Function-Preserving Warm Start

Goal: prove Typhon can retain SHViT baseline before trying to improve it.

Train only:

- new mixer gates
- second mixer parameters
- optional fusion layer
- classifier head if necessary

Freeze:

- patch embedding
- blocks1
- original mixer path
- most FFN/convolution weights

Use:

- 2 to 5 epochs
- LR for new parameters: `3e-4`
- LR for copied pretrained parameters: `0`
- weight decay: `0.01`
- teacher distillation from original SHViT logits

Loss:

```text
loss = CE(student, labels)
     + 0.5 * KL(student / T, SHViT_teacher / T) * T^2
```

Use `T = 2` or `T = 4`.

This prevents Typhon from moving away from SHViT before the extra mixer has learned anything useful.

### Phase 2: Controlled ImageNet-1K Fine-Tuning

Unfreeze the later Typhon blocks gradually.

Suggested parameter groups:

| Parameters | LR |
|---|---:|
| new mixer/fusion/gates | `3e-4` |
| copied attention/proj in blocks2/3 | `3e-5` |
| copied FFN/conv in blocks2/3 | `1e-5` |
| head | `5e-5` |
| patch_embed + blocks1 | frozen |

Use:

- warmup: 2 to 5 epochs
- cosine decay: 30 to 60 epochs
- label smoothing: `0.1`
- mixup: `0.1` to `0.2`
- cutmix: `0.5` to `1.0`
- gradient clipping: `1.0`
- EMA if available

Stop early if ImageNet-1K val drops while ImageNet-R proxy does not improve.

### Phase 3: Robustness-Oriented Fine-Tuning

After Typhon matches or exceeds SHViT on ImageNet-1K, run a short robustness pass:

- train on ImageNet-1K with stronger style/shape augmentations
- evaluate ImageNet-200 and ImageNet-R every epoch
- keep the SHViT teacher distillation term
- use lower LR, around `1e-5` for copied weights and `5e-5` for new mixer/fusion parameters

Do not fine-tune directly on ImageNet-R unless the thesis permits using ImageNet-R labels for training. If ImageNet-R remains a test set, use it only for evaluation and model selection should be done on an IID validation subset or a separate proxy.

## Fast Tests Before Another Long Run

To avoid waiting 69 hours before learning the result, run these checkpoints:

1. After initialization, evaluate Typhon without training. It should be close to SHViT. If it is not, the architecture is not function-preserving.
2. After 1 epoch, evaluate ImageNet-1K val subset and ImageNet-R 10k subset.
3. Track mixer-output similarity. If the two mixers stay near-identical, the architecture is not learning multi-head behavior.
4. Run per-mixer ablation: disable mixer 2, then disable mixer 1. If disabling mixer 2 changes almost nothing, the second mixer is unused.
5. Run representation analysis on `pooled` features and logits after 1, 3, and 5 epochs.

Minimum stop rule:

```text
Stop the run if by epoch 3:
- ImageNet-1K val is more than 0.5 points below SHViT, and
- ImageNet-R proxy is not at least 0.5 points above SHViT or the previous Typhon checkpoint.
```

## Thesis Interpretation

The current evidence supports this statement:

> Duplicating the entire late SHViT backbone in DHViT increased capacity but did not produce semantically useful complementary representations. Typhon is a more principled modification because it injects additional attention capacity inside the SHSA mixer while preserving a single backbone path. In the current implementation, Typhon improves over DHViT on ImageNet-R, but it does not yet exceed the original SHViT-S1. The likely reasons are that the current mixer fusion is not equivalent to true multi-head attention, the copied SHViT initialization is not function-preserving, and fine-tuning allows pretrained representations to drift too aggressively.
