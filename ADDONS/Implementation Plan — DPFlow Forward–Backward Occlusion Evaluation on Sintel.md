# Implementation Plan — DPFlow Forward–Backward Occlusion Evaluation on Sintel

## 1. Objective

Implement an independent evaluation pipeline to investigate the occlusion-handling behavior of the **original/public DPFlow model checkpoint** on the **MPI Sintel** dataset.

The experiment has two complementary goals:

1. Evaluate whether **forward–backward flow consistency** can identify occluded regions.
2. Evaluate the actual optical-flow accuracy of DPFlow separately on:
   - all pixels,
   - non-occluded pixels,
   - occluded pixels,
   - optionally occlusion-boundary pixels.

The implementation must **not modify the DPFlow model architecture or training code**.

The evaluation should be implemented as a separate inference/metrics pipeline so that the original DPFlow implementation remains unchanged and the experiment is reproducible.

---

## 2. Experimental Background

Given two consecutive frames:

$$
I_t,\ I_{t+1}
$$

run DPFlow in both directions:

### Forward

$$
F_{fw}=F_{t\rightarrow t+1}
$$

### Backward

$$
F_{bw}=F_{t+1\rightarrow t}
$$

For a visible/non-occluded pixel \(x\), the forward and backward flows should approximately cancel after mapping them to the same coordinate system:

$$
F_{fw}(x)
+
F_{bw}(x+F_{fw}(x))
\approx 0
$$

Therefore define forward–backward consistency error:

$$
C(x)=
\left\|
F_{fw}(x)
+
W(F_{bw},F_{fw})(x)
\right\|_2
$$

where \(W\) samples/warps the backward flow at the forward-mapped location.

A high consistency error is treated as a **candidate occlusion signal**.

Important:

> Forward–backward inconsistency is only an occlusion cue. It must NOT be interpreted directly as proof that DPFlow predicts accurate flow in occluded regions.

A model can produce mutually consistent but incorrect forward/backward flows.

Therefore the experiment must separately evaluate flow accuracy using Sintel ground truth.

---

## 3. Critical Coordinate-System Requirement

Do NOT directly subtract the two flow tensors:

```text
flow_forward - flow_backward
```

This is incorrect because the two flows are defined in different image coordinate systems.

Correct procedure:

```text
I_t ----------------------> I_t+1
       F_forward

I_t <---------------------- I_t+1
       F_backward
```

For a pixel \(x\) in frame \(t\):

$$
x'=x+F_{fw}(x)
$$

Then sample the backward flow at \(x'\):

$$
F_{bw}(x')
$$

and calculate:

$$
C(x)=
\|F_{fw}(x)+F_{bw}(x')\|_2
$$

The implementation must use a valid differentiable/non-differentiable sampling mechanism appropriate for evaluation, typically `torch.nn.functional.grid_sample`, or an existing repository warping utility if one is already validated and compatible with the DPFlow flow convention.

Before implementation, inspect the repository to determine whether DPFlow already provides a correct warping utility.

---

## 4. Scope

### Must implement

- Load the public/best-performing DPFlow checkpoint.
- Run forward inference.
- Run backward inference.
- Ensure the two inferences are independent.
- Compute forward–backward consistency.
- Handle out-of-bounds coordinates.
- Generate predicted occlusion masks from consistency.
- Load Sintel ground-truth optical flow.
- Load Sintel ground-truth occlusion information.
- Compute EPE metrics.
- Split EPE into occluded and non-occluded regions.
- Compute occlusion detection metrics.
- Save numerical results.
- Generate useful visualizations.

### Must NOT implement

- No DPFlow architecture modification.
- No changes to DPFlow training.
- No new network module.
- No fine-tuning.
- No retraining.
- No modification of the original checkpoint.
- No modification of the original correlation implementation unless required purely to fix an existing evaluation incompatibility.
- Do not introduce a new occlusion-aware loss.

This is an evaluation experiment, not a model modification.

---

## 5. Repository Inspection Before Coding

The agent MUST inspect the existing repository before implementing anything.

Specifically inspect:

1. DPFlow model definition.
2. DPFlow inference entry point.
3. Existing checkpoint loading mechanism.
4. Existing Sintel dataset loader.
5. Existing flow visualization utilities.
6. Existing flow warping utilities.
7. Flow output resolution and scaling.
8. Flow coordinate convention.
9. Input resizing/padding behavior.
10. Existing evaluation metrics.

Do not duplicate functionality if the repository already has a reliable implementation.

The agent must determine:

```text
Input resolution
      ↓
DPFlow
      ↓
Output flow resolution
      ↓
Flow scaling
      ↓
Sintel GT resolution
```

Any resizing of images must be accompanied by correct flow scaling.

---

## 6. Checkpoint Selection

Use the **publicly available DPFlow checkpoint corresponding to the best reported/released performance**, if such a checkpoint is available in the repository/project.

Do not retrain.

Record:

- checkpoint path/name,
- dataset/training regime,
- reported performance,
- inference configuration,
- image preprocessing configuration.

The final experiment report must clearly identify the exact checkpoint used.

---

## 7. Dataset

Use **MPI Sintel**, preferably the official training set when ground-truth flow and occlusion information are required.

Important clarification:

Sintel does not organize frames into folders such as:

```text
occluded/
non_occluded/
```

Instead, occlusion is represented as **pixel-level ground-truth information/masks** associated with the frame/flow data.

The evaluation must therefore construct masks from the provided ground truth.

For each frame pair, conceptually obtain:

```text
I_t
I_t+1
F_GT
M_occ_GT
```

where:

- `I_t`: first frame,
- `I_t+1`: second frame,
- `F_GT`: ground-truth optical flow,
- `M_occ_GT`: ground-truth occlusion mask.

The exact Sintel file format and occlusion representation must be verified from the dataset loader/data files rather than assumed.

---

## 8. Inference Pipeline

For each valid consecutive frame pair:

```text
I_t
I_t+1
```

run:

```text
flow_fw = DPFlow(I_t, I_t+1)
flow_bw = DPFlow(I_t+1, I_t)
```

The two inference calls must be independent.

If DPFlow maintains recurrent/internal state, make sure state is reset between the two directions.

Use evaluation mode:

```python
model.eval()
```

and disable gradient computation:

```python
torch.no_grad()
```

Do not accidentally reuse hidden states/features from the forward pass for the backward pass.

---

## 9. Flow Resolution and Scaling

This is a critical implementation detail.

If the model output has resolution:

$$
H_m \times W_m
$$

while Sintel GT has:

$$
H_{gt}\times W_{gt}
$$

the predicted flow must be resized appropriately.

When resizing flow spatially, scale the vector components accordingly:

$$
u' = u \cdot \frac{W_{gt}}{W_m}
$$

$$
v' = v \cdot \frac{H_{gt}}{H_m}
$$

Do not simply interpolate the flow tensor without scaling its displacement components.

If the existing DPFlow inference API already returns full-resolution flow with correct scaling, reuse that behavior.

The agent must verify this from the source code.

---

## 10. Forward–Backward Consistency Implementation

For each pixel:

$$
x=(x,y)
$$

construct its forward-mapped position:

$$
x'=x+F_{fw}(x)
$$

Then sample:

$$
F_{bw}(x')
$$

using the correct coordinate convention.

Calculate:

$$
E_{fb}(x)=
\left\|
F_{fw}(x)+F_{bw}(x')
\right\|_2
$$

Save this as:

```text
consistency_error
```

Shape should conceptually be:

```text
[B, H, W]
```

or equivalent.

---

## 11. Out-of-Bounds Handling

Some forward-mapped coordinates may fall outside the second image:

$$
x' \notin [0,W-1]\times[0,H-1]
$$

These pixels do not have a valid backward-flow sample.

Create:

```text
fb_valid_mask
```

such that:

```text
True  = valid backward-flow sampling location
False = out of bounds
```

Do NOT silently treat out-of-bounds pixels as ordinary consistency errors.

Metrics involving forward–backward consistency must explicitly account for this validity mask.

The implementation should report the proportion of pixels that are invalid/out-of-bounds.

---

## 12. Occlusion Prediction

Convert consistency error into a predicted occlusion mask.

Do not hard-code an arbitrary statement such as:

```text
error < 1 = good
error > 1 = occluded
```

because absolute flow magnitude varies between scenes.

Prefer a relative consistency criterion.

A possible formulation is:

$$
E_{rel}(x)=
\frac{
\|F_{fw}(x)+F_{bw}(x')\|_2
}{
\|F_{fw}(x)\|_2+
\|F_{bw}(x')\|_2+
\epsilon
}
$$

Alternatively, use a magnitude-dependent threshold:

$$
E_{fb}(x)
>
\alpha
\left(
\|F_{fw}(x)\|+
\|F_{bw}(x')\|
\right)
+\beta
$$

The exact formulation must be documented.

Do not choose a threshold solely because it gives a visually pleasing mask.

---

## 13. Threshold Evaluation

Because occlusion prediction depends on a threshold, evaluate multiple thresholds.

For example:

```text
threshold sweep
       ↓
precision
recall
F1
IoU
       ↓
F1-vs-threshold
PR curve
```

Prefer reporting:

- Precision
- Recall
- F1
- IoU
- AUPRC

If practical, also report AUROC.

A threshold sweep is preferable to reporting only one arbitrary threshold.

If a fixed threshold is eventually selected for visualization, clearly distinguish:

```text
threshold-independent metrics
```

from:

```text
fixed-threshold metrics
```

---

## 14. Occlusion Detection Metrics

Compare:

```text
M_occ_pred
```

against:

```text
M_occ_GT
```

using only valid pixels.

Required metrics:

### Precision

$$
Precision=\frac{TP}{TP+FP}
$$

### Recall

$$
Recall=\frac{TP}{TP+FN}
$$

### F1

$$
F1=\frac{2PR}{P+R}
$$

### IoU

$$
IoU=\frac{TP}{TP+FP+FN}
$$

Recommended:

- AUPRC
- AUROC

These metrics answer:

> How well does forward–backward inconsistency identify actual occluded pixels?

They do NOT directly measure flow accuracy.

---

## 15. Optical Flow Accuracy

Use Sintel ground-truth flow:

$$
F_{GT}
$$

and predicted forward flow:

$$
F_{pred}=F_{fw}
$$

Compute endpoint error:

$$
EPE(x)=
\|F_{pred}(x)-F_{GT}(x)\|_2
$$

Then use the Sintel occlusion mask to separate pixels.

---

## 16. Required EPE Metrics

Report:

### Overall EPE

$$
EPE_{all}
$$

### Non-occluded EPE

$$
EPE_{non-occ}
=
\frac{1}{|N|}
\sum_{x\in N}EPE(x)
$$

### Occluded EPE

$$
EPE_{occ}
=
\frac{1}{|O|}
\sum_{x\in O}EPE(x)
$$

where:

```text
O = ground-truth occluded pixels
N = ground-truth non-occluded pixels
```

This distinction is essential.

The primary metric for investigating occlusion handling should be:

$$
\boxed{EPE_{occ}}
$$

while `EPE_non-occ` is needed to ensure that an apparent improvement in occluded regions does not come at an unreasonable cost elsewhere.

---

## 17. Recommended Additional Metrics

If supported by the existing Sintel evaluation implementation, also report:

- Fl-all / outlier rate.
- Fl-occ.
- Fl-noc.
- EPE distribution.
- Median EPE.
- Percentiles if useful.

Do not implement duplicate versions of standard Sintel metrics if the repository already provides a trusted implementation.

---

## 18. Occlusion Boundary Evaluation

Occlusion boundaries are especially difficult because they often correspond to motion discontinuities.

If feasible, derive an occlusion-boundary band from the GT occlusion mask, for example using morphological boundary extraction.

Then calculate:

$$
EPE_{boundary}
$$

This should be considered an **additional analysis**, not a mandatory first implementation if it significantly complicates the pipeline.

Potential report:

```text
EPE_all
EPE_non-occ
EPE_occ
EPE_boundary
```

This helps determine whether the model struggles specifically near motion discontinuities.

---

## 19. Important Interpretation Rules

The agent must NOT make the following invalid conclusion:

> "Low forward–backward consistency error means DPFlow handles occlusion well."

Instead:

### Forward–backward consistency

Measures whether the two directional predictions are mutually consistent.

### Occlusion F1/IoU/AUPRC

Measures how useful this consistency signal is for identifying actual occluded pixels.

### Occluded EPE metrics

Measures actual flow estimation quality on ground-truth occluded pixels.

Therefore:

$$
\boxed{
\text{Occlusion handling}
\neq
\text{Forward-backward consistency alone}
}
$$

A strong evaluation combines all of them.

---

## 20. Expected Output Structure

Create a dedicated evaluation directory, for example:

```text
occlusion_eval/
├── inference_occlusion.py
├── metrics_occlusion.py
├── visualization.py
├── config/
│   └── sintel_occlusion.yaml
└── results/
    ├── raw/
    ├── metrics/
    └── visualizations/
```

The exact directory structure should follow the repository's existing conventions if applicable.

---

## 21. Per-Frame Saved Results

For each evaluated frame pair, optionally save:

```text
flow_forward
flow_backward
consistency_error
occlusion_pred
occlusion_gt
flow_gt
valid_mask
```

Because storage may become large, the implementation should make raw-result saving configurable.

For example:

```text
save_raw_results = true/false
save_visualizations = true/false
```

Metrics must still be reproducible without requiring visualization files.

---

## 22. Visualization Requirements

For a small number of representative examples, generate:

1. First frame.
2. Second frame.
3. Forward flow visualization.
4. Backward flow visualization.
5. Forward–backward consistency error map.
6. GT occlusion mask.
7. Predicted occlusion mask.
8. EPE map.
9. Optionally GT flow.

A useful layout is conceptually:

```text
I_t | I_t+1 | F_fw | F_bw
--------------------------------
FB consistency | GT occ | Pred occ
--------------------------------
EPE map
```

Do not use visualization as the primary quantitative evaluation.

---

## 23. Aggregation

Metrics must be aggregated over the complete evaluation set.

Do not average per-frame percentages naively if that produces a different result from global pixel-level aggregation.

Prefer accumulating counts:

```text
TP
FP
FN
TN
```

over all valid pixels, then calculate global:

```text
Precision
Recall
F1
IoU
```

For EPE, aggregate:

```text
sum EPE
number of valid pixels
```

separately for:

```text
all
occluded
non-occluded
boundary
```

Then compute the final mean.

Also report the number of evaluated frame pairs and pixels.

---

## 24. Result Table

The final report should contain at least:

| Metric | DPFlow |
| --- | ---: |
| Overall EPE | ... |
| Non-occluded EPE | ... |
| Occluded EPE | ... |
| Occlusion Precision | ... |
| Occlusion Recall | ... |
| Occlusion F1 | ... |
| Occlusion IoU | ... |
| Occlusion AUPRC | ... |

If implemented:

| Additional Metric | DPFlow |
| --- | ---: |
| Boundary EPE | ... |
| Fl-all | ... |
| Fl-occ | ... |
| Fl-noc | ... |

---

## 25. Reproducibility

The evaluation must record:

- DPFlow checkpoint.
- Git commit/repository version if available.
- Sintel split.
- Sintel rendering type if applicable.
- Input resolution.
- Any resize/crop/padding settings.
- Flow scaling behavior.
- Forward-backward consistency formula.
- Occlusion threshold/formula.
- epsilon value.
- Number of evaluated pairs.
- GPU/device.
- Inference precision, e.g. FP32/FP16, if applicable.

The experiment should be executable from a single command or clearly documented command sequence.

---

## 26. Validation Before Full Evaluation

Before running the complete Sintel evaluation, implement sanity checks on a small number of frame pairs.

Verify:

### Check 1 — Flow direction

Forward and backward flow should visually point in opposite directions for ordinary motion.

### Check 2 — Coordinate convention

A known synthetic translation should produce the expected forward/backward relationship.

### Check 3 — Warping

Verify that:

$$
x'=x+F_{fw}(x)
$$

maps pixels to the correct location.

### Check 4 — Out-of-bounds mask

Verify that coordinates outside the image are marked invalid.

### Check 5 — Flow scaling

Compare predicted flow resolution against GT and verify vector magnitude scaling.

### Check 6 — EPE

Compare the custom EPE implementation against the repository's existing Sintel metric implementation if available.

### Check 7 — Occlusion mask orientation

Make sure:

```text
occluded = 1
non-occluded = 0
```

is interpreted consistently throughout the pipeline.

---

## 27. Unit Tests / Synthetic Sanity Test

Before using real Sintel results, create a minimal synthetic test.

Example:

Assume:

$$
F_{fw}=(5,0)
$$

and:

$$
F_{bw}=(-5,0)
$$

at the corresponding locations.

The consistency error should be approximately:

$$
0
$$

For an intentionally inconsistent backward flow:

$$
F_{bw}=(-3,0)
$$

the consistency error should be approximately:

$$
2
$$

This validates the forward-backward calculation independently of DPFlow.

Also test an out-of-bounds forward coordinate.

---

## 28. Performance Requirements

The experiment is evaluation-only and should avoid unnecessary GPU memory usage.

Use:

```python
torch.no_grad()
```

and inference mode where appropriate.

Do not keep unnecessary computational graphs.

Process frame pairs sequentially unless the repository's evaluation infrastructure already supports batching safely.

Avoid storing the entire dataset's raw tensors in GPU memory.

Move results to CPU before long-term storage.

---

## 29. Expected Final Deliverables

The implementation should produce:

### Code

```text
inference/evaluation script
metrics implementation
visualization implementation
configuration
```

### Quantitative results

```text
Overall EPE
Non-occ EPE
Occ EPE
Occlusion Precision
Occlusion Recall
Occlusion F1
Occlusion IoU
AUPRC
```

plus optional:

```text
Boundary EPE
Fl-all
Fl-occ
Fl-noc
```

### Visual results

Representative examples showing:

```text
Input pair
Forward flow
Backward flow
Consistency error
GT occlusion
Predicted occlusion
EPE
```

### Experiment metadata

Exact checkpoint, dataset split, preprocessing, threshold/formula, and evaluation configuration.

---

## 30. Final Research Question

The implementation should ultimately allow the following questions to be answered separately:

### Q1 — Can forward–backward consistency identify occlusions?

Evaluate:

$$
F1,\ IoU,\ AUPRC
$$

between predicted and GT occlusion masks.

### Q2 — How accurate is DPFlow inside actual occluded regions?

Evaluate:

$$
\boxed{EPE_{occ}}
$$

### Q3 — Does DPFlow sacrifice normal-region accuracy?

Evaluate:

$$
EPE_{non-occ}
$$

### Q4 — Is the model particularly weak around occlusion boundaries?

Evaluate:

$$
EPE_{boundary}
$$

### Q5 — Is low forward–backward error actually associated with correct flow?

Compare consistency error against flow EPE. This is an optional but valuable analysis.

---

## 31. Important Principle

The experiment must distinguish three concepts:

```text
Bidirectional consistency
        ↓
Can the two predictions agree?

Occlusion detection
        ↓
Does disagreement correspond to actual occlusion?

Flow accuracy
        ↓
Is the predicted flow actually correct?
```

These are related but not equivalent.

The final analysis should therefore never use forward–backward consistency alone as evidence that DPFlow "handles occlusion well".

The strongest evidence is the combination:

$$
\boxed{
EPE_{occ}
+
EPE_{non-occ}
+
Occlusion\ F1/IoU/AUPRC
}
$$

with forward–backward consistency providing the mechanism for the occlusion-analysis component.
