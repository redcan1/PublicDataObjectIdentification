# Fine-tuned vehicle detectors

YOLO11s-OBB (DOTA-v1 pretrained, `yolo11s-obb.pt`) fine-tuned on the blind hand counts in
`labels/irvine` and `labels/sacramento` (NAIP 0.6 m, upsampled to 0.2 m). Stored with Git LFS.

| file | trained on | use |
|---|---|---|
| `yolo11s-obb-cv0.pt`, `-cv1.pt`, `-cv2.pt` | 3-fold split by lot (`build_training_set.py --folds 3 --fold k --seed 82`) | score each lot with the model that did not see it; average the three for area-wide trends |
| `yolo11s-obb-ft1.pt` | single split (`--holdout-frac 0.34 --seed 3`) | first fine-tune; superseded by the fold models |

Recommended confidence cut: 0.20 (out-of-fold r = 0.95 vs hand counts in both cities).
Run with e.g. `aerialdetect run configs/irvine.yaml --set model.weights=models/yolo11s-obb-cv0.pt`.
