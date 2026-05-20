# Model comparison — held-out test set

All numbers from `artifacts/<model>/test_metrics.json`. The
`@safety` rows pull the safety operating point from
`artifacts/<model>/calibration.json` — the largest threshold
that keeps validation drowsy recall ≥ 0.95.

| Model | Thr | N | Acc | Macro-F1 | F1 (drowsy) | Recall (drowsy) | Prec. (drowsy) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| baseline_cnn | 0.50 | 7504 | 0.6406 | 0.6371 | 0.6728 | 0.6708 | 0.6749 | 0.6014 |
| baseline_cnn @safety | 0.05 | 7504 | 0.5678 | 0.5116 | 0.6773 | 0.8234 | 0.5753 |   --   |
| alexnet | 0.50 | 7504 | 0.5204 | 0.5204 | 0.5180 | 0.4678 | 0.5803 | 0.5309 |
| alexnet @safety | 0.05 | 7504 | 0.5252 | 0.4557 | 0.6502 | 0.8009 | 0.5472 |   --   |
| mobilenet_v2 | 0.50 | 7504 | 0.7216 | 0.7050 | 0.7750 | 0.8701 | 0.6986 | 0.6629 |
| mobilenet_v2 @safety | 0.05 | 7504 | 0.6514 | 0.5659 | 0.7585 | 0.9940 | 0.6133 |   --   |
| two_stream (eye) | 0.50 | 16823 | 0.9047 | 0.9023 | 0.8873 | 0.9478 | 0.8341 | 0.9765 |
| two_stream (face) | 0.50 | 7504 | 0.5003 | 0.4741 | 0.5914 | 0.6565 | 0.5381 | 0.5966 |
| alexnet_combined (ddd) | 0.50 | 7504 | 0.8597 | 0.8588 | 0.8698 | 0.8505 | 0.8899 |   --   |
| alexnet_combined (uta) | 0.50 | 7709 | 0.6927 | 0.6818 | 0.6230 | 0.5265 | 0.7627 |   --   |
| alexnet_combined (combined) | 0.50 | 15213 | 0.7751 | 0.7744 | 0.7618 | 0.6971 | 0.8398 |   --   |
| baseline_cnn_combined (ddd) | 0.50 | 7504 | 0.6800 | 0.6640 | 0.5908 | 0.4192 | 1.0000 |   --   |
| baseline_cnn_combined (uta) | 0.50 | 7709 | 0.3859 | 0.3527 | 0.2062 | 0.1655 | 0.2737 |   --   |
| baseline_cnn_combined (combined) | 0.50 | 15213 | 0.5310 | 0.5066 | 0.3969 | 0.2991 | 0.5899 |   --   |
| mobilenet_v2_combined (ddd) | 0.50 | 7504 | 0.8277 | 0.8262 | 0.8423 | 0.8353 | 0.8494 |   --   |
| mobilenet_v2_combined (uta) | 0.50 | 7709 | 0.7123 | 0.7084 | 0.6745 | 0.6182 | 0.7420 |   --   |
| mobilenet_v2_combined (combined) | 0.50 | 15213 | 0.7692 | 0.7692 | 0.7661 | 0.7325 | 0.8030 |   --   |

**Two-stream caveat.** The eye and face branches are evaluated on
different held-out subsets (MRL eye crops vs DDD face crops)
because no test sample carries both modalities — the rows are not
directly comparable to each other or to a single fused model.

**Combined-training rows.** Models trained by `src.train_combined`
appear as one row per evaluation domain — `(ddd)` is the original
cabin-camera test split, `(uta)` is the held-out UTA-RLDD webcam
subjects, `(combined)` is both pooled. ROC-AUC is blank for these
rows because `train_combined` records macro-F1 metrics only.
