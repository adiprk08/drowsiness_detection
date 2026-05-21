# Dataset summary - combined deployment model

Frame counts feeding the MobileNetV2 DDD+UTA model. All
splits are subject-disjoint: no person appears in more
than one split.

## Datasets

| Dataset | Camera domain | Subjects | Alert | Drowsy | Total |
|---|---|---|---|---|---|
| DDD | cabin camera | 28 | 19,445 | 22,348 | 41,793 |
| UTA-RLDD | webcam / phone | 48 | 27,089 | 27,001 | 54,090 |
| **Total** | - | 76 | 46,534 | 49,349 | 95,883 |

DDD "subjects" are per-video pseudo-subject groups used for leak-free splitting; UTA subjects are the 48 recorded individuals.

## Train / validation / test split

| Split | DDD frames | UTA frames | Total | Drowsy share |
|---|---|---|---|---|
| train | 28,031 | 38,712 | 66,743 | 51.7% |
| val | 6,258 | 7,669 | 13,927 | 50.0% |
| test | 7,504 | 7,709 | 15,213 | 51.6% |
| **Total** | 41,793 | 54,090 | 95,883 | 51.5% |

UTA subject split: 34 train / 7 val / 7 test (seed 42).
