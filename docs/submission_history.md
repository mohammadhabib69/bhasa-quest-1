# Submission history

Fourteen submissions were made in total; the six most recent are listed here. Kaggle scores the
final entry on the **private** split (70 % of the test set), while the **public** split (30 %,
≈ 614 rows) is visible during the competition.

| # | File | Public LB | Private LB | Selected |
|---|---|---|---|---|
| 1 | `submission (8).csv` | 0.76023 | **0.78786** | yes |
| 2 | `submission_v9.csv` | 0.75744 | 0.78768 | no |
| 3 | `submission_v8.csv` | 0.75881 | 0.78340 | yes |
| 4 | `submission_best_v4.csv` | 0.74806 | 0.78078 | no |
| 5 | `submission_best_v3.csv` | 0.73936 | 0.76765 | no |
| 6 | `submission_best_v2.csv` | 0.73825 | 0.75858 | no |

## What the ordering says

Three observations drove the strategy in the final days.

**1. The public split is not a reliable estimator.**
`submission_v9` *lost* to `submission_v8` on public (0.75744 vs 0.75881) but *won* on private
(0.78768 vs 0.78340) — the ordering flips. Across a 614-row split the bootstrap standard deviation
of macro-F1 is ≈ 0.018, so a 0.003 difference between two candidates carries almost no information.
Meanwhile the two contenders for first place were separated by 0.003 on public and 0.023 on private.

**2. Every submission scored higher privately than publicly.**
The private subset is simply easier: the same model gains 0.02–0.04 when evaluated on it. Absolute
scores are therefore not comparable across splits, and any reasoning of the form "OOF says 0.754 so
private should be 0.754" is wrong.

**3. The final selection was a hedge, not a ranking.**
Two submissions were selected — the highest-OOF stacker and a slightly different blending variant —
because Kaggle keeps the better of the two on the private split. The variant was not the second-best
public score; it was the second-best *out-of-fold* model with a different error profile.

## Leaderboard outcome

| Rank | Team | Private score |
|---|---|---|
| 1 | **Mohammad Habib** (14 entries) | **0.78786** |
| 2 | Hawlader Sajib (50 entries) | 0.76455 |
| 3 | Tasnimul Hasan Rahat (12 entries) | 0.76121 |

The team that led the public board for most of the competition — and spent fifty submissions doing
it — finished second. Fourteen submissions selected on out-of-fold evidence finished first.
