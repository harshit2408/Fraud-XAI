# Slice Metrics — Validation Split

Threshold: `0.0004` (recomputed on validation (no frozen threshold in artifact); cost model: cost_fn=500, cost_fp=5, revenue_tp=480)

Model: `models/xgb_model.pkl` | Rows: 59053 | Fraud rate: 0.0349

Rows flagged `reliable=False` have fewer than 30 validation examples — treat their metrics as directional, not conclusive.

**Reading precision here:** with `revenue_tp` this much larger than `cost_fp` relative to the ~3.5% fraud base rate, the business-value-optimal policy sits close to "flag almost everyone" — the low precision / high recall seen in every slice below is an expected consequence of the configured cost model in `config/config.yaml`, not a defect in slicing. Revisit `thresholds.revenue_tp` in that config if a higher-precision operating point is wanted.


## ProductCD

| dimension | slice | count | fraud_rate | pr_auc | reliable | TP | TN | FP | FN | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ProductCD | W | 48089 | 0.0201 | 0.5358 | True | 925 | 27538 | 19583 | 43 | 0.0451 | 0.9556 | 0.0861 | 0.5919 |
| ProductCD | C | 6404 | 0.1254 | 0.8255 | True | 786 | 1996 | 3605 | 17 | 0.1790 | 0.9788 | 0.3027 | 0.4344 |
| ProductCD | R | 2059 | 0.0559 | 0.8629 | True | 114 | 1067 | 877 | 1 | 0.1150 | 0.9913 | 0.2061 | 0.5736 |
| ProductCD | H | 1579 | 0.0678 | 0.7211 | True | 104 | 620 | 852 | 3 | 0.1088 | 0.9720 | 0.1957 | 0.4585 |
| ProductCD | S | 922 | 0.0738 | 0.8301 | True | 66 | 489 | 365 | 2 | 0.1531 | 0.9706 | 0.2645 | 0.6020 |



## hour_bucket

| dimension | slice | count | fraud_rate | pr_auc | reliable | TP | TN | FP | FN | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hour_bucket | evening | 24346 | 0.0355 | 0.6673 | True | 833 | 12991 | 10491 | 31 | 0.0736 | 0.9641 | 0.1367 | 0.5678 |
| hour_bucket | afternoon | 18530 | 0.0254 | 0.6395 | True | 458 | 10351 | 7708 | 13 | 0.0561 | 0.9724 | 0.1061 | 0.5833 |
| hour_bucket | night | 13530 | 0.0404 | 0.7221 | True | 527 | 7007 | 5976 | 20 | 0.0810 | 0.9634 | 0.1495 | 0.5568 |
| hour_bucket | morning | 2647 | 0.0676 | 0.8921 | True | 177 | 1361 | 1107 | 2 | 0.1379 | 0.9888 | 0.2420 | 0.5810 |



## card_tenure_bucket

| dimension | slice | count | fraud_rate | pr_auc | reliable | TP | TN | FP | FN | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| card_tenure_bucket | new | 14764 | 0.0289 | 0.7423 | True | 415 | 9982 | 4355 | 12 | 0.0870 | 0.9719 | 0.1597 | 0.7042 |
| card_tenure_bucket | mid_1 | 14763 | 0.0387 | 0.6763 | True | 551 | 7745 | 6446 | 21 | 0.0787 | 0.9633 | 0.1456 | 0.5619 |
| card_tenure_bucket | established | 14763 | 0.0347 | 0.6104 | True | 488 | 7493 | 6757 | 25 | 0.0674 | 0.9513 | 0.1258 | 0.5406 |
| card_tenure_bucket | mid_2 | 14763 | 0.0372 | 0.7519 | True | 541 | 6490 | 7724 | 8 | 0.0655 | 0.9854 | 0.1228 | 0.4763 |

