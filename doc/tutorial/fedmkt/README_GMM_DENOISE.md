# FedMKT + ARC-C noisy-label GMM denoising

This extension adds a noisy-label experiment on top of the existing FedMKT baseline without modifying `python/fate_llm/algo/fedmkt`.

## 1. Generate noisy ARC-C data

```bash
cd /home/cmcc/lhy/Fedllm-CN
python doc/tutorial/fedmkt/add_arc_noise.py \
  --config doc/tutorial/fedmkt/configs/default.yaml
```

By default, the script reads `data.tasks.arc_c.data_dir` from `default.yaml`, flips labels in `client_0` ... `client_3`, keeps `common` / `validation` / `test` clean, and saves the result to `noise.output_dir` / `denoise.noisy_data_dir`.

Useful overrides:

```bash
python doc/tutorial/fedmkt/add_arc_noise.py \
  --noise_rate 0.4 \
  --noise_type sym \
  --noise_parts clients \
  --output_dir /home/cmcc/lhy/data/arc_c_noisy
```

The saved dataset preserves ARC columns and adds `answerKey_clean`, `answerKey_noisy`, `clean_idx`, `noisy_idx`, `train_idx`, `is_noisy`, `noise_source`, and `noise_seed`.

## 2. Run FedMKT-GMM on noisy data

```bash
cd /home/cmcc/lhy/Fedllm-CN/doc/tutorial/fedmkt
python3 test_gmm_denoise.py \
  --parties arbiter:10002 guest:9998 host:9999 host:10000 host:10001 \
  --log_level INFO
```

The script resolves `denoise.noisy_data_dir` / `noise.output_dir` in `default.yaml` and then calls the new `fate_llm.algo.fedmkt_gmm` implementation.  SLM private-data training uses `GMMNoiseTrainer`, while public FedMKT knowledge distillation still reuses the baseline FedMKT logic.

## 3. CE vs symmetric CE

Default GMM fitting uses per-sample CE loss:

```yaml
denoise:
  loss_type: ce
```

To run the FedRG-style symmetric CE variant:

```bash
FEDMKT_DENOISE_LOSS_TYPE=symmetric_ce python3 test_gmm_denoise.py --parties ... --log_level INFO
```

or edit:

```yaml
denoise:
  loss_type: symmetric_ce
  sce_alpha: 1.0
  sce_beta: 0.1
  rce_epsilon: 1.0e-4
```

## 4. Metrics

`GMMNoiseTrainer` logs the following metrics to wandb, swanlab, and the local metric logger when enabled:

- `*/noise_detect_acc`
- `*/noise_detect_precision`
- `*/noise_detect_recall`
- `*/batch_noise_rate`
- `*/clean_prob_mean`
- `*/sample_weight_mean`
- `*/gmm_low_mean` / `*/gmm_high_mean`
- normal FedMKT loss and ARC accuracy metrics are preserved

The `denoise.target_min_improve: 0.05` value is recorded as the intended target; actual +5 point improvement must be verified from the saved metrics rather than assumed.
