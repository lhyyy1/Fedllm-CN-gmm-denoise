#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import math
import logging

import numpy as np

from fate_llm.algo.fedmkt.utils.vars_define import (
    ALIGNED_OTHER_LOGITS,
    ALIGNED_OTHER_INDICES,
    ALIGNED_OTHER_METRIC,
)


logger = logging.getLogger(__name__)


def _softmax(values, temperature):
    if not values:
        return []
    scale = max(float(temperature), 1e-12)
    scaled = [float(v) / scale for v in values]
    max_value = max(scaled)
    exps = [math.exp(v - max_value) for v in scaled]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values) for _ in values]
    return [v / total for v in exps]


def _aggregate_position_probabilities(logits_by_model, indices_by_model, temperature, epsilon):
    """Aggregate one token position on the union of all aligned SLM supports."""
    union_indices = []
    union_seen = set()
    probability_sum = {}
    valid_model_count = 0

    for logits, indices in zip(logits_by_model, indices_by_model):
        if not logits or not indices:
            continue
        limit = min(len(logits), len(indices))
        logits = logits[:limit]
        indices = indices[:limit]
        probabilities = _softmax(logits, temperature)
        valid_model_count += 1

        for token_id, probability in zip(indices, probabilities):
            token_id = int(token_id)
            if token_id not in union_seen:
                union_seen.add(token_id)
                union_indices.append(token_id)
            probability_sum[token_id] = probability_sum.get(token_id, 0.0) + float(probability)

    if valid_model_count == 0 or not union_indices:
        return [], []

    averaged_probabilities = [
        max(probability_sum.get(token_id, 0.0) / valid_model_count, float(epsilon))
        for token_id in union_indices
    ]
    total_probability = sum(averaged_probabilities)
    averaged_probabilities = [p / total_probability for p in averaged_probabilities]

    # DataCollatorForFedMKT applies softmax(logits / distill_temperature).
    # Store pseudo-logits so that the collator reconstructs avg_prob stably.
    pseudo_logits = [
        float(temperature) * math.log(max(p, float(epsilon)))
        for p in averaged_probabilities
    ]
    return pseudo_logits, union_indices


def _mean_metric(metrics):
    values = []
    for metric in metrics:
        if metric is None:
            continue
        if isinstance(metric, (list, tuple)):
            values.extend(float(v) for v in metric)
        else:
            values.append(float(metric))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _gen_lagrange_coeffs(alpha_s, beta_s):
    """Return matrix U where U @ values_at_beta gives values_at_alpha."""
    alpha_s = np.asarray(alpha_s, dtype=np.complex128)
    beta_s = np.asarray(beta_s, dtype=np.complex128)
    coeffs = np.zeros((len(alpha_s), len(beta_s)), dtype=np.complex128)
    for i, alpha in enumerate(alpha_s):
        for j, beta in enumerate(beta_s):
            numerator = np.prod([alpha - other for other in beta_s if other != beta])
            denominator = np.prod([beta - other for other in beta_s if other != beta])
            coeffs[i, j] = numerator / denominator
    return coeffs


def _split_additive_blocks(values, num_blocks, rng):
    """
    Split each input vector into same-shaped additive blocks.

    For every client vector x, this returns K vectors x_1, ..., x_K with
    sum_k x_k = x. This follows the PPVFD-style split idea while keeping the
    FedMKT teacher vector shape unchanged for every block.
    """
    values = np.asarray(values, dtype=np.float64)
    num_blocks = max(int(num_blocks), 1)
    if num_blocks == 1:
        return values[:, None, :].astype(np.complex128)

    split_weights = rng.random((values.shape[0], num_blocks, values.shape[1]))
    split_weights_sum = split_weights.sum(axis=1, keepdims=True)
    split_weights = split_weights / np.maximum(split_weights_sum, 1e-12)
    return (values[:, None, :] * split_weights).astype(np.complex128)


def _mmlcc_decode_sum(
    client_values,
    num_blocks,
    privacy_guarantee,
    beta_radius,
    noise_sigma,
    noise_clip_theta,
    rng,
):
    """
    Run analog MMLCC for one shared-support vector.

    Each client vector is split into K same-shaped additive blocks. The server
    decodes the K summed data blocks and adds them back to reconstruct the
    summed teacher vector.
    """
    client_values = np.asarray(client_values, dtype=np.float64)
    num_clients, value_dim = client_values.shape
    if num_clients < 1:
        return np.zeros(value_dim, dtype=np.float64)

    num_blocks = max(int(num_blocks), 1)
    privacy_guarantee = max(int(privacy_guarantee), 0)
    num_encoded_blocks = num_blocks + privacy_guarantee
    num_coded_fragments = max(num_clients, num_encoded_blocks)

    alphas = np.exp(2j * np.pi * np.arange(num_coded_fragments) / num_coded_fragments)
    betas = beta_radius * np.exp(2j * np.pi * np.arange(num_encoded_blocks) / num_encoded_blocks)

    data_blocks = _split_additive_blocks(client_values, num_blocks, rng)
    if privacy_guarantee:
        std = math.sqrt(float(noise_sigma) ** 2 / privacy_guarantee / 2)
        privacy_blocks = rng.normal(
            scale=std,
            size=(num_clients, privacy_guarantee, value_dim, 2),
        )
        privacy_blocks = np.clip(
            privacy_blocks,
            -float(noise_clip_theta) * std,
            float(noise_clip_theta) * std,
        )
        privacy_blocks = privacy_blocks.view(np.complex128).squeeze(-1)
        blocks_to_encode = np.concatenate([data_blocks, privacy_blocks], axis=1)
    else:
        blocks_to_encode = data_blocks

    encoding_matrix = _gen_lagrange_coeffs(alphas, betas)
    encoded_shares = np.einsum("nk,ckd->cnd", encoding_matrix, blocks_to_encode)
    receiver_uploads = encoded_shares.sum(axis=0)
    decoding_matrix = _gen_lagrange_coeffs(betas, alphas)
    decoded_blocks = np.einsum("kn,nd->kd", decoding_matrix, receiver_uploads)
    return decoded_blocks[:num_blocks].real.sum(axis=0)


def aggregate_aligned_slm_teachers_dataset(
    aligned_dataset,
    blending_num,
    distill_temperature,
    probability_epsilon=1e-12,
    num_blocks=1,
    privacy_guarantee=1,
    beta_radius=1.15,
    noise_sigma=1.0,
    noise_clip_theta=6.0,
    seed=42,
):
    """
    Collapse multiple aligned SLM teachers into one aggregated teacher.

    This is the stable FedMKT integration point for MMLCC-style aggregation:
    all SLM logits are already mapped to the LLM vocabulary, so every union
    support contains token ids with the same semantics.
    """
    num_blocks = max(int(num_blocks), 1)
    if blending_num <= 1:
        return aligned_dataset, {
            "relative_error": 0.0,
            "positions": 0,
            "privacy_guarantee": int(max(int(privacy_guarantee), 0)),
            "num_blocks": int(num_blocks),
        }

    required_columns = []
    for idx in range(blending_num):
        required_columns.extend(
            [
                f"{ALIGNED_OTHER_LOGITS}_{idx}",
                f"{ALIGNED_OTHER_INDICES}_{idx}",
                f"{ALIGNED_OTHER_METRIC}_{idx}",
            ]
        )
    missing_columns = [col for col in required_columns if col not in aligned_dataset.column_names]
    if missing_columns:
        raise ValueError(f"Cannot aggregate aligned SLM teachers, missing columns: {missing_columns}")

    rng = np.random.default_rng(seed)
    error_stats = {"squared_error": 0.0, "squared_norm": 0.0, "positions": 0}

    def aggregate_batch(examples):
        batch_size = len(examples[f"{ALIGNED_OTHER_LOGITS}_0"])
        aggregated_logits = []
        aggregated_indices = []
        aggregated_metrics = []

        for row_idx in range(batch_size):
            first_model_steps = examples[f"{ALIGNED_OTHER_LOGITS}_0"][row_idx]
            step_count = len(first_model_steps)
            row_logits = []
            row_indices = []

            for step_idx in range(step_count):
                logits_by_model = [
                    examples[f"{ALIGNED_OTHER_LOGITS}_{model_idx}"][row_idx][step_idx]
                    for model_idx in range(blending_num)
                ]
                indices_by_model = [
                    examples[f"{ALIGNED_OTHER_INDICES}_{model_idx}"][row_idx][step_idx]
                    for model_idx in range(blending_num)
                ]
                direct_step_logits, step_indices = _aggregate_position_probabilities(
                    logits_by_model,
                    indices_by_model,
                    distill_temperature,
                    probability_epsilon,
                )
                if step_indices:
                    direct_probabilities = np.asarray(
                        _softmax(direct_step_logits, distill_temperature),
                        dtype=np.float64,
                    )
                    per_client_values = []
                    for model_logits, model_indices in zip(logits_by_model, indices_by_model):
                        model_values = np.zeros(len(step_indices), dtype=np.float64)
                        limit = min(len(model_logits), len(model_indices))
                        model_probabilities = _softmax(model_logits[:limit], distill_temperature)
                        index_to_offset = {int(token_id): offset for offset, token_id in enumerate(step_indices)}
                        for token_id, probability in zip(model_indices[:limit], model_probabilities):
                            offset = index_to_offset.get(int(token_id))
                            if offset is not None:
                                model_values[offset] = float(probability)
                        per_client_values.append(model_values)

                    decoded_sum = _mmlcc_decode_sum(
                        per_client_values,
                        num_blocks,
                        privacy_guarantee,
                        beta_radius,
                        noise_sigma,
                        noise_clip_theta,
                        rng,
                    )
                    decoded_probabilities = decoded_sum / max(len(per_client_values), 1)
                    decoded_probabilities = np.maximum(decoded_probabilities, float(probability_epsilon))
                    decoded_probabilities = decoded_probabilities / decoded_probabilities.sum()
                    direct_probabilities = np.maximum(direct_probabilities, float(probability_epsilon))
                    direct_probabilities = direct_probabilities / direct_probabilities.sum()

                    diff = decoded_probabilities - direct_probabilities
                    error_stats["squared_error"] += float(np.dot(diff, diff))
                    error_stats["squared_norm"] += float(np.dot(direct_probabilities, direct_probabilities))
                    error_stats["positions"] += 1
                    step_logits = [
                        float(distill_temperature) * math.log(max(float(p), float(probability_epsilon)))
                        for p in decoded_probabilities
                    ]
                else:
                    step_logits = direct_step_logits
                row_logits.append(step_logits)
                row_indices.append(step_indices)

            metrics = [
                examples[f"{ALIGNED_OTHER_METRIC}_{model_idx}"][row_idx]
                for model_idx in range(blending_num)
            ]
            aggregated_logits.append(row_logits)
            aggregated_indices.append(row_indices)
            aggregated_metrics.append(_mean_metric(metrics))

        examples[f"{ALIGNED_OTHER_LOGITS}_0"] = aggregated_logits
        examples[f"{ALIGNED_OTHER_INDICES}_0"] = aggregated_indices
        examples[f"{ALIGNED_OTHER_METRIC}_0"] = aggregated_metrics
        return examples

    aggregated_dataset = aligned_dataset.map(
        aggregate_batch,
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="Aggregate aligned SLM teachers with MMLCC-style union support.",
    )

    columns_to_remove = []
    for idx in range(1, blending_num):
        columns_to_remove.extend(
            [
                f"{ALIGNED_OTHER_LOGITS}_{idx}",
                f"{ALIGNED_OTHER_INDICES}_{idx}",
                f"{ALIGNED_OTHER_METRIC}_{idx}",
            ]
        )
    columns_to_remove = [col for col in columns_to_remove if col in aggregated_dataset.column_names]
    if columns_to_remove:
        aggregated_dataset = aggregated_dataset.remove_columns(columns_to_remove)

    relative_error = math.sqrt(error_stats["squared_error"]) / (
        math.sqrt(error_stats["squared_norm"]) + 1e-12
    )
    metrics = {
        "relative_error": float(relative_error),
        "positions": int(error_stats["positions"]),
        "privacy_guarantee": int(max(int(privacy_guarantee), 0)),
        "num_blocks": int(num_blocks),
    }

    logger.info(
        "Aggregated %s aligned SLM teachers into one teacher with union support.",
        blending_num,
    )
    return aggregated_dataset, metrics
