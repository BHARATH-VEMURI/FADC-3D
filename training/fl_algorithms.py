"""
Federated aggregation algorithms — v1: FedAvg and FedBN.

FedAvg (McMahan et al. 2017): weighted average of all parameters and buffers
across participating clients, weighted by the client's sample count.

FedBN (Li et al. ICLR 2021): same as FedAvg but BatchNorm layer state is kept
local per client (skipped during aggregation). Strong match for multi-site
MRI where each site has different intensity statistics — letting BN absorb
per-site shifts and sharing only the conv weights tends to improve non-IID
convergence on imaging benchmarks.

Both aggregators are model-agnostic — they work on state_dict keys. FedBN
identifies BN layers at runtime via isinstance checks, so any
BatchNorm{1,2,3}d module is detected regardless of name.
"""
from typing import Sequence

import torch
import torch.nn as nn


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def get_bn_state_dict_keys(model: nn.Module) -> set[str]:
    """All state_dict keys belonging to BatchNorm modules.

    Covers learnable params (weight, bias) AND non-learnable buffers
    (running_mean, running_var, num_batches_tracked). FedBN must skip ALL of
    these — sharing running stats across clients with different intensity
    distributions defeats the whole point.
    """
    bn_keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _BN_TYPES):
            prefix = f"{module_name}." if module_name else ""
            for pname, _ in module.named_parameters(recurse=False):
                bn_keys.add(prefix + pname)
            for bname, _ in module.named_buffers(recurse=False):
                bn_keys.add(prefix + bname)
    return bn_keys


def fedavg_aggregate(
    client_states: Sequence[dict[str, torch.Tensor]],
    client_weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Sample-weighted average over all keys in the state_dicts.

    All state_dicts must share the same keys and tensor shapes — they came
    from copies of the same model architecture.

    Integer buffers (e.g. num_batches_tracked) cannot be averaged with a float
    weighted sum because the dtype is int64. We take the max across clients
    instead — preserves the "number of forward passes this BN has seen" semantic
    in a way that is meaningful after aggregation.
    """
    assert len(client_states) == len(client_weights) and len(client_states) > 0
    total_weight = sum(client_weights)
    assert total_weight > 0, "client_weights must sum to > 0"
    norm_weights = [w / total_weight for w in client_weights]

    reference = client_states[0]
    aggregated: dict[str, torch.Tensor] = {}

    for key in reference.keys():
        ref_tensor = reference[key]

        if _is_integer_buffer(ref_tensor):
            stacked = torch.stack([s[key] for s in client_states])
            aggregated[key] = stacked.max(dim=0).values
            continue

        acc = torch.zeros_like(ref_tensor, dtype=torch.float32)
        for state, w in zip(client_states, norm_weights):
            acc.add_(state[key].to(torch.float32), alpha=w)
        aggregated[key] = acc.to(ref_tensor.dtype)

    return aggregated


def fedbn_aggregate(
    client_states: Sequence[dict[str, torch.Tensor]],
    client_weights: Sequence[float],
    bn_keys: set[str],
    prior_global_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Average non-BN params (FedAvg), keep BN params local per client.

    Since the orchestrator stores a single global state_dict, "keep local"
    means: for BN keys, leave the global state UNCHANGED — the round's BN
    updates have already been written to each client's local checkpoint (which
    the orchestrator persists per-client between rounds).

    Args:
      client_states:        full state_dicts from each participating client
      client_weights:       sample counts (or any positive weight)
      bn_keys:              keys to skip during aggregation
      prior_global_state:   the global state_dict BEFORE this round's local
                            training; used as fallback for BN keys

    The orchestrator is responsible for re-injecting each client's persisted
    BN state when that client trains next round.
    """
    fedavg = fedavg_aggregate(client_states, client_weights)
    for key in bn_keys:
        if key in prior_global_state:
            fedavg[key] = prior_global_state[key].clone()
    return fedavg


def _is_integer_buffer(t: torch.Tensor) -> bool:
    return t.dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8)


def split_bn_state(
    state: dict[str, torch.Tensor],
    bn_keys: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Partition a state_dict into (shared_state, bn_state) — useful for
    persisting per-client BN between rounds in FedBN.
    """
    shared = {k: v for k, v in state.items() if k not in bn_keys}
    bn     = {k: v for k, v in state.items() if k in bn_keys}
    return shared, bn


def merge_bn_state(
    base_state: dict[str, torch.Tensor],
    bn_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Overlay bn_state onto base_state. Used to inject per-client BN before
    that client trains in FedBN. base_state is mutated in-place and returned.
    """
    for k, v in bn_state.items():
        base_state[k] = v
    return base_state


if __name__ == "__main__":
    # Synthetic check on UNet3D — verifies BN key detection + FedAvg/FedBN
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.unet_3d import UNet3D

    model = UNet3D(in_channels=2, out_channels=2, base_filters=4)
    bn_keys = get_bn_state_dict_keys(model)
    print(f"Detected {len(bn_keys)} BN state_dict keys")
    for k in sorted(bn_keys)[:4]:
        print(f"  {k}")
    print(f"  ... ({len(bn_keys) - 4} more)" if len(bn_keys) > 4 else "")

    # Two synthetic clients with slightly perturbed weights
    state_a = {k: v.clone() for k, v in model.state_dict().items()}
    state_b = {k: v.clone() + 0.1 for k, v in model.state_dict().items()
               if v.dtype.is_floating_point}
    # Restore integer buffers untouched in state_b
    for k, v in model.state_dict().items():
        if not v.dtype.is_floating_point:
            state_b[k] = v.clone()

    agg = fedavg_aggregate([state_a, state_b], [100, 200])
    sample_key = next(iter(agg.keys()))
    print(f"\nFedAvg: aggregated dict has {len(agg)} keys (matches model: {len(model.state_dict())})")

    prior = {k: v.clone() for k, v in model.state_dict().items()}
    agg_bn = fedbn_aggregate([state_a, state_b], [100, 200], bn_keys, prior)
    diffs_in_bn = sum(
        not torch.equal(agg_bn[k], prior[k]) for k in bn_keys
    )
    print(f"FedBN: BN keys that were modified (should be 0): {diffs_in_bn}")
    print("OK." if diffs_in_bn == 0 else "FAIL — BN keys leaked into aggregation.")
