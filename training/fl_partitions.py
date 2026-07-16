"""
Federated Learning client partitioning.

Two strategies:
  - natural_partition: one client per collection (DUKE/ISPY1/ISPY2/NACT).
    This is the paper's headline FL setting — real multi-site heterogeneity.
  - iid_random_partition: shuffle all cases and split evenly into K clients.
    Sanity-check ablation — isolates FL infrastructure from non-IID effects.

Both return: {client_id: list[case_dict]} where each case_dict is the same
shape as discover_cases_from_cache() output (patient_id, collection, client_id).
"""
import random
from typing import Callable

from data.mama_mia_dataset import CLIENT_MAP, COLLECTIONS


def natural_partition(cases: list[dict]) -> dict[int, list[dict]]:
    """One client per collection. client_id matches CLIENT_MAP."""
    partition = {cid: [] for cid in CLIENT_MAP.values()}
    for c in cases:
        partition[c["client_id"]].append(c)
    return partition


def iid_random_partition(cases: list[dict], n_clients: int, seed: int) -> dict[int, list[dict]]:
    """Shuffle cases and split into n_clients near-equal shards.

    Uses an isolated random.Random instance so the global RNG state set by the
    seeding logic in train_federated.py is not perturbed.
    """
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)

    partition = {i: [] for i in range(n_clients)}
    for idx, case in enumerate(shuffled):
        partition[idx % n_clients].append(case)
    return partition


def partition_factory(name: str, n_clients: int, seed: int) -> Callable[[list[dict]], dict[int, list[dict]]]:
    """Return a partition function bound to its config.

    Centralizing the dispatch here keeps train_federated.py model/algo-agnostic.
    """
    if name == "natural":
        return lambda cases: natural_partition(cases)
    if name == "iid":
        return lambda cases: iid_random_partition(cases, n_clients=n_clients, seed=seed)
    raise ValueError(f"Unknown partition: {name!r}. Use 'natural' or 'iid'.")


def partition_summary(partition: dict[int, list[dict]]) -> str:
    """Compact one-line-per-client summary for the run log."""
    lines = []
    total = 0
    for cid in sorted(partition.keys()):
        cases = partition[cid]
        n = len(cases)
        total += n
        # Per-client collection counts (mostly homogeneous for natural, mixed for IID)
        by_col = {}
        for c in cases:
            by_col[c["collection"]] = by_col.get(c["collection"], 0) + 1
        col_str = ", ".join(f"{k}={v}" for k, v in sorted(by_col.items()))
        lines.append(f"  Client {cid}: {n:4d} cases | {col_str}")
    lines.append(f"  Total: {total} cases across {len(partition)} clients")
    return "\n".join(lines)


if __name__ == "__main__":
    # Synthetic smoke — build fake cases per collection, run both partitions.
    fake = []
    counts = {"DUKE": 200, "ISPY1": 120, "ISPY2": 700, "NACT": 50}
    for col, n in counts.items():
        for i in range(n):
            fake.append({
                "patient_id": f"{col.lower()}_{i:03d}",
                "collection": col,
                "client_id":  CLIENT_MAP[col],
            })

    print("Natural partition:")
    print(partition_summary(natural_partition(fake)))
    print()
    print("IID random partition (4 clients, seed=42):")
    print(partition_summary(iid_random_partition(fake, n_clients=4, seed=42)))
