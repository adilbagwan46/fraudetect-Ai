from __future__ import annotations

import hashlib

import pandas as pd


def _bucket(value: str, namespace: str, seed: str, buckets: int) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def add_demo_relationship_fields(
    frame: pd.DataFrame,
    *,
    seed: str,
    device_buckets: int,
    ip_buckets: int,
    source_kind: str,
) -> pd.DataFrame:
    """Add reproducible demo identifiers without using the fraud label.

    Bucket collisions intentionally create shared relationships. These fields are
    synthetic product-demo data and must not be described as PaySim source fields.
    """

    if device_buckets <= 0 or ip_buckets <= 0:
        raise ValueError("Relationship bucket counts must be positive")

    enriched = frame.copy()
    enriched["device_id"] = enriched["customer_id"].map(
        lambda value: f"DEV-{_bucket(str(value), 'device', seed, device_buckets):05d}"
    )
    enriched["ip_id"] = enriched["customer_id"].map(
        lambda value: f"IP-{_bucket(str(value), 'ip', seed, ip_buckets):05d}"
    )
    enriched["data_provenance"] = f"{source_kind}_with_synthetic_relationship_enrichment"
    return enriched
