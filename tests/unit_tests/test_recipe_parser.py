# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for ``tests/test_utils/python_scripts/recipe_parser.py``.

These guard the scope-filter selection logic that turns recipe rows into JET
workloads. They run as plain in-process assertions (no GPU, no distributed
init) by exercising the flatten + filter pipeline on synthetic manifests.
"""
from typing import List, Optional

from tests.test_utils.python_scripts import recipe_parser
from tests.test_utils.python_scripts.recipe_parser import dotdict

TEST_CASE = "hybrid_dynamic_inference_tp1_pp1_dp8_583m_flashinfer"


def _flatten(manifest: dict) -> List[dotdict]:
    """Reproduce ``load_and_flatten`` from an in-memory manifest (no disk I/O)."""
    return recipe_parser.set_build_dependency(
        recipe_parser.flatten_workload(recipe_parser.flatten_products(dotdict(**manifest)))
    )


def _manifest(
    scope: List[str],
    test_case: str = TEST_CASE,
    environment: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
) -> dict:
    """Build a minimal single-product recipe manifest for the given scope tiers."""
    return {
        "type": "basic",
        "format_version": 1,
        "maintainers": ["mcore"],
        "loggers": ["stdout"],
        "spec": {
            "name": "{test_case}_{environment}_{platforms}",
            "model": "hybrid",
            "build": "mcore-pyt-{environment}",
            "nodes": 1,
            "gpus": 1,
            "n_repeat": 1,
            "platforms": "dgx_a100",
            "script": "echo run",
        },
        "products": [
            {
                "test_case": [test_case],
                "products": [
                    {
                        "environment": environment or ["dev"],
                        "scope": scope,
                        "platforms": platforms or ["dgx_h100"],
                    }
                ],
            }
        ],
    }


def test_multi_tier_scope_deduped_under_union_filter():
    """A ``scope: [L1, L2]`` row matched by ``--scope L1,L2`` yields one workload.

    Regression for the empty-workload bug: the row flattens into an L1 row and
    an L2 row (identical except for scope); both match the union filter; without
    dedup ``filter_by_test_case`` sees a duplicate and returns ``None``, so
    ``load_workloads`` yields an empty JET pipeline.
    """
    workloads = _flatten(_manifest(scope=["L1", "L2"]))
    assert len(workloads) == 2

    filtered = recipe_parser.filter_by_scope(workloads, "L1,L2")
    assert len(filtered) == 1

    match = recipe_parser.filter_by_test_case(filtered, TEST_CASE)
    assert match is not None
    assert match.spec["test_case"] == TEST_CASE


def test_single_tier_scope_still_selected():
    """A single-tier ``scope: [L2]`` row remains selectable by the union filter."""
    workloads = _flatten(_manifest(scope=["L2"]))

    filtered = recipe_parser.filter_by_scope(workloads, "L1,L2")
    assert len(filtered) == 1
    assert recipe_parser.filter_by_test_case(filtered, TEST_CASE) is not None


def test_multi_tier_scope_deduped_per_environment():
    """Dedup collapses scope duplicates without merging distinct environments."""
    workloads = _flatten(_manifest(scope=["L1", "L2"], environment=["dev", "lts"]))
    assert len(workloads) == 4

    filtered = recipe_parser.filter_by_scope(workloads, "L1,L2")
    assert sorted(workload.spec["environment"] for workload in filtered) == ["dev", "lts"]


def test_distinct_cadence_rows_not_deduped():
    """Rows differing beyond scope (cadence) are preserved, not collapsed.

    ``scope: [mr, nightly]`` aliases to ``L2`` (default cadence) and ``L3``
    (``nightly`` cadence); a filter selecting both tiers must keep both rows.
    """
    workloads = _flatten(_manifest(scope=["mr", "nightly"]))

    filtered = recipe_parser.filter_by_scope(workloads, "L2,L3")
    assert len(filtered) == 2
    assert sorted(workload.spec["scope"] for workload in filtered) == ["L2", "L3"]
