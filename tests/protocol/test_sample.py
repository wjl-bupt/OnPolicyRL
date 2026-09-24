"""Protocol-layer tests for the sampling declarations (src/protocol/sample.py).

Three layers, three contracts: element names (SB3-compatible), operation set
(closed enum), and the minibatch spec. The buffer implementation that consumes
these is held to exactly what is pinned here.
"""

import torch
import pytest

from protocol.sample import (
    MinibatchSpec,
    RolloutSamples,
    SampleMapping,
    SampleOp,
    TrajectorySamples,
    sample_class,
    validate_against_schema,
)

SCHEMA = {"obs": None, "action": None, "logprob": None, "value": None,
          "probs": None}


def test_rollout_samples_is_a_named_tuple():
    mb = RolloutSamples(
        observations=torch.randn(4, 2), actions=torch.zeros(4, dtype=torch.long),
        old_values=torch.randn(4), old_log_prob=torch.randn(4),
        advantages=torch.randn(4), returns=torch.randn(4))
    # SB3-compatible element names, attribute access and unpacking both work
    assert mb.observations.shape == (4, 2)
    obs, act, ov, olp, adv, ret = mb
    assert act.dtype == torch.long and adv.shape == (4,)


def test_sample_class_extends_without_touching_the_core():
    ext = sample_class(("probs",))("obs", None, None, None, None, None,
                                   probs=torch.rand(3, 5))
    assert ext._fields == (*RolloutSamples._fields, "probs")
    with pytest.raises(ValueError, match="must not collide"):
        sample_class(("observations",))
    with pytest.raises(ValueError, match="identifiers"):
        sample_class(("not-an-identifier",))


def test_sample_mapping_defaults_and_sources():
    m = SampleMapping()
    assert m.sources() == {
        "observations": "obs", "actions": "action", "old_log_prob": "logprob",
        "old_values": "value", "advantages": "advantages", "returns": "returns",
    }
    # estimator output slots must map to themselves -- they are not schema fields
    with pytest.raises(ValueError, match="must map to itself"):
        SampleMapping(returns="my_returns")


def test_validate_against_schema_catches_setup_time_errors():
    validate_against_schema(SampleMapping(), SCHEMA, extra=("probs",))
    with pytest.raises(ValueError, match="not in the schema"):
        validate_against_schema(SampleMapping(actions="act"), SCHEMA)
    with pytest.raises(ValueError, match="not declared in the buffer schema"):
        validate_against_schema(SampleMapping(), SCHEMA, extra=("anchor",))


def test_minibatch_spec_rejects_bad_modes_and_counts():
    assert MinibatchSpec().mode == "flat"
    with pytest.raises(ValueError, match="mode must be"):
        MinibatchSpec(mode="random")
    with pytest.raises(ValueError, match="mode must be"):
        MinibatchSpec(mode="sequence")   # recurrent blocks are not declared yet
    with pytest.raises(ValueError, match="num_minibatches"):
        MinibatchSpec(num_minibatches=0)


def test_minibatch_spec_trajectory_options_belong_to_trajectory_mode():
    spec = MinibatchSpec(mode="trajectory", trajectory_frames=512, min_trajectories=2)
    assert spec.trajectory_frames == 512
    with pytest.raises(ValueError, match="trajectory-mode options"):
        MinibatchSpec(mode="flat", trajectory_frames=512)
    with pytest.raises(ValueError, match="trajectory-mode options"):
        MinibatchSpec(mode="flat", min_trajectories=0)
    with pytest.raises(ValueError, match="trajectory_frames"):
        MinibatchSpec(mode="trajectory", trajectory_frames=0)


def test_trajectory_samples_declaration():
    assert TrajectorySamples._fields == (
        "observations", "actions", "old_values", "old_log_prob",
        "advantages", "returns", "lengths", "last_values")
    ext = sample_class(("probs",), base=TrajectorySamples)
    assert ext._fields == (*TrajectorySamples._fields, "probs")
    with pytest.raises(ValueError, match="must not collide"):
        sample_class(("lengths",), base=TrajectorySamples)


def test_sample_op_is_a_closed_set():
    assert {op.value for op in SampleOp} == {"flatten", "whole", "sequence"}
