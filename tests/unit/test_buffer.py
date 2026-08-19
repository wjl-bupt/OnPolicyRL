"""Buffer, schema and declarative-operator tests."""

import pytest
import torch

import oprl
from oprl.schema import Op


def test_buffer_operators():
    """Declarative operators: Op.WHOLE is never sliced, Op.ACCUMULATE adds in place."""
    env = oprl.make_env("CartPole-v1", num_envs=2, seed=1)
    buf = oprl.RolloutBuffer(
        4, 2, env.obs_space, env.action_space,
        extra={
            "whole_field": oprl.Field((), torch.long, sample_op=Op.WHOLE),
            "acc_field": oprl.Field((), torch.float32, write_op=Op.ACCUMULATE),
        },
    )
    assert "whole_field" in buf.describe()
    # ACCUMULATE: two writes within one step must sum.
    buf.write(acc_field=torch.ones(2))
    buf.write(acc_field=torch.ones(2))
    assert torch.allclose(buf["acc_field"][0], torch.full((2,), 2.0))
    env.close()


def test_unknown_field_rejected():
    """A misspelled field fails at write time, not three hours into training."""
    env = oprl.make_env("CartPole-v1", num_envs=2, seed=1)
    buf = oprl.RolloutBuffer(4, 2, env.obs_space, env.action_space)
    with pytest.raises(KeyError, match="undeclared fields"):
        buf.write(typo_field=torch.zeros(2))
    env.close()
