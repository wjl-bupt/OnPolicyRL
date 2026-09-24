"""RolloutBuffer tests -- the buffer against its protocol declarations.

What is pinned here:
1. the protocol linkage: schema-driven allocation, sample() yields the declared
   NamedTuple, declaration mistakes die at construction,
2. write semantics: write ops (STORE/ACCUMULATE/LAST), unknown-field rejection,
   two-phase write + advance, bootstrap slot, estimator slots,
3. sampling: invalid (autoreset) steps dropped, deterministic under a generator,
   WHOLE extras passed through uncut,
4. time structure: segments() respects terminated/truncated/valid.
"""

import torch
import pytest

from protocol.buffer import Field, SampleOp as FieldSampleOp  # noqa: F401
from protocol.buffer import WriteOp, base_schema, merge
from protocol.sample import SampleMapping, SampleOp, MinibatchSpec, RolloutSamples
from memory.buffer import RolloutBuffer

T, N, ACT, OBS = 16, 4, 3, 4   # CartPole: obs 4-dim, 3 actions


def _schema(extra=()):
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    sch = base_schema(env.observation_space, env.action_space)
    for name, f in extra:
        sch[name] = f
    return sch


def _fill(buf: RolloutBuffer, term_at=None, invalid_at=None, seed=0):
    """Simulate one collect: write every step, mark bootstrap + estimates."""
    g = torch.Generator().manual_seed(seed)
    for t in range(buf.T):
        buf.write_obs(torch.randn(buf.N, OBS, generator=g))
        buf.write(action=torch.randint(0, ACT, (buf.N,), generator=g),
                  logprob=torch.randn(buf.N, generator=g),
                  value=torch.randn(buf.N, generator=g))
        rew = torch.randn(buf.N, generator=g)
        if invalid_at is not None and t == invalid_at:
            buf.write(reward=rew)
            buf.write_masks(_masks(t, term=False, trunc=False, valid=False))
        else:
            term = term_at is not None and t == term_at
            buf.write(reward=rew)
            buf.write_masks(_masks(t, term=term, trunc=False, valid=True))
        buf.advance()
    buf.set_bootstrap_value(torch.randn(buf.N, generator=g))
    buf.set_estimates(torch.randn(buf.T, buf.N, generator=g),
                      torch.randn(buf.T, buf.N, generator=g))


def _masks(t, term=False, trunc=False, valid=True):
    class M:
        pass

    m = M()
    m.terminated = torch.zeros(N, dtype=torch.bool)
    m.truncated = torch.zeros(N, dtype=torch.bool)
    m.valid = torch.full((N,), valid, dtype=torch.bool)
    if term:
        m.terminated[:] = True
    if trunc:
        m.truncated[:] = True
    return m


# --------------------------------------------------------------------------- #
#  protocol linkage
# --------------------------------------------------------------------------- #


def test_sample_yields_the_declared_named_tuple():
    buf = RolloutBuffer(T, N, _schema())
    _fill(buf)
    out = list(buf.sample())
    assert out, "no minibatches"
    mb = out[0]
    assert isinstance(mb, RolloutSamples)
    assert mb._fields == ("observations", "actions", "old_values", "old_log_prob",
                          "advantages", "returns")
    total = sum(mb.actions.shape[0] for mb in out)
    assert total == T * N, "every valid sample appears exactly once"
    assert mb.observations.shape[1] == OBS and mb.actions.dtype == torch.long


def test_declaration_mistakes_die_at_construction():
    with pytest.raises(ValueError, match="not in the schema"):
        RolloutBuffer(T, N, _schema(), mapping=SampleMapping(actions="act"))
    with pytest.raises(ValueError, match="not declared in the buffer schema"):
        RolloutBuffer(T, N, _schema(), extra_samples=("probs",))


def test_whole_extra_passes_through_uncut():
    buf = RolloutBuffer(T, N, _schema(extra=[("anchor", Field((), torch.float32,
                                                            sample_op=FieldSampleOp.WHOLE))]),
                        extra_samples=("anchor",))
    _fill(buf)
    mb = next(iter(buf.sample()))
    assert torch.equal(mb.anchor, buf["anchor"])            # the whole tensor
    assert "anchor" not in buf._mapping.sources()


def test_core_sample_field_must_be_flatten():
    sch = _schema()
    sch["action"] = Field((), torch.long, sample_op=FieldSampleOp.WHOLE)
    with pytest.raises(ValueError, match="cut per sample in flat mode"):
        RolloutBuffer(T, N, sch)


def test_sequence_mode_is_not_declared_yet():
    with pytest.raises(ValueError, match="mode must be"):
        RolloutBuffer(T, N, _schema(), spec=MinibatchSpec(mode="sequence"))


# --------------------------------------------------------------------------- #
#  write semantics
# --------------------------------------------------------------------------- #


def test_unknown_field_fails_with_known_list():
    buf = RolloutBuffer(T, N, _schema())
    with pytest.raises(KeyError, match="known:"):
        buf.write(log_prob=torch.zeros(N))   # typo: schema says `logprob`


def test_accumulate_and_last_write_ops():
    sch = _schema(extra=[("shaped", Field((), torch.float32, write_op=WriteOp.ACCUMULATE)),
                         ("dbg", Field((), torch.float32, write_op=WriteOp.LAST))])
    buf = RolloutBuffer(T, N, sch)
    buf.write(shaped=torch.ones(N))
    buf.write(shaped=torch.ones(N))          # accumulates in the same slot
    buf.write(dbg=torch.full((N,), 1.0))
    buf.write(dbg=torch.full((N,), 2.0))     # only the last write survives
    assert torch.equal(buf["shaped"][0], torch.full((N,), 2.0))
    assert torch.equal(buf["dbg"][0], torch.full((N,), 2.0))


def test_two_phase_write_and_full_guard():
    buf = RolloutBuffer(T, N, _schema())
    buf.write(action=torch.zeros(N, dtype=torch.long))   # before env.step
    buf.write(reward=torch.zeros(N))                     # after env.step
    buf.advance()
    assert buf.pos == 1
    for _ in range(T - 1):
        buf.advance()
    with pytest.raises(RuntimeError, match="buffer is full"):
        buf.write(reward=torch.zeros(N))


def test_bootstrap_and_estimate_slots():
    buf = RolloutBuffer(T, N, _schema())
    _fill(buf)
    assert buf["value"].shape[0] == T + 1, "value keeps the bootstrap slot"
    assert buf.bootstrap_value.shape == (N,)
    assert buf.values.shape == (T, N)
    with pytest.raises(ValueError, match="estimate shape"):
        buf.set_estimates(torch.zeros(T + 1, N), torch.zeros(T, N))


# --------------------------------------------------------------------------- #
#  sampling
# --------------------------------------------------------------------------- #


def test_invalid_autoreset_steps_are_dropped():
    buf = RolloutBuffer(T, N, _schema())
    _fill(buf, invalid_at=5)
    out = list(buf.sample())
    total = sum(mb.actions.shape[0] for mb in out)
    assert total == T * N - N, "the whole dummy step (all N envs) is excluded"


def test_sample_is_deterministic_under_a_generator():
    a, b = RolloutBuffer(T, N, _schema()), RolloutBuffer(T, N, _schema())
    _fill(a, seed=1)
    _fill(b, seed=1)
    ga = torch.Generator().manual_seed(0)
    gb = torch.Generator().manual_seed(0)
    ma = [mb.observations for mb in a.sample(generator=ga)]
    mb_b = [mb.observations for mb in b.sample(generator=gb)]
    assert len(ma) == len(mb_b)
    assert all(torch.equal(x, y) for x, y in zip(ma, mb_b))


def test_advantages_are_estimator_slots_not_writable():
    buf = RolloutBuffer(T, N, _schema())
    with pytest.raises(KeyError, match="undeclared"):
        buf.write(advantages=torch.zeros(N))


def test_sample_elements_are_aligned_by_name():
    """Regression for a real bug: the assembly must place tensors BY ELEMENT
    NAME. A positional zip once swapped old_values/old_log_prob (the mapping's
    dataclass order differs from the NamedTuple's declaration order) -- nothing
    raised, the policy just collapsed. Pin: the value element must be the
    marker computed from the SAME rows as the observations element."""
    buf = RolloutBuffer(T, N, _schema())
    g = torch.Generator().manual_seed(0)
    for t in range(T):
        obs = torch.randn(N, OBS, generator=g)
        buf.write_obs(obs)
        buf.write(action=torch.randint(0, ACT, (N,), generator=g),
                  logprob=torch.randn(N, generator=g),
                  value=obs.sum(-1) + 100.0)            # row marker
        buf.write_masks(_masks(t))
        buf.advance()
    buf.set_bootstrap_value(torch.zeros(N))
    buf.set_estimates(torch.zeros(T, N), torch.zeros(T, N))
    for data in buf.sample():
        assert torch.allclose(data.old_values, data.observations.sum(-1) + 100.0), \
            "old_values and observations came from different rows"


# --------------------------------------------------------------------------- #
#  trajectory mode
# --------------------------------------------------------------------------- #


def test_trajectory_mode_packs_whole_trajectories_to_target():
    buf = RolloutBuffer(T, N, _schema(),
                        spec=MinibatchSpec(mode="trajectory", trajectory_frames=12,
                                           min_trajectories=0))
    _fill(buf, term_at=5)      # per env: [0..6) terminated + [6..T) -> lengths 6,10
    out = list(buf.sample(generator=torch.Generator().manual_seed(0)))
    from protocol.sample import TrajectorySamples
    assert out and all(isinstance(mb, TrajectorySamples) for mb in out)
    # every batch reaches the target; nothing is dropped; all frames accounted for
    assert all(int(mb.lengths.sum()) >= 12 for mb in out)
    assert sum(int(mb.lengths.sum()) for mb in out) == T * N
    assert all(mb.actions.shape[0] == int(mb.lengths.sum()) for mb in out)
    assert {int(x) for mb in out for x in mb.lengths} == {6, 10}


def test_trajectory_last_values_zeroed_on_true_termination():
    buf = RolloutBuffer(T, N, _schema(),
                        spec=MinibatchSpec(mode="trajectory", trajectory_frames=12,
                                           min_trajectories=0))
    _fill(buf, term_at=5)
    out = list(buf.sample(generator=torch.Generator().manual_seed(1)))
    for mb in out:
        for length, lv in zip(mb.lengths, mb.last_values):
            if int(length) == 6:        # ended on a true termination
                assert float(lv) == 0.0
            else:                        # ran to the boundary -> bootstrap value
                assert float(lv) != 0.0


def test_trajectory_mode_drops_small_trailing_batch():
    """Drop rule, deterministically: the target (100) exceeds the whole rollout
    (64 frames), so the trailing batch is ALL trajectories; with
    min_trajectories=9 it is discarded and sample() yields nothing."""
    buf = RolloutBuffer(T, N, _schema(),
                        spec=MinibatchSpec(mode="trajectory", trajectory_frames=100,
                                           min_trajectories=9))
    _fill(buf, term_at=5)      # 8 trajectories (N=4 envs x 2 segments), 64 frames
    out = list(buf.sample(generator=torch.Generator().manual_seed(0)))
    assert out == []


def test_trajectory_mode_keeps_everything_when_drop_disabled():
    buf = RolloutBuffer(T, N, _schema(),
                        spec=MinibatchSpec(mode="trajectory", trajectory_frames=100,
                                           min_trajectories=0))
    _fill(buf, term_at=5)
    out = list(buf.sample(generator=torch.Generator().manual_seed(0)))
    assert len(out) == 1                                    # one batch, no mid cut
    assert out[0].lengths.shape[0] == 8 and int(out[0].lengths.sum()) == T * N


def test_trajectory_batches_respect_target_and_drop_rule():
    spec = MinibatchSpec(mode="trajectory", trajectory_frames=20, min_trajectories=3)
    buf = RolloutBuffer(T, N, _schema(), spec=spec)
    _fill(buf, term_at=5)
    for seed in range(8):
        out = list(buf.sample(generator=torch.Generator().manual_seed(seed)))
        assert sum(int(mb.lengths.sum()) for mb in out) <= T * N
        for i, mb in enumerate(out):
            frames, n_traj = int(mb.lengths.sum()), mb.lengths.shape[0]
            if i < len(out) - 1:                 # mid batches pack to the target
                assert frames >= 20
            else:                                # trailing: kept only if big enough
                assert n_traj >= 3 or frames >= 20


def test_trajectory_frames_none_derives_from_num_minibatches():
    buf = RolloutBuffer(T, N, _schema(),
                        spec=MinibatchSpec(mode="trajectory", num_minibatches=4,
                                           min_trajectories=0))
    _fill(buf, term_at=5)      # total 32 -> target 32/4 = 8
    out = list(buf.sample(generator=torch.Generator().manual_seed(2)))
    assert sum(int(mb.lengths.sum()) for mb in out) == T * N
    assert all(int(mb.lengths.sum()) >= 8 for mb in out)


def test_trajectory_mode_extras_follow_sample_op():
    from protocol.sample import TrajectorySamples, sample_class
    sch = _schema(extra=[("anchor", Field((), torch.float32,
                                          sample_op=FieldSampleOp.WHOLE))])
    buf = RolloutBuffer(T, N, sch,
                        spec=MinibatchSpec(mode="trajectory", trajectory_frames=12,
                                           min_trajectories=0),
                        extra_samples=("anchor",))
    assert buf._samples_cls.__name__ == "TrajectorySamples"
    assert "anchor" in buf._samples_cls._fields
    _fill(buf, term_at=5)
    mb = next(iter(buf.sample(generator=torch.Generator().manual_seed(3))))
    assert torch.equal(mb.anchor, buf["anchor"])


def test_trajectory_mode_accepts_non_flatten_core():
    """In trajectory mode the per-sample FLATTEN restriction is lifted: time is
    preserved inside trajectories, so WHOLE core fields are legal."""
    sch = _schema()
    sch["action"] = Field((), torch.long, sample_op=FieldSampleOp.WHOLE)
    buf = RolloutBuffer(T, N, sch, spec=MinibatchSpec(mode="trajectory"))
    _fill(buf)


# --------------------------------------------------------------------------- #
#  time structure
# --------------------------------------------------------------------------- #


def test_segments_cut_at_termination_and_invalid():
    buf = RolloutBuffer(T, N, _schema())
    _fill(buf, term_at=5, invalid_at=6)
    segs = buf.segments()
    # every env: [0..5] terminated at 5, t=6 invalid (no segment), [7..T)
    assert (N, 0, 6) and all((n, 0, 6) in segs for n in range(N))
    assert all((n, 6, s) not in segs for n in range(N) for s in range(T))
    assert all((n, 7, T) in segs for n in range(N))


def test_describe_lists_every_field():
    buf = RolloutBuffer(T, N, _schema())
    text = buf.describe()
    for name in ("obs", "action", "logprob", "reward", "value", "terminated",
                 "truncated", "valid", "advantages"):
        assert name in text
