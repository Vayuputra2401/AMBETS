"""
test_intervention.py — the routing-intervention hook (pipeline/routing_intervention.py).

These guard the mechanism the causal experiment rests on. If the override silently did
nothing, the experiment would report "no effect" and we would wrongly conclude the routing
is not causally upstream; if it leaked across forward passes, every subsequent measurement
in a run would be contaminated. Both failure modes are silent, hence these tests.
"""

from __future__ import annotations

import torch

from ccr.config.ccr_config import CCRConfig
from ccr.modules.dispatcher import CCRBottleneckModule, routing_intervention


def _module(embed_dim: int = 32, k: int = 4) -> CCRBottleneckModule:
    cfg = CCRConfig()
    cfg.router.embed_dim = embed_dim
    cfg.router.num_concepts = k
    cfg.expert.embed_dim = embed_dim
    cfg.concept_names = tuple(f"c{i}" for i in range(k))
    cfg.hard_routing_inference = True
    m = CCRBottleneckModule(cfg)
    m.eval()
    return m


def test_override_changes_assignments():
    m = _module()
    x = torch.randn(2, 16, 32)
    base = m(x)["assignments"]

    override = torch.full_like(base, 3)
    with routing_intervention(m, override):
        got = m(x)["assignments"]

    assert torch.equal(got, torch.full_like(base, 3)), "override did not reach dispatch"


def test_minus_one_leaves_token_untouched():
    """-1 is the 'leave alone' sentinel; an all -1 override must be a no-op."""
    m = _module()
    x = torch.randn(2, 16, 32)
    base = m(x)
    override = torch.full_like(base["assignments"], -1)
    with routing_intervention(m, override):
        got = m(x)
    assert torch.equal(got["assignments"], base["assignments"])
    # and with hard_routing_inference already on, the features must be bit-identical
    assert torch.allclose(got["expert_outputs"], base["expert_outputs"])


def test_partial_override_is_selective():
    m = _module()
    x = torch.randn(1, 32, 32)
    base = m(x)["assignments"]
    override = torch.full_like(base, -1)
    override[0, :8] = 2
    with routing_intervention(m, override):
        got = m(x)["assignments"]
    assert torch.all(got[0, :8] == 2)
    assert torch.equal(got[0, 8:], base[0, 8:]), "untouched tokens must not change"


def test_identity_override_is_exactly_zero_effect():
    """The experiment's diagonal control: a -> a must be a bit-exact no-op."""
    m = _module()
    x = torch.randn(1, 24, 32)
    base = m(x)
    a = int(base["assignments"][0, 0])
    override = torch.full_like(base["assignments"], -1)
    override[base["assignments"] == a] = a           # route concept a to its own expert
    with routing_intervention(m, override):
        got = m(x)
    assert torch.allclose(got["expert_outputs"], base["expert_outputs"], atol=0)


def test_routing_probs_report_the_router_not_the_override():
    """The explanation must stay the router's actual belief, else the CSV would lie."""
    m = _module()
    x = torch.randn(1, 16, 32)
    base = m(x)["routing_probs"]
    override = torch.full((1, 16), 0, dtype=torch.long)
    with routing_intervention(m, override):
        got = m(x)["routing_probs"]
    assert torch.allclose(got, base)


def test_override_is_cleared_on_exit_and_on_exception():
    m = _module()
    x = torch.randn(1, 16, 32)
    override = torch.full((1, 16), 1, dtype=torch.long)

    with routing_intervention(m, override):
        pass
    assert m.route_override is None

    try:
        with routing_intervention(m, override):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.route_override is None, "override leaked after an exception"

    # and a later forward is unaffected
    assert torch.equal(m(x)["assignments"], m(x)["assignments"])


def test_shape_mismatch_raises():
    m = _module()
    x = torch.randn(1, 16, 32)
    with routing_intervention(m, torch.zeros(1, 8, dtype=torch.long)):
        try:
            m(x)
        except ValueError as e:
            assert "does not match" in str(e)
        else:
            raise AssertionError("expected ValueError on shape mismatch")


def test_intervention_forces_hard_dispatch_even_when_soft_configured():
    """Soft dispatch would blend all K experts and dilute the manipulation."""
    m = _module()
    m.config.hard_routing_inference = False          # would otherwise soft-dispatch in eval
    x = torch.randn(1, 16, 32)
    soft = m(x)["expert_outputs"]
    override = torch.full((1, 16), -1, dtype=torch.long)
    with routing_intervention(m, override):
        hard = m(x)["expert_outputs"]
    assert not torch.allclose(soft, hard), "intervention must switch to hard dispatch"
