"""Unit tests of the training-view fit check (no GPU needed)."""
import torch

from mvbrdf_shr.world.fit_check import DEFAULT_TAU_DB, accept_field, fuse_with_check


def test_accept_field_uses_the_threshold():
    assert DEFAULT_TAU_DB == 40.0
    assert accept_field(58.5) and accept_field(40.0)
    assert not accept_field(22.7) and not accept_field(39.99)
    assert accept_field(30.0, tau=25.0)


def test_fuse_with_check_selects_the_source():
    hybrid = torch.full((1, 3, 2, 2), 0.7)
    single = torch.full((1, 3, 2, 2), 0.2)
    assert torch.equal(fuse_with_check(hybrid, single, 58.0), hybrid)
    assert torch.equal(fuse_with_check(hybrid, single, 17.0), single)
