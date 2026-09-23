from types import SimpleNamespace

import pytest
import torch

from frontier.training import trainer


@pytest.mark.parametrize(("status", "expected_clean"), [("", True), (" M src/model.py\n", False)])
def test_environment_records_git_worktree_state(monkeypatch, status, expected_clean):
    def fake_run(command, **kwargs):
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout="C:/repo\n")
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        if command == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout=status)
        raise AssertionError(f"unexpected subprocess command: {command}")

    monkeypatch.setattr(
        trainer,
        "subprocess",
        SimpleNamespace(run=fake_run, CalledProcessError=RuntimeError),
    )

    environment = trainer._environment(torch.device("cpu"))

    assert environment["git_revision"] == "abc123"
    assert environment["git_worktree_clean"] is expected_clean
