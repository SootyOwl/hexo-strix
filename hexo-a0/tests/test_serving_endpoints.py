"""Smoke tests for the HTTP app and CLI wiring.

Full play-parity verification is the Task 1.8 manual play-through.
"""


import pytest


def test_app_exposes_run_and_handler():
    from hexo_a0.serving import app
    assert callable(app.run) and callable(app.make_handler_class)


import os


def test_serve_subcommand_registered(capsys):
    with pytest.raises(SystemExit):
        from hexo_a0 import cli
        cli.main(["serve", "--help"])
    assert "checkpoint" in capsys.readouterr().out


CKPT = "runs/gine-mini-4l-128p32v-jkcat/checkpoints/self_play/champion.pt"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="ckpt absent")
def test_analyze_ignores_client_params():
    from hexo_a0.serving.app import AnalyzeContext, handle_analyze
    ctx = AnalyzeContext.from_checkpoint(CKPT, win_length=6, placement_radius=8,
                                       max_moves=400, max_analyze_moves=220)
    status, body = handle_analyze({"moves": [[0, 0], [1, 0], [2, 0]],
                                   "checkpoint": "/etc/passwd", "win_length": 99}, ctx)
    assert status == 200 and "value" in body


@pytest.mark.skipif(not os.path.exists(CKPT), reason="ckpt absent")
def test_trajectory_rejects_overlong():
    from hexo_a0.serving.app import AnalyzeContext, handle_trajectory
    ctx = AnalyzeContext.from_checkpoint(CKPT, win_length=6, placement_radius=8,
                                       max_moves=400, max_analyze_moves=220)
    status, _ = handle_trajectory({"moves": [[0, 0]] + [[i, 0] for i in range(1, 500)]}, ctx)
    assert status == 400


@pytest.mark.skipif(not os.path.exists(CKPT), reason="ckpt absent")
def test_analyze_returns_candidate_set():
    from hexo_a0.serving.app import AnalyzeContext, handle_analyze
    ctx = AnalyzeContext.from_checkpoint(CKPT, win_length=6, placement_radius=8,
                                       max_moves=400, max_analyze_moves=220,
                                       mcts_sims=16, m_actions=16)
    status, body = handle_analyze({"moves": [[0, 0], [1, 0], [2, 0]]}, ctx)
    assert status == 200
    assert "candidate_set" in body and any(body["candidate_set"])
    assert "q_hat" in body and body["q_hat"] is not None
