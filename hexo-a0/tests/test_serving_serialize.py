"""state_to_dict serialization for the serving module."""
from datetime import datetime, timezone

import hexo_rs
from hexo_a0.serving.game import GameRecord
from hexo_a0.serving.serialize import state_to_dict


def _record():
    now = datetime.now(timezone.utc)  # GameRecord types created_at/last_active_at as datetime
    st = hexo_rs.GameState(hexo_rs.GameConfig(6, 8, 400))
    st.apply_move(1, 0)  # P2's stone (engine seeded P1 at origin → P2 moves first)
    return GameRecord(game_id="t", created_at=now, last_active_at=now, state=st,
                      human_side="P2", bot_side="P1",
                      move_log=[(0, 0, "P1"), (1, 0, "P2")], human_name="alice",
                      difficulty="standard")


def test_state_to_dict_shape():
    d = state_to_dict(_record(), placement_radius=8, win_length=6)
    assert d["terminal"] is False
    assert any(tuple(s[0]) == (1, 0) for s in d["stones"])
    assert d["current_player"] == "P2" and d["is_human_turn"] is True
    assert d["win_length"] == 6 and d["placement_radius"] == 8
    assert d["last_moves"] == [[1, 0]]
    assert d["difficulty"] == "standard"


def test_htttx_only_when_terminal_and_requested():
    # not terminal → htttx is None even when requested
    d = state_to_dict(_record(), placement_radius=8, win_length=6, include_htttx=True)
    assert d["htttx"] is None
