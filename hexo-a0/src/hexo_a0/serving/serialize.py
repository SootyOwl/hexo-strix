"""GameRecord → API state dict (ported from scripts/play_server.py:406-453)."""
from .game import GameRecord
from .htttx import serialize_htttx


def state_to_dict(rec: GameRecord, *, placement_radius: int, win_length: int,
                  include_htttx: bool = False) -> dict:
    """Serialize GameRecord into the API state shape from the spec."""
    state = rec.state
    stones = [[[int(q), int(r)], p] for ((q, r), p) in state.placed_stones()]
    is_terminal_engine = state.is_terminal()
    terminal = is_terminal_engine or rec.terminal_recorded
    if is_terminal_engine:
        winner = state.winner()
        result_type = "win" if winner is not None else "max_moves"
    elif rec.terminal_recorded:
        winner = rec.bot_side
        result_type = "resign"
    else:
        winner = None
        result_type = None
    # last_moves: stones placed in the most recent turn (≤2 same-player, latest first→reversed).
    last_moves: list[list[int]] = []
    if rec.move_log:
        latest_player = rec.move_log[-1][2]
        for q, r, p in reversed(rec.move_log):
            if p != latest_player:
                break
            last_moves.append([q, r])
            if len(last_moves) == 2:
                break
        last_moves.reverse()
    cur = state.current_player() if not terminal else None
    is_human_turn = (cur == rec.human_side) and not terminal
    return {
        "game_id": rec.game_id,
        "stones": stones,
        "human_side": rec.human_side,
        "human_name": rec.human_name,
        "current_player": cur,
        "moves_remaining_this_turn": int(state.moves_remaining_this_turn()) if not terminal else 0,
        "is_human_turn": is_human_turn,
        "n_moves": len(rec.move_log),
        "last_moves": last_moves,
        "placement_radius": placement_radius,
        "win_length": win_length,
        "terminal": terminal,
        "winner": winner,
        "result_type": result_type,
        "difficulty": rec.difficulty,
        "htttx": serialize_htttx(rec.move_log) if (terminal and include_htttx) else None,
    }
