"""Pygame hex viewer — watch the AI play HeXO."""

import math
import sys
import time

import pygame

# Hex layout constants
HEX_SIZE = 24  # radius of each hexagon in pixels
SQRT3 = math.sqrt(3)

# Colors
BG = (30, 30, 36)
GRID_LINE = (55, 55, 65)
EMPTY_HEX = (45, 45, 55)
P1_COLOR = (80, 200, 220)     # cyan
P2_COLOR = (220, 100, 200)    # magenta
LAST_MOVE = (255, 220, 80)    # yellow highlight
TEXT_COLOR = (200, 200, 210)
DIM_TEXT = (120, 120, 130)
WHITE = (255, 255, 255)


def axial_to_pixel(q: int, r: int, size: float) -> tuple[float, float]:
    """Convert axial hex coords to pixel coords (flat-top hexagons)."""
    x = size * (3 / 2 * q)
    y = size * (SQRT3 / 2 * q + SQRT3 * r)
    return x, y


def draw_hexagon(surface, center, size, color, outline=None, width=0):
    """Draw a regular hexagon (flat-top)."""
    cx, cy = center
    points = []
    for i in range(6):
        angle = math.pi / 180 * (60 * i)
        px = cx + size * math.cos(angle)
        py = cy + size * math.sin(angle)
        points.append((px, py))
    if width == 0:
        pygame.draw.polygon(surface, color, points)
    if outline:
        pygame.draw.polygon(surface, outline, points, 2)


def run_viewer(game_config_dict, model, mcts_config, device, model_config=None):
    """Run the pygame viewer — plays one AI game and displays it."""
    import torch
    import hexo_rs
    from hexo_a0.mcts import evaluate_state, gumbel_mcts

    # Select graph builder based on model type
    graph_fn = None
    if model_config is not None:
        threat = getattr(model_config, "threat_features", False)
        rel = getattr(model_config, "relative_stone_encoding", False)
        if model_config.graph_type == "axis":
            from hexo_a0.graph import game_to_axis_graph
            prune = model_config.prune_empty_edges
            graph_fn = lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
        else:
            from hexo_a0.graph import game_to_graph
            graph_fn = lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)

    game_config = hexo_rs.GameConfig(**game_config_dict)
    greedy_mode = True

    pygame.init()
    screen_w, screen_h = 900, 700
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
    pygame.display.set_caption("HeXO Watch")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 18)
    font_big = pygame.font.Font(None, 26)

    model.eval()
    game = hexo_rs.GameState(game_config)
    move_num = 0
    last_move = None
    status = "Thinking..."
    paused = False
    game_over = False
    move_delay = 0.3  # seconds between moves
    last_move_time = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Restart
                    game = hexo_rs.GameState(game_config)
                    move_num = 0
                    last_move = None
                    game_over = False
                    status = "Thinking..."
                elif event.key == pygame.K_g:
                    greedy_mode = not greedy_mode
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    move_delay = max(0.05, move_delay - 0.1)
                elif event.key == pygame.K_MINUS:
                    move_delay = min(5.0, move_delay + 0.1)
            elif event.type == pygame.VIDEORESIZE:
                screen_w, screen_h = event.w, event.h
                screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)

        # AI move
        now = time.time()
        if not game_over and not paused and (now - last_move_time) >= move_delay:
            if not game.is_terminal():
                player = game.current_player()
                remaining = game.moves_remaining_this_turn()

                with torch.no_grad():
                    if greedy_mode:
                        model.eval()
                        logits, value, coords = evaluate_state(model, game, device, graph_fn=graph_fn)
                        action = coords[logits.argmax().item()]
                    else:
                        action, policy = gumbel_mcts(game, model, mcts_config, device, graph_fn=graph_fn)

                game.apply_move(action[0], action[1])
                move_num += 1
                last_move = action
                last_move_time = now

                if game.is_terminal():
                    winner = game.winner()
                    if winner:
                        status = f"{winner} wins in {move_num} moves!"
                    else:
                        status = f"Draw after {move_num} moves"
                    game_over = True
                else:
                    next_p = game.current_player()
                    status = f"Move {move_num}: {player} placed at ({action[0]}, {action[1]})"

        # --- Draw ---
        screen.fill(BG)

        # Center offset
        cx = screen_w / 2
        cy = screen_h / 2 - 20

        stones = game.placed_stones()
        stone_map = {(q, r): p for (q, r), p in stones}

        # Draw legal move hexagons (empty, subtle)
        if not game_over:
            legal = game.legal_moves()
            for q, r in legal:
                px, py = axial_to_pixel(q, r, HEX_SIZE)
                # Only draw near existing stones (within distance 3)
                near = any(
                    max(abs(q - sq), abs(r - sr), abs((q + r) - (sq + sr))) <= 3
                    for (sq, sr) in stone_map
                )
                if near:
                    draw_hexagon(screen, (cx + px, cy + py), HEX_SIZE * 0.85,
                                 EMPTY_HEX, outline=GRID_LINE)

        # Draw placed stones
        for (q, r), player in stones:
            px, py = axial_to_pixel(q, r, HEX_SIZE)
            screen_pos = (cx + px, cy + py)

            if (q, r) == last_move:
                # Last move: yellow ring
                draw_hexagon(screen, screen_pos, HEX_SIZE * 0.9, LAST_MOVE)
                color = P1_COLOR if player == "P1" else P2_COLOR
                draw_hexagon(screen, screen_pos, HEX_SIZE * 0.7, color)
            else:
                color = P1_COLOR if player == "P1" else P2_COLOR
                draw_hexagon(screen, screen_pos, HEX_SIZE * 0.85, color)

            # Label
            label = "X" if player == "P1" else "O"
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=screen_pos)
            screen.blit(text, text_rect)

        # HUD
        hud_y = 12
        status_text = font_big.render(status, True, TEXT_COLOR)
        screen.blit(status_text, (12, hud_y))

        if greedy_mode:
            mode_label = "GREEDY"
        else:
            mode_label = f"MCTS (sims={mcts_config.n_simulations}, m={mcts_config.m_actions})"
        mode_text = font.render(mode_label, True, TEXT_COLOR)
        screen.blit(mode_text, (12, hud_y + 28))

        controls = "SPACE: pause  R: restart  G: greedy/mcts  +/-: speed  Q: quit"
        ctrl_text = font.render(controls, True, DIM_TEXT)
        screen.blit(ctrl_text, (12, screen_h - 28))

        speed_text = font.render(f"delay: {move_delay:.1f}s", True, DIM_TEXT)
        screen.blit(speed_text, (screen_w - 140, screen_h - 28))

        if paused:
            pause_text = font_big.render("PAUSED", True, LAST_MOVE)
            pr = pause_text.get_rect(center=(screen_w / 2, screen_h - 50))
            screen.blit(pause_text, pr)

        # Legend
        legend_y = hud_y + 30
        pygame.draw.circle(screen, P1_COLOR, (20, legend_y + 8), 8)
        screen.blit(font.render("X (P1)", True, P1_COLOR), (34, legend_y))
        pygame.draw.circle(screen, P2_COLOR, (120, legend_y + 8), 8)
        screen.blit(font.render("O (P2)", True, P2_COLOR), (134, legend_y))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
