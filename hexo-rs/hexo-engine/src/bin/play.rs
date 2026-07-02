use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind};
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph};

use hexo_engine::game::{GameConfig, MoveError};
use hexo_engine::{Coord, GameState, Player};

const FILLED: &str = "\u{2b22}"; // ⬢
const EMPTY: &str = "\u{2b21}";  // ⬡

/// Hex char is 2 cols wide. W=4 gives 2 cols gap between hexes.
/// H=2 gives 1 blank row between hex rows. r_offset = W/2 = 2.
const W: i32 = 4;
const H: i32 = 2;
const R_OFF: i32 = W / 2; // stagger offset per row
const STATUS_H: u16 = 5;

#[derive(Clone, Copy)]
enum Theme {
    Dark,
    Light,
}

struct App {
    game: GameState,
    cursor: Option<Coord>,
    message: String,
    config: GameConfig,
    origin_x: i32,
    origin_y: i32,
    theme: Theme,
}

impl App {
    fn new(config: GameConfig) -> Self {
        Self {
            game: GameState::with_config(config),
            cursor: None,
            message: "Click to place. q: quit, r: restart, t: theme".into(),
            config,
            origin_x: 0,
            origin_y: 0,
            theme: Theme::Dark,
        }
    }

    fn restart(&mut self) {
        self.game = GameState::with_config(self.config);
        self.cursor = None;
        self.message = "New game started.".into();
    }

    fn screen_to_axial(&self, col: u16, row: u16) -> Coord {
        // Center of the (0,0) hex.
        let ccx = self.origin_x as f64 + 1.0; // hex char center ~1 col in
        let ccy = self.origin_y as f64;

        let dx = col as f64 - ccx;
        let dy = row as f64 - ccy;

        let rf = dy / H as f64;
        let qf = (dx - rf * R_OFF as f64) / W as f64;

        // Cube-coordinate rounding.
        let cx = qf;
        let cz = rf;
        let cy = -cx - cz;

        let mut rx = cx.round();
        let ry = cy.round();
        let mut rz = cz.round();

        let ex = (rx - cx).abs();
        let ey = (ry - cy).abs();
        let ez = (rz - cz).abs();

        if ex > ey && ex > ez {
            rx = -ry - rz;
        } else if ey <= ez {
            rz = -rx - ry;
        }

        (rx as i32, rz as i32)
    }

    fn try_place(&mut self, coord: Coord) {
        if self.game.is_terminal() {
            self.message = "Game over! Press 'r' to restart.".into();
            return;
        }
        match self.game.apply_move(coord) {
            Ok(()) => {
                let stones = self.game.placed_stones().len();
                if self.game.is_terminal() {
                    match self.game.winner() {
                        Some(p) => {
                            self.message = format!(
                                "{} wins! ({stones} stones). Press 'r'.",
                                sym(p)
                            );
                        }
                        None => {
                            self.message = format!("Draw ({stones} stones). Press 'r'.");
                        }
                    }
                } else {
                    let p = self.game.current_player().unwrap();
                    let rem = self.game.moves_remaining_this_turn();
                    self.message = format!(
                        "{} to move ({rem} left). ({},{}). {stones} stones.",
                        sym(p), coord.0, coord.1,
                    );
                }
            }
            Err(MoveError::CellOccupied) => {
                self.message = format!("({},{}) occupied!", coord.0, coord.1);
            }
            Err(MoveError::OutOfRange) => {
                self.message = format!("({},{}) out of range!", coord.0, coord.1);
            }
            Err(MoveError::GameOver) => {
                self.message = "Game over! Press 'r'.".into();
            }
        }
    }
}

fn sym(p: Player) -> &'static str {
    match p {
        Player::P1 => "X",
        Player::P2 => "O",
    }
}

fn axial_to_screen(q: i32, r: i32, ox: i32, oy: i32) -> (i32, i32) {
    (ox + q * W + r * R_OFF, oy + r * H)
}

fn in_board(sx: i32, sy: i32, area: Rect) -> bool {
    sx >= area.x as i32
        && sx + 2 < (area.x + area.width) as i32 // hex char is ~2 cols
        && sy >= area.y as i32
        && sy < (area.y + area.height) as i32
}

fn min_stone_dist(coord: Coord, stones: &[(Coord, Player)]) -> i32 {
    stones
        .iter()
        .map(|&(c, _)| hexo_engine::hex::hex_distance(coord, c))
        .min()
        .unwrap_or(i32::MAX)
}

fn fade_style(dist: i32, max_dist: i32, theme: Theme) -> Option<Style> {
    if dist > max_dist {
        return None;
    }
    let t = (dist - 1).max(0) as f64 / (max_dist - 1).max(1) as f64;
    let idx = match theme {
        Theme::Dark => (250.0 - t * 12.0) as u8,
        Theme::Light => (236.0 + t * 12.0) as u8,
    };
    Some(Style::default().fg(Color::Indexed(idx)))
}

fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let [board_area, status_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(STATUS_H)]).areas(area);

    let ox = board_area.x as i32 + board_area.width as i32 / 2;
    let oy = board_area.y as i32 + board_area.height as i32 / 2;
    app.origin_x = ox;
    app.origin_y = oy;

    let stones = app.game.placed_stones();
    let legal = app.game.legal_moves_set();
    let radius = app.config.placement_radius;

    // 1. Empty hex outlines with fade.
    if !app.game.is_terminal() {
        for &(q, r) in legal.iter() {
            let (sx, sy) = axial_to_screen(q, r, ox, oy);
            if !in_board(sx, sy, board_area) {
                continue;
            }
            let dist = min_stone_dist((q, r), &stones);
            if let Some(style) = fade_style(dist, radius, app.theme) {
                frame.render_widget(
                    Span::styled(EMPTY, style),
                    Rect::new(sx as u16, sy as u16, 3, 1),
                );
            }
        }
    }

    // 2. Placed stones.
    for &((q, r), player) in &stones {
        let (sx, sy) = axial_to_screen(q, r, ox, oy);
        if !in_board(sx, sy, board_area) {
            continue;
        }
        let color = match player {
            Player::P1 => Color::Cyan,
            Player::P2 => Color::Magenta,
        };
        frame.render_widget(
            Span::styled(FILLED, Style::default().fg(color).bold()),
            Rect::new(sx as u16, sy as u16, 3, 1),
        );
    }

    // 3. Cursor highlight.
    if let Some((q, r)) = app.cursor {
        let (sx, sy) = axial_to_screen(q, r, ox, oy);
        if in_board(sx, sy, board_area) {
            let is_stone = stones.iter().any(|&(c, _)| c == (q, r));
            let is_legal = legal.contains(&(q, r));

            if is_stone {
                // Bright outline around the stone.
                frame.render_widget(
                    Span::styled(FILLED, Style::default().fg(Color::Yellow).bold()),
                    Rect::new(sx as u16, sy as u16, 3, 1),
                );
            } else if is_legal {
                frame.render_widget(
                    Span::styled(FILLED, Style::default().fg(Color::Yellow)),
                    Rect::new(sx as u16, sy as u16, 3, 1),
                );
            } else {
                frame.render_widget(
                    Span::styled(EMPTY, Style::default().fg(Color::DarkGray)),
                    Rect::new(sx as u16, sy as u16, 3, 1),
                );
            }
        }
    }

    // Status bar.
    let player_info = if let Some(p) = app.game.current_player() {
        let rem = app.game.moves_remaining_this_turn();
        let marker = match p {
            Player::P1 => format!("{FILLED} X"),
            Player::P2 => format!("{FILLED} O"),
        };
        format!(" {marker} to move \u{2502} {rem} left \u{2502} {} stones ", app.game.placed_stones().len())
    } else {
        match app.game.winner() {
            Some(p) => format!(" {} wins! ", sym(p)),
            None => " Draw! ".into(),
        }
    };

    let status = Paragraph::new(vec![
        Line::from(app.message.as_str()),
        Line::from(player_info),
        Line::from(" q: quit \u{2502} r: restart \u{2502} t: theme \u{2502} click: place "),
    ])
    .block(Block::bordered().title(" HeXO "));
    frame.render_widget(status, status_area);
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = if args.len() > 1 {
        let win_len: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
        let radius: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let max_moves: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
        GameConfig { win_length: win_len, placement_radius: radius, max_moves }
    } else {
        GameConfig::FULL_HEXO
    };

    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = ratatui::init();
    let mut app = App::new(config);

    loop {
        terminal.draw(|frame| render(frame, &mut app))?;
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('r') => app.restart(),
                    KeyCode::Char('t') => {
                        app.theme = match app.theme {
                            Theme::Dark => Theme::Light,
                            Theme::Light => Theme::Dark,
                        };
                    }
                    _ => {}
                },
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::Moved | MouseEventKind::Drag(MouseButton::Left) => {
                        app.cursor = Some(app.screen_to_axial(mouse.column, mouse.row));
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        let coord = app.screen_to_axial(mouse.column, mouse.row);
                        app.try_place(coord);
                        app.cursor = Some(coord);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    ratatui::restore();
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    disable_raw_mode()?;
    Ok(())
}
