//! App state + event loop. v0.1 holds the bank-list view + help overlay; later
//! commits add tabbed views (manage / use / search / observability).

use std::time::Duration;

use color_eyre::eyre::{Result, WrapErr};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{layout::Constraint, layout::Layout, widgets::Paragraph, Terminal};
use sqlx::PgPool;

use crate::{
    db::{self, BankSummary},
    theme,
    views::{common, manage},
};

pub struct App {
    pool: PgPool,
    redacted_url: String,
    banks: Vec<BankSummary>,
    load_error: Option<String>,
    manage_state: manage::BankListState,
    help_open: bool,
    status: String,
    should_quit: bool,
}

impl App {
    pub async fn new(pool: PgPool, redacted_url: String) -> Self {
        let mut app = Self {
            pool,
            redacted_url,
            banks: Vec::new(),
            load_error: None,
            manage_state: manage::BankListState::new(),
            help_open: false,
            status: "press ? for help · r to refresh · q to quit".to_string(),
            should_quit: false,
        };
        app.reload_banks().await;
        app
    }

    async fn reload_banks(&mut self) {
        match db::banks::list_with_counts(&self.pool).await {
            Ok(banks) => {
                self.banks = banks;
                self.load_error = None;
                if let Some(i) = self.manage_state.table.selected() {
                    if self.banks.is_empty() {
                        self.manage_state.table.select(None);
                    } else if i >= self.banks.len() {
                        self.manage_state.table.select(Some(self.banks.len() - 1));
                    }
                } else if !self.banks.is_empty() {
                    self.manage_state.table.select(Some(0));
                }
                self.status = format!("banks loaded: {}", self.banks.len());
            }
            Err(e) => {
                self.load_error = Some(format!("{e}"));
                self.status = "load failed — press r to retry".to_string();
            }
        }
    }

    pub async fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<()> {
        while !self.should_quit {
            terminal
                .draw(|f| self.draw(f))
                .map_err(|e| color_eyre::eyre::eyre!("draw frame: {e}"))?;

            // Poll with a short timeout — keeps the loop responsive and gives
            // us a free hook for periodic refreshes once we add the event-stream view.
            if event::poll(Duration::from_millis(250)).wrap_err("event poll")? {
                if let Event::Key(key) = event::read().wrap_err("event read")? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key).await;
                    }
                }
            }
        }
        Ok(())
    }

    async fn handle_key(&mut self, key: KeyEvent) {
        // Help overlay swallows any key (close on next press).
        if self.help_open {
            self.help_open = false;
            return;
        }

        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) => self.should_quit = true,
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => self.should_quit = true,
            (KeyCode::Esc, _) => self.should_quit = true,
            (KeyCode::Char('?'), _) => self.help_open = true,
            (KeyCode::Char('r'), _) => {
                self.status = "reloading…".to_string();
                self.reload_banks().await;
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
                self.manage_state.select_next(self.banks.len());
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
                self.manage_state.select_prev(self.banks.len());
            }
            (KeyCode::Home, _) | (KeyCode::Char('g'), _) => {
                if !self.banks.is_empty() {
                    self.manage_state.table.select(Some(0));
                }
            }
            (KeyCode::End, _) | (KeyCode::Char('G'), _) => {
                if !self.banks.is_empty() {
                    self.manage_state.table.select(Some(self.banks.len() - 1));
                }
            }
            _ => {}
        }
    }

    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        let [header_area, body_area, footer_area] = common::chrome_layout(area);

        common::render_header(frame, header_area, "banks", &self.redacted_url);

        // Split body into table + 1-line status strip.
        let chunks = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(body_area);

        manage::render(
            frame,
            chunks[0],
            &self.banks,
            &mut self.manage_state,
            self.load_error.as_deref(),
        );

        // Status line.
        let status_line = ratatui::text::Line::from(vec![ratatui::text::Span::styled(
            format!(" {}", self.status),
            theme::respect_no_color(theme::dim()),
        )]);
        frame.render_widget(Paragraph::new(status_line), chunks[1]);

        common::render_footer(
            frame,
            footer_area,
            &[
                ("↑↓/j/k", "select"),
                ("r", "refresh"),
                ("?", "help"),
                ("q", "quit"),
            ],
        );

        if self.help_open {
            common::render_help_overlay(
                frame,
                area,
                &[
                    ("↑ / k", "previous bank"),
                    ("↓ / j", "next bank"),
                    ("g / Home", "first bank"),
                    ("G / End", "last bank"),
                    ("r", "reload banks from substrate"),
                    ("?", "toggle this help"),
                    ("q / Esc / Ctrl-C", "quit"),
                ],
            );
        }
    }
}
