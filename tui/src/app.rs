//! App state + event loop. Two tabs at v0.1: bank-list (manage) and event
//! stream (observability). Future commits add use/search axes + drill-downs.

use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout},
    widgets::Paragraph,
    Terminal,
};
use sqlx::PgPool;

use crate::{
    db::{self, BankSummary},
    theme,
    views::{common, manage, observability},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Banks,
    Events,
}

impl Tab {
    fn title(&self) -> &'static str {
        match self {
            Tab::Banks => "banks",
            Tab::Events => "events",
        }
    }
}

pub struct App {
    pool: PgPool,
    redacted_url: String,
    tab: Tab,

    // Banks view state
    banks: Vec<BankSummary>,
    banks_error: Option<String>,
    manage_state: manage::BankListState,

    // Events view state
    events: Vec<db::Event>,
    events_error: Option<String>,
    events_state: observability::EventStreamState,
    last_events_refresh: Instant,

    help_open: bool,
    status: String,
    should_quit: bool,
}

const EVENTS_POLL_INTERVAL: Duration = Duration::from_millis(750);
const EVENTS_PER_KIND: i64 = 50;
const EVENTS_TOTAL_CAP: usize = 200;

impl App {
    pub async fn new(pool: PgPool, redacted_url: String) -> Self {
        let mut app = Self {
            pool,
            redacted_url,
            tab: Tab::Banks,
            banks: Vec::new(),
            banks_error: None,
            manage_state: manage::BankListState::new(),
            events: Vec::new(),
            events_error: None,
            events_state: observability::EventStreamState::new(),
            last_events_refresh: Instant::now()
                .checked_sub(EVENTS_POLL_INTERVAL * 2)
                .unwrap_or_else(Instant::now),
            help_open: false,
            status: "Tab to switch · ? for help · q to quit".to_string(),
            should_quit: false,
        };
        app.reload_banks().await;
        app
    }

    async fn reload_banks(&mut self) {
        match db::banks::list_with_counts(&self.pool).await {
            Ok(banks) => {
                self.banks = banks;
                self.banks_error = None;
                if self.banks.is_empty() {
                    self.manage_state.table.select(None);
                } else {
                    let sel = self.manage_state.table.selected().unwrap_or(0);
                    self.manage_state
                        .table
                        .select(Some(sel.min(self.banks.len() - 1)));
                }
                self.status = format!("banks loaded: {}", self.banks.len());
            }
            Err(e) => {
                self.banks_error = Some(format!("{e}"));
                self.status = "load failed — press r to retry".to_string();
            }
        }
    }

    async fn reload_events(&mut self) {
        match db::events::recent(&self.pool, None, EVENTS_PER_KIND, EVENTS_TOTAL_CAP).await {
            Ok(events) => {
                self.events = events;
                self.events_error = None;
                self.events_state.maintain_auto_scroll(self.events.len());
            }
            Err(e) => {
                self.events_error = Some(format!("{e}"));
            }
        }
        self.last_events_refresh = Instant::now();
    }

    pub async fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<()> {
        while !self.should_quit {
            terminal
                .draw(|f| self.draw(f))
                .map_err(|e| color_eyre::eyre::eyre!("draw frame: {e}"))?;

            // Background refresh tick for the events view — only when that tab
            // is visible, so the bank list isn't paying for polls it doesn't use.
            if self.tab == Tab::Events && self.last_events_refresh.elapsed() >= EVENTS_POLL_INTERVAL
            {
                self.reload_events().await;
            }

            // 100ms event poll keeps keystrokes responsive AND lets the
            // events-tab refresh tick fire on its own schedule above.
            if event::poll(Duration::from_millis(100)).wrap_err("event poll")? {
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
        if self.help_open {
            self.help_open = false;
            return;
        }

        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) => self.should_quit = true,
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => self.should_quit = true,
            (KeyCode::Esc, _) => self.should_quit = true,
            (KeyCode::Char('?'), _) => self.help_open = true,

            (KeyCode::Tab, _) | (KeyCode::Char('2'), _) if self.tab == Tab::Banks => {
                self.tab = Tab::Events;
                if self.events.is_empty() && self.events_error.is_none() {
                    self.reload_events().await;
                }
            }
            (KeyCode::Tab, _) | (KeyCode::Char('1'), _) if self.tab == Tab::Events => {
                self.tab = Tab::Banks;
            }
            (KeyCode::Char('1'), _) => self.tab = Tab::Banks,
            (KeyCode::Char('2'), _) => self.tab = Tab::Events,
            (KeyCode::BackTab, _) => {
                self.tab = match self.tab {
                    Tab::Banks => Tab::Events,
                    Tab::Events => Tab::Banks,
                };
            }

            (KeyCode::Char('r'), _) => {
                self.status = "reloading…".into();
                match self.tab {
                    Tab::Banks => self.reload_banks().await,
                    Tab::Events => self.reload_events().await,
                }
            }

            // Navigation — dispatches per active tab.
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => match self.tab {
                Tab::Banks => self.manage_state.select_next(self.banks.len()),
                Tab::Events => self.events_state.select_next(self.events.len()),
            },
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => match self.tab {
                Tab::Banks => self.manage_state.select_prev(self.banks.len()),
                Tab::Events => self.events_state.select_prev(self.events.len()),
            },
            (KeyCode::Home, _) | (KeyCode::Char('g'), _) => match self.tab {
                Tab::Banks => {
                    if !self.banks.is_empty() {
                        self.manage_state.table.select(Some(0));
                    }
                }
                Tab::Events => self.events_state.jump_top(self.events.len()),
            },
            (KeyCode::End, _) | (KeyCode::Char('G'), _) => match self.tab {
                Tab::Banks => {
                    if !self.banks.is_empty() {
                        self.manage_state.table.select(Some(self.banks.len() - 1));
                    }
                }
                Tab::Events => {
                    let len = self.events.len();
                    if len > 0 {
                        self.events_state.table.select(Some(len - 1));
                        self.events_state.auto_scroll = false;
                    }
                }
            },
            (KeyCode::Char('f'), _) if self.tab == Tab::Events => {
                self.events_state.auto_scroll = !self.events_state.auto_scroll;
                self.status = format!(
                    "auto-scroll {}",
                    if self.events_state.auto_scroll {
                        "on"
                    } else {
                        "off"
                    }
                );
            }

            _ => {}
        }
    }

    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        let [header_area, body_area, footer_area] = common::chrome_layout(area);

        // Header now also shows the active tab indicator.
        let active = self.tab.title();
        let title = format!("{}  ·  ⇥ tab switches", active);
        common::render_header(frame, header_area, &title, &self.redacted_url);

        let chunks = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(body_area);

        match self.tab {
            Tab::Banks => manage::render(
                frame,
                chunks[0],
                &self.banks,
                &mut self.manage_state,
                self.banks_error.as_deref(),
            ),
            Tab::Events => observability::render(
                frame,
                chunks[0],
                &self.events,
                &mut self.events_state,
                None,
                self.events_error.as_deref(),
            ),
        }

        let status_line = ratatui::text::Line::from(vec![ratatui::text::Span::styled(
            format!(" {}", self.status),
            theme::respect_no_color(theme::dim()),
        )]);
        frame.render_widget(Paragraph::new(status_line), chunks[1]);

        let hints: &[(&str, &str)] = match self.tab {
            Tab::Banks => &[
                ("Tab/2", "events"),
                ("↑↓/j/k", "select"),
                ("r", "refresh"),
                ("?", "help"),
                ("q", "quit"),
            ],
            Tab::Events => &[
                ("Tab/1", "banks"),
                ("↑↓/j/k", "select"),
                ("f", "auto-scroll"),
                ("r", "refresh"),
                ("?", "help"),
                ("q", "quit"),
            ],
        };
        common::render_footer(frame, footer_area, hints);

        if self.help_open {
            common::render_help_overlay(
                frame,
                area,
                &[
                    ("Tab / ⇧Tab", "switch between banks and events"),
                    ("1 / 2", "jump directly to banks / events"),
                    ("↑ / k", "previous row"),
                    ("↓ / j", "next row"),
                    ("g / Home", "first row (re-enables auto-scroll on events)"),
                    ("G / End", "last row"),
                    ("f", "toggle auto-scroll (events tab)"),
                    ("r", "reload current view from substrate"),
                    ("?", "toggle this help"),
                    ("q / Esc / Ctrl-C", "quit"),
                ],
            );
        }
    }
}
