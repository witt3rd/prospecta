//! App state + event loop. Three tabs at v0.1: bank-list (manage), document
//! drill-down (docs), event stream (observability). Future commits add use
//! and search axes.

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
    views::{common, docs, manage, observability},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Banks,
    Docs,
    Events,
}

impl Tab {
    fn title(&self) -> &'static str {
        match self {
            Tab::Banks => "banks",
            Tab::Docs => "docs",
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

    // Docs view state
    docs_state: docs::DocsState,

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
const DOCS_PAGE_LIMIT: i64 = 200;

impl App {
    pub async fn new(pool: PgPool, redacted_url: String) -> Self {
        let mut app = Self {
            pool,
            redacted_url,
            tab: Tab::Banks,
            banks: Vec::new(),
            banks_error: None,
            manage_state: manage::BankListState::new(),
            docs_state: docs::DocsState::new(),
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

    async fn enter_docs_for_current_bank(&mut self) {
        let bank_id = match self
            .manage_state
            .table
            .selected()
            .and_then(|i| self.banks.get(i))
        {
            Some(b) => b.bank.bank_id.clone(),
            None => {
                self.status = "no bank selected".into();
                return;
            }
        };
        self.tab = Tab::Docs;
        self.docs_state.level = docs::DocsLevel::Documents;
        self.docs_state.bank_id = Some(bank_id.clone());
        self.docs_state.focused_document = None;
        self.docs_state.items.clear();
        self.docs_state.documents_table.select(Some(0));
        self.reload_documents_for_current_bank().await;
        self.status = format!("docs · {}", bank_id);
    }

    async fn reload_documents_for_current_bank(&mut self) {
        let Some(bank_id) = self.docs_state.bank_id.clone() else {
            return;
        };
        match db::documents::list_for_bank(&self.pool, &bank_id, DOCS_PAGE_LIMIT, 0).await {
            Ok(docs_) => {
                self.docs_state.documents = docs_;
                self.docs_state.error = None;
                if self.docs_state.documents.is_empty() {
                    self.docs_state.documents_table.select(None);
                } else {
                    let sel = self.docs_state.documents_table.selected().unwrap_or(0);
                    self.docs_state
                        .documents_table
                        .select(Some(sel.min(self.docs_state.documents.len() - 1)));
                }
            }
            Err(e) => self.docs_state.error = Some(format!("{e}")),
        }
    }

    async fn enter_items_for_selected_document(&mut self) {
        let Some(doc) = self.docs_state.selected_document().cloned() else {
            self.status = "no document selected".into();
            return;
        };
        self.docs_state.level = docs::DocsLevel::Items;
        self.docs_state.focused_document = Some(doc.id);
        self.docs_state.items_table.select(Some(0));
        match db::documents::list_for_document(&self.pool, doc.id).await {
            Ok(items) => {
                self.docs_state.items = items;
                self.docs_state.error = None;
            }
            Err(e) => self.docs_state.error = Some(format!("{e}")),
        }
        self.status = format!("items · doc {}", docs::short_uuid(&doc.id));
    }

    fn ascend_docs(&mut self) {
        match self.docs_state.level {
            docs::DocsLevel::Items => {
                self.docs_state.level = docs::DocsLevel::Documents;
                self.docs_state.items.clear();
                self.docs_state.focused_document = None;
                self.status = format!(
                    "docs · {}",
                    self.docs_state.bank_id.clone().unwrap_or_default()
                );
            }
            docs::DocsLevel::Documents => {
                // Back up to banks tab.
                self.tab = Tab::Banks;
                self.status = "banks".into();
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

            if self.tab == Tab::Events && self.last_events_refresh.elapsed() >= EVENTS_POLL_INTERVAL
            {
                self.reload_events().await;
            }

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
            (KeyCode::Char('?'), _) => self.help_open = true,

            // Direct tab jumps.
            (KeyCode::Char('1'), _) => self.tab = Tab::Banks,
            (KeyCode::Char('2'), _) => {
                if self.docs_state.bank_id.is_some() {
                    self.tab = Tab::Docs;
                } else {
                    self.enter_docs_for_current_bank().await;
                }
            }
            (KeyCode::Char('3'), _) => {
                self.tab = Tab::Events;
                if self.events.is_empty() && self.events_error.is_none() {
                    self.reload_events().await;
                }
            }
            (KeyCode::Tab, _) => {
                self.tab = match self.tab {
                    Tab::Banks => Tab::Docs,
                    Tab::Docs => Tab::Events,
                    Tab::Events => Tab::Banks,
                };
                if self.tab == Tab::Docs && self.docs_state.bank_id.is_none() {
                    // Auto-bind the currently selected bank if user tabs over with no context yet.
                    self.enter_docs_for_current_bank().await;
                }
                if self.tab == Tab::Events && self.events.is_empty() && self.events_error.is_none()
                {
                    self.reload_events().await;
                }
            }
            (KeyCode::BackTab, _) => {
                self.tab = match self.tab {
                    Tab::Banks => Tab::Events,
                    Tab::Docs => Tab::Banks,
                    Tab::Events => Tab::Docs,
                };
            }

            (KeyCode::Char('r'), _) => {
                self.status = "reloading…".into();
                match self.tab {
                    Tab::Banks => self.reload_banks().await,
                    Tab::Docs => match self.docs_state.level {
                        docs::DocsLevel::Documents => {
                            self.reload_documents_for_current_bank().await
                        }
                        docs::DocsLevel::Items => {
                            if let Some(id) = self.docs_state.focused_document {
                                if let Ok(items) =
                                    db::documents::list_for_document(&self.pool, id).await
                                {
                                    self.docs_state.items = items;
                                }
                            }
                        }
                    },
                    Tab::Events => self.reload_events().await,
                }
            }

            // Drill-down: Enter / → descends; Esc / ← ascends.
            (KeyCode::Enter, _) | (KeyCode::Right, _) | (KeyCode::Char('l'), _) => match self.tab {
                Tab::Banks => self.enter_docs_for_current_bank().await,
                Tab::Docs => {
                    if self.docs_state.level == docs::DocsLevel::Documents {
                        self.enter_items_for_selected_document().await;
                    }
                }
                Tab::Events => {}
            },
            (KeyCode::Esc, _) | (KeyCode::Left, _) | (KeyCode::Char('h'), _) => match self.tab {
                Tab::Docs => self.ascend_docs(),
                _ => {
                    // Esc with nothing to ascend = quit, matches v0.1 spec from earlier commit.
                    if key.code == KeyCode::Esc {
                        self.should_quit = true;
                    }
                }
            },

            // Navigation — dispatches per active tab.
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => match self.tab {
                Tab::Banks => self.manage_state.select_next(self.banks.len()),
                Tab::Docs => self.docs_state.select_next(),
                Tab::Events => self.events_state.select_next(self.events.len()),
            },
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => match self.tab {
                Tab::Banks => self.manage_state.select_prev(self.banks.len()),
                Tab::Docs => self.docs_state.select_prev(),
                Tab::Events => self.events_state.select_prev(self.events.len()),
            },
            (KeyCode::Home, _) | (KeyCode::Char('g'), _) => match self.tab {
                Tab::Banks => {
                    if !self.banks.is_empty() {
                        self.manage_state.table.select(Some(0));
                    }
                }
                Tab::Docs => match self.docs_state.level {
                    docs::DocsLevel::Documents => {
                        if !self.docs_state.documents.is_empty() {
                            self.docs_state.documents_table.select(Some(0));
                        }
                    }
                    docs::DocsLevel::Items => {
                        if !self.docs_state.items.is_empty() {
                            self.docs_state.items_table.select(Some(0));
                        }
                    }
                },
                Tab::Events => self.events_state.jump_top(self.events.len()),
            },
            (KeyCode::End, _) | (KeyCode::Char('G'), _) => match self.tab {
                Tab::Banks => {
                    if !self.banks.is_empty() {
                        self.manage_state.table.select(Some(self.banks.len() - 1));
                    }
                }
                Tab::Docs => match self.docs_state.level {
                    docs::DocsLevel::Documents => {
                        let len = self.docs_state.documents.len();
                        if len > 0 {
                            self.docs_state.documents_table.select(Some(len - 1));
                        }
                    }
                    docs::DocsLevel::Items => {
                        let len = self.docs_state.items.len();
                        if len > 0 {
                            self.docs_state.items_table.select(Some(len - 1));
                        }
                    }
                },
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

        let active = self.tab.title();
        let title = format!("{}  ·  1/2/3 or Tab", active);
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
            Tab::Docs => docs::render(frame, chunks[0], &mut self.docs_state),
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
                ("Enter/→", "open docs"),
                ("Tab", "next tab"),
                ("↑↓/j/k", "select"),
                ("r", "refresh"),
                ("?", "help"),
                ("q", "quit"),
            ],
            Tab::Docs => &[
                ("Enter/→", "descend"),
                ("Esc/←", "ascend"),
                ("↑↓/j/k", "select"),
                ("r", "refresh"),
                ("?", "help"),
                ("q", "quit"),
            ],
            Tab::Events => &[
                ("Tab", "next tab"),
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
                    ("Tab / ⇧Tab", "cycle banks → docs → events"),
                    ("1 / 2 / 3", "jump to banks / docs / events"),
                    ("Enter / → / l", "drill down (bank → docs → items)"),
                    ("Esc / ← / h", "ascend one level (items → docs → banks)"),
                    ("↑ / k", "previous row"),
                    ("↓ / j", "next row"),
                    ("g / Home", "first row"),
                    ("G / End", "last row"),
                    ("f", "toggle auto-scroll (events tab)"),
                    ("r", "reload current view from substrate"),
                    ("?", "toggle this help"),
                    ("q / Ctrl-C", "quit"),
                ],
            );
        }
    }
}
