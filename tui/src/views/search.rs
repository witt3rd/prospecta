//! Search view — free-text query over both lexical channels.
//!
//! Two zones: a query input box (top) and a results table (below). The view
//! carries an input mode so the app loop knows whether keystrokes edit the
//! query or navigate results.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
    Frame,
};

use crate::{db::SearchHit, theme};

/// Whether keystrokes edit the query or navigate the result list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Typing into the query box. Enter runs the search.
    Editing,
    /// Navigating results. `/` or `i` returns to editing.
    Browsing,
}

pub struct SearchState {
    /// Bank the search is scoped to. None until a bank is bound.
    pub bank_id: Option<String>,
    pub query: String,
    pub mode: SearchMode,
    pub hits: Vec<SearchHit>,
    pub table: TableState,
    pub error: Option<String>,
    /// True once a search has actually run (distinguishes "no results" from
    /// "haven't searched yet").
    pub searched: bool,
}

impl SearchState {
    pub fn new() -> Self {
        Self {
            bank_id: None,
            query: String::new(),
            mode: SearchMode::Editing,
            hits: Vec::new(),
            table: TableState::default(),
            error: None,
            searched: false,
        }
    }

    pub fn push_char(&mut self, c: char) {
        self.query.push(c);
    }

    pub fn backspace(&mut self) {
        self.query.pop();
    }

    pub fn select_next(&mut self) {
        if self.hits.is_empty() {
            return;
        }
        let i = match self.table.selected() {
            Some(i) if i + 1 < self.hits.len() => i + 1,
            Some(i) => i,
            None => 0,
        };
        self.table.select(Some(i));
    }

    pub fn select_prev(&mut self) {
        if self.hits.is_empty() {
            return;
        }
        let i = match self.table.selected() {
            Some(i) if i > 0 => i - 1,
            _ => 0,
        };
        self.table.select(Some(i));
    }

    pub fn selected_hit(&self) -> Option<&SearchHit> {
        self.table.selected().and_then(|i| self.hits.get(i))
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &mut SearchState) {
    // input box (3 rows) · results table (rest) · preview pane (when a row is
    // selected, the bottom shows the matched index_text + body chunk in full).
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
            Constraint::Length(8),
        ])
        .split(area);

    render_input(frame, chunks[0], state);
    render_results(frame, chunks[1], state);
    render_preview(frame, chunks[2], state);
}

fn render_input(frame: &mut Frame, area: Rect, state: &SearchState) {
    let editing = state.mode == SearchMode::Editing;
    let border_style = if editing {
        theme::respect_no_color(theme::accent())
    } else {
        theme::respect_no_color(theme::dim())
    };

    let bank = state.bank_id.as_deref().unwrap_or("(no bank)");
    let title = format!(" search · {bank} ");

    // Cursor block when editing.
    let mut spans = vec![Span::styled(
        state.query.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    if editing {
        spans.push(Span::styled("▏", theme::respect_no_color(theme::accent())));
    }

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(border_style);
    frame.render_widget(Paragraph::new(Line::from(spans)).block(block), area);
}

fn render_results(frame: &mut Frame, area: Rect, state: &mut SearchState) {
    if let Some(err) = state.error.as_deref() {
        let block = Block::default()
            .title(" results · error ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::error()));
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                err.to_string(),
                theme::respect_no_color(theme::error()),
            )))
            .block(block)
            .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }

    if state.hits.is_empty() {
        let body = if !state.searched {
            "type a query and press Enter — searches the question + body channels"
        } else {
            "no matches in either lexical channel for that query"
        };
        let block = Block::default()
            .title(" results ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::dim()));
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                body,
                theme::respect_no_color(theme::dim()),
            )))
            .block(block)
            .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }

    let rows: Vec<Row> = state
        .hits
        .iter()
        .map(|h| {
            let channel_style = match (h.content_hit, h.body_hit) {
                (true, true) => Style::default().fg(Color::LightGreen),
                (true, false) => theme::accent(),
                (false, true) => Style::default().fg(Color::LightBlue),
                (false, false) => theme::dim(),
            };
            let gen_marker = if h.llm_generated { "llm" } else { "given" };
            Row::new(vec![
                Line::from(Span::styled(
                    h.channel_label().to_string(),
                    theme::respect_no_color(channel_style),
                )),
                Line::from(format!("{:.3}", h.best_rank())),
                Line::from(gen_marker.to_string()),
                Line::from(truncate(&h.content, 60)),
                Line::from(truncate(h.source.as_deref().unwrap_or("—"), 24)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(9),
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Min(30),
        Constraint::Length(26),
    ];

    let header = Row::new(vec![
        "channel",
        "rank",
        "gen",
        "matched index_text",
        "source",
    ])
    .style(theme::respect_no_color(theme::table_header()));

    let title = format!(" results · {} hits ", state.hits.len());
    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(theme::respect_no_color(theme::selected_row()))
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(theme::respect_no_color(theme::dim())),
        );

    frame.render_stateful_widget(table, area, &mut state.table);
}

fn render_preview(frame: &mut Frame, area: Rect, state: &SearchState) {
    let block = Block::default()
        .title(" preview · matched item ")
        .borders(Borders::ALL)
        .border_style(theme::respect_no_color(theme::dim()));

    let Some(h) = state.selected_hit() else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  select a result to preview the index_text + body chunk",
                theme::respect_no_color(theme::dim()),
            )))
            .block(block),
            area,
        );
        return;
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("index_text  ", theme::respect_no_color(theme::dim())),
            Span::styled(
                h.content.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("body        ", theme::respect_no_color(theme::dim())),
            Span::raw(h.original_chunk.clone()),
        ]),
        Line::from(vec![
            Span::styled("ranks       ", theme::respect_no_color(theme::dim())),
            Span::raw(format!(
                "question={:.4}  body={:.4}",
                h.content_rank, h.body_rank
            )),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn truncate(s: &str, max: usize) -> String {
    // Collapse newlines so a multi-line chunk stays one table row.
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if flat.chars().count() <= max {
        flat
    } else {
        let head: String = flat.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}
