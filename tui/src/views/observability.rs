//! Event-stream view — axis 4 "review observability" entry point.
//!
//! Tail-style view of the five append-only event tables, interleaved by
//! created_at DESC, color-coded by event kind. Polled (not LISTEN/NOTIFY)
//! for v0.1.

use chrono::{DateTime, Utc};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState},
    Frame,
};

use crate::{
    db::{Event, EventKind},
    theme,
};

pub struct EventStreamState {
    pub table: TableState,
    pub auto_scroll: bool,
}

impl EventStreamState {
    pub fn new() -> Self {
        let mut t = TableState::default();
        t.select(Some(0));
        Self {
            table: t,
            auto_scroll: true,
        }
    }

    pub fn select_next(&mut self, len: usize) {
        if len == 0 {
            self.table.select(None);
            return;
        }
        // Manual navigation disables auto-scroll until user hits Home / 'g'.
        self.auto_scroll = false;
        let i = self.table.selected().map(|i| (i + 1) % len).unwrap_or(0);
        self.table.select(Some(i));
    }

    pub fn select_prev(&mut self, len: usize) {
        if len == 0 {
            self.table.select(None);
            return;
        }
        self.auto_scroll = false;
        let i = self
            .table
            .selected()
            .map(|i| if i == 0 { len - 1 } else { i - 1 })
            .unwrap_or(0);
        self.table.select(Some(i));
    }

    pub fn jump_top(&mut self, len: usize) {
        if len > 0 {
            self.table.select(Some(0));
            self.auto_scroll = true;
        }
    }

    /// Called on each refresh. Pins selection to row 0 when auto-scroll is on.
    pub fn maintain_auto_scroll(&mut self, len: usize) {
        if self.auto_scroll && len > 0 {
            self.table.select(Some(0));
        }
    }
}

fn kind_style(kind: EventKind) -> Style {
    if std::env::var_os("NO_COLOR").is_some() {
        return Style::default();
    }
    match kind {
        EventKind::Retain => Style::default().fg(Color::LightGreen),
        EventKind::Recall => Style::default().fg(Color::LightBlue),
        EventKind::Formulate => Style::default().fg(Color::LightCyan),
        EventKind::LlmCall => Style::default().fg(Color::LightYellow),
        EventKind::SweepPass => Style::default().fg(Color::Magenta),
    }
}

fn format_when(ts: DateTime<Utc>) -> String {
    // HH:MM:SS in local time is friendlier than full RFC3339 in a tail view.
    let local: chrono::DateTime<chrono::Local> = ts.into();
    local.format("%H:%M:%S").to_string()
}

fn format_duration(ms: Option<i32>) -> String {
    match ms {
        Some(n) if n >= 1000 => format!("{:.1}s", n as f64 / 1000.0),
        Some(n) => format!("{}ms", n),
        None => "—".to_string(),
    }
}

pub fn render(
    frame: &mut Frame,
    area: Rect,
    events: &[Event],
    state: &mut EventStreamState,
    bank_filter: Option<&str>,
    load_error: Option<&str>,
) {
    if let Some(err) = load_error {
        let block = Block::default()
            .title(" events · load error ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::error()));
        let body = Line::from(Span::styled(
            err.to_string(),
            theme::respect_no_color(theme::error()),
        ));
        frame.render_widget(Paragraph::new(body).block(block), area);
        return;
    }

    if events.is_empty() {
        let block = Block::default()
            .title(" events ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::dim()));
        let body = Line::from(Span::styled(
            "no events yet — retain or recall something to see the stream populate",
            theme::respect_no_color(theme::dim()),
        ));
        frame.render_widget(Paragraph::new(body).block(block), area);
        return;
    }

    let rows: Vec<Row> = events
        .iter()
        .map(|e| {
            let kind_cell = Span::styled(
                format!("{:<9}", e.kind.tag()),
                kind_style(e.kind).add_modifier(Modifier::BOLD),
            );
            let err_marker = if e.has_error {
                Span::styled("!", theme::respect_no_color(theme::error()))
            } else {
                Span::raw(" ")
            };
            Row::new(vec![
                Line::from(format_when(e.created_at)),
                Line::from(kind_cell),
                Line::from(e.bank_id.clone().unwrap_or_else(|| "—".into())),
                Line::from(e.id.to_string()),
                Line::from(format_duration(e.duration_ms)),
                Line::from(vec![
                    err_marker,
                    Span::raw(" "),
                    Span::raw(e.summary.clone()),
                ]),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(10),
        Constraint::Length(11),
        Constraint::Length(14),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Min(20),
    ];

    let header = Row::new(vec!["time", "kind", "bank", "id", "dur", "summary"])
        .style(theme::respect_no_color(theme::table_header()));

    let title = match (bank_filter, state.auto_scroll) {
        (Some(b), true) => format!(" events · {} · {} ▼ ", b, events.len()),
        (Some(b), false) => format!(" events · {} · {} ", b, events.len()),
        (None, true) => format!(" events · all banks · {} ▼ ", events.len()),
        (None, false) => format!(" events · all banks · {} ", events.len()),
    };

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(theme::respect_no_color(theme::selected_row()))
        .highlight_symbol("▶ ")
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(theme::respect_no_color(theme::dim())),
        );

    frame.render_stateful_widget(table, area, &mut state.table);
}
