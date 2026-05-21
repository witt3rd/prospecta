//! Document browser — axis 1 "manage" drill-down.
//!
//! Two-pane stack: documents in a bank → memory_items in a document.
//! State machine carries which level is active; render() dispatches.

use ratatui::{
    layout::{Constraint, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
    Frame,
};
use uuid::Uuid;

use crate::{
    db::{Document, MemoryItem},
    theme,
};

/// Which depth of the drill-down is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocsLevel {
    /// Documents in the current bank.
    Documents,
    /// Memory items for the selected document.
    Items,
}

pub struct DocsState {
    pub level: DocsLevel,
    pub bank_id: Option<String>,
    pub documents: Vec<Document>,
    pub documents_table: TableState,
    /// Document whose items are loaded into `items`.
    pub focused_document: Option<Uuid>,
    pub items: Vec<MemoryItem>,
    pub items_table: TableState,
    pub error: Option<String>,
}

impl DocsState {
    pub fn new() -> Self {
        let mut docs_table = TableState::default();
        docs_table.select(Some(0));
        let mut items_table = TableState::default();
        items_table.select(Some(0));
        Self {
            level: DocsLevel::Documents,
            bank_id: None,
            documents: Vec::new(),
            documents_table: docs_table,
            focused_document: None,
            items: Vec::new(),
            items_table,
            error: None,
        }
    }

    pub fn selected_document(&self) -> Option<&Document> {
        let i = self.documents_table.selected()?;
        self.documents.get(i)
    }

    pub fn select_next(&mut self) {
        match self.level {
            DocsLevel::Documents => Self::step(&mut self.documents_table, self.documents.len(), 1),
            DocsLevel::Items => Self::step(&mut self.items_table, self.items.len(), 1),
        }
    }

    pub fn select_prev(&mut self) {
        match self.level {
            DocsLevel::Documents => Self::step(&mut self.documents_table, self.documents.len(), -1),
            DocsLevel::Items => Self::step(&mut self.items_table, self.items.len(), -1),
        }
    }

    fn step(state: &mut TableState, len: usize, delta: i32) {
        if len == 0 {
            state.select(None);
            return;
        }
        let cur = state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(len as i32)) as usize;
        state.select(Some(next));
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &mut DocsState) {
    if let Some(err) = state.error.as_deref() {
        let block = Block::default()
            .title(" documents · load error ")
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

    match state.level {
        DocsLevel::Documents => render_documents(frame, area, state),
        DocsLevel::Items => render_items(frame, area, state),
    }
}

fn render_documents(frame: &mut Frame, area: Rect, state: &mut DocsState) {
    if state.bank_id.is_none() {
        empty_panel(
            frame,
            area,
            " documents ",
            "pick a bank on the banks tab and press → / Enter to drill in",
        );
        return;
    }

    if state.documents.is_empty() {
        empty_panel(
            frame,
            area,
            " documents ",
            "no documents in this bank yet — retain something via the Python CLI",
        );
        return;
    }

    let rows: Vec<Row> = state
        .documents
        .iter()
        .map(|d| {
            Row::new(vec![
                short_uuid(&d.id),
                d.item_count.to_string(),
                d.source.clone().unwrap_or_else(|| "—".into()),
                d.tags.join(","),
                d.created_at.format("%Y-%m-%d %H:%M").to_string(),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(10),
        Constraint::Length(6),
        Constraint::Min(20),
        Constraint::Length(20),
        Constraint::Length(17),
    ];

    let header = Row::new(vec!["id", "items", "source", "tags", "created"])
        .style(theme::respect_no_color(theme::table_header()));

    let title = format!(
        " documents · {} · {} ",
        state.bank_id.as_deref().unwrap_or("?"),
        state.documents.len()
    );
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

    frame.render_stateful_widget(table, area, &mut state.documents_table);
}

fn render_items(frame: &mut Frame, area: Rect, state: &mut DocsState) {
    let title = format!(
        " items · doc {} · {} ",
        state
            .focused_document
            .as_ref()
            .map(short_uuid)
            .unwrap_or_else(|| "?".into()),
        state.items.len()
    );

    if state.items.is_empty() {
        empty_panel(frame, area, &title, "no memory_items for this document");
        return;
    }

    let rows: Vec<Row> = state
        .items
        .iter()
        .map(|it| {
            Row::new(vec![
                short_uuid(&it.id),
                if it.llm_generated {
                    "llm".into()
                } else {
                    "caller".into()
                },
                it.tags.join(","),
                it.content.clone(),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(10),
        Constraint::Length(7),
        Constraint::Length(16),
        Constraint::Min(20),
    ];

    let header = Row::new(vec!["id", "src", "tags", "content (index_text)"])
        .style(theme::respect_no_color(theme::table_header()));

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

    frame.render_stateful_widget(table, area, &mut state.items_table);
}

fn empty_panel(frame: &mut Frame, area: Rect, title: &str, body: &str) {
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(theme::respect_no_color(theme::dim()));
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            body.to_string(),
            theme::respect_no_color(theme::dim()),
        )))
        .block(block)
        .wrap(Wrap { trim: false }),
        area,
    );
}

pub fn short_uuid(id: &Uuid) -> String {
    let s = id.to_string();
    s.chars().take(8).collect()
}
