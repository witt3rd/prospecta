//! Bank list view — axis 1 "manage" entry point.

use ratatui::{
    layout::{Constraint, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Row, Table, TableState},
    Frame,
};

use crate::{db::BankSummary, theme};

pub struct BankListState {
    pub table: TableState,
}

impl BankListState {
    pub fn new() -> Self {
        let mut t = TableState::default();
        t.select(Some(0));
        Self { table: t }
    }

    pub fn select_next(&mut self, len: usize) {
        if len == 0 {
            self.table.select(None);
            return;
        }
        let i = self.table.selected().map(|i| (i + 1) % len).unwrap_or(0);
        self.table.select(Some(i));
    }

    pub fn select_prev(&mut self, len: usize) {
        if len == 0 {
            self.table.select(None);
            return;
        }
        let i = self
            .table
            .selected()
            .map(|i| if i == 0 { len - 1 } else { i - 1 })
            .unwrap_or(0);
        self.table.select(Some(i));
    }
}

pub fn render(
    frame: &mut Frame,
    area: Rect,
    banks: &[BankSummary],
    state: &mut BankListState,
    load_error: Option<&str>,
) {
    if let Some(err) = load_error {
        let block = Block::default()
            .title(" banks · load error ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::error()));
        let body = Line::from(Span::styled(
            err.to_string(),
            theme::respect_no_color(theme::error()),
        ));
        frame.render_widget(ratatui::widgets::Paragraph::new(body).block(block), area);
        return;
    }

    if banks.is_empty() {
        let block = Block::default()
            .title(" banks ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::dim()));
        let body = Line::from(Span::styled(
            "no banks yet — create one with `prospecta create-bank --id <id> --embedding-dim <n>`",
            theme::respect_no_color(theme::dim()),
        ));
        frame.render_widget(ratatui::widgets::Paragraph::new(body).block(block), area);
        return;
    }

    let rows: Vec<Row> = banks
        .iter()
        .map(|b| {
            Row::new(vec![
                b.bank.bank_id.clone(),
                b.bank.embedding_dim.to_string(),
                b.documents.to_string(),
                b.memory_items.to_string(),
                b.retain_events.to_string(),
                b.recall_events.to_string(),
                b.bank.mission.clone().unwrap_or_default(),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(20),
        Constraint::Length(7),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Min(10),
    ];

    let header = Row::new(vec![
        "bank_id", "dim", "docs", "items", "retains", "recalls", "mission",
    ])
    .style(theme::respect_no_color(theme::table_header()));

    let title = format!(" banks · {} ", banks.len());
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
