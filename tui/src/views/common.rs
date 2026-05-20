//! Shared widget helpers — header bar, footer bar, status banner.
//!
//! Every view renders these around its body so the chrome stays consistent.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme;

/// Three-row chrome layout: header (1 row), body (rest), footer (1 row).
pub fn chrome_layout(area: Rect) -> [Rect; 3] {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(area);
    [chunks[0], chunks[1], chunks[2]]
}

/// Header bar: "prospecta-tui · <view title> · <redacted DATABASE_URL>".
pub fn render_header(frame: &mut Frame, area: Rect, view_title: &str, redacted_url: &str) {
    let line = Line::from(vec![
        Span::styled("prospecta-tui", theme::respect_no_color(theme::header())),
        Span::styled("  ·  ", theme::respect_no_color(theme::dim())),
        Span::styled(
            view_title.to_string(),
            theme::respect_no_color(theme::accent()),
        ),
        Span::styled("  ·  ", theme::respect_no_color(theme::dim())),
        Span::styled(
            redacted_url.to_string(),
            theme::respect_no_color(theme::dim()),
        ),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// Footer bar: keybinding hints. View supplies what's relevant; `?` and `q` are
/// always present (rendered last by the caller).
pub fn render_footer(frame: &mut Frame, area: Rect, hints: &[(&str, &str)]) {
    let mut spans: Vec<Span> = Vec::new();
    for (i, (key, label)) in hints.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", theme::respect_no_color(theme::footer())));
        }
        spans.push(Span::styled(
            format!(" {key} "),
            theme::respect_no_color(theme::accent()),
        ));
        spans.push(Span::styled(
            label.to_string(),
            theme::respect_no_color(theme::footer()),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Overlay the help cheat-sheet. Pops up centered, dismissed with any key.
pub fn render_help_overlay(frame: &mut Frame, area: Rect, lines: &[(&str, &str)]) {
    // 60% wide, 60% tall, centered.
    let popup = centered_rect(70, 70, area);
    // Clear underneath.
    frame.render_widget(ratatui::widgets::Clear, popup);

    let body: Vec<Line> = lines
        .iter()
        .map(|(key, desc)| {
            Line::from(vec![
                Span::styled(
                    format!("  {key:<12} ", key = key),
                    theme::respect_no_color(theme::accent()),
                ),
                Span::styled((*desc).to_string(), Style::default()),
            ])
        })
        .collect();

    let block = Block::default()
        .title(" keybindings  (press any key to close) ")
        .borders(Borders::ALL)
        .border_style(theme::respect_no_color(theme::help_title()));

    frame.render_widget(Paragraph::new(body).block(block), popup);
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
