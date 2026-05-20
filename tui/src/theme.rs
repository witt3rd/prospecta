//! Shared theme — keep style decisions in one place so views stay consistent.

use ratatui::style::{Color, Modifier, Style};

pub fn header() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

pub fn footer() -> Style {
    Style::default().fg(Color::DarkGray)
}

pub fn selected_row() -> Style {
    Style::default()
        .bg(Color::Indexed(236))
        .add_modifier(Modifier::BOLD)
}

pub fn table_header() -> Style {
    Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD)
}

pub fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

pub fn accent() -> Style {
    Style::default().fg(Color::LightMagenta)
}

pub fn error() -> Style {
    Style::default()
        .fg(Color::LightRed)
        .add_modifier(Modifier::BOLD)
}

pub fn help_title() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

/// Honor NO_COLOR (accessibility convention). When set to any value, return a
/// plain default style; otherwise pass `s` through unchanged.
pub fn respect_no_color(s: Style) -> Style {
    if std::env::var_os("NO_COLOR").is_some() {
        Style::default()
    } else {
        s
    }
}
