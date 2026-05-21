//! Per-bank dashboard — health signals at a glance.
//!
//! 2x2 card grid + per-prompt llm_call breakdown table along the bottom.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, Wrap},
    Frame,
};

use crate::{
    db::{DashboardStats, LlmCallStat},
    theme,
};

pub struct DashboardState {
    pub bank_id: Option<String>,
    pub stats: Option<DashboardStats>,
    pub error: Option<String>,
}

impl DashboardState {
    pub fn new() -> Self {
        Self {
            bank_id: None,
            stats: None,
            error: None,
        }
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &mut DashboardState) {
    if let Some(err) = state.error.as_deref() {
        let block = Block::default()
            .title(" dashboard · load error ")
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

    let Some(stats) = state.stats.as_ref() else {
        let body = match state.bank_id.as_deref() {
            Some(b) => format!("loading dashboard for {b}…"),
            None => "pick a bank on the banks tab to populate the dashboard".into(),
        };
        let block = Block::default()
            .title(" dashboard ")
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
    };

    // Split: 2x2 card grid on top (~12 rows), llm_calls table below.
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(14), Constraint::Min(5)])
        .split(area);

    render_cards(frame, outer[0], stats);
    render_llm_table(frame, outer[1], stats);
}

fn render_cards(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    let top_bottom = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(top_bottom[0]);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(top_bottom[1]);

    render_substrate_card(frame, top[0], s);
    render_activity_card(frame, top[1], s);
    render_latency_card(frame, bottom[0], s);
    render_health_card(frame, bottom[1], s);
}

fn card_block(title: &str) -> Block<'static> {
    Block::default()
        .title(format!(" {title} "))
        .borders(Borders::ALL)
        .border_style(theme::respect_no_color(theme::dim()))
}

fn kv(key: &str, val: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {key:<14} "),
            theme::respect_no_color(theme::dim()),
        ),
        Span::styled(val, Style::default().add_modifier(Modifier::BOLD)),
    ])
}

fn kv_styled(key: &str, val: String, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {key:<14} "),
            theme::respect_no_color(theme::dim()),
        ),
        Span::styled(
            val,
            theme::respect_no_color(style).add_modifier(Modifier::BOLD),
        ),
    ])
}

fn render_substrate_card(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    let lines = vec![
        kv("bank", s.bank_id.clone()),
        kv("embedding_dim", s.embedding_dim.to_string()),
        kv("documents", s.documents.to_string()),
        kv("memory_items", s.memory_items.to_string()),
    ];
    frame.render_widget(Paragraph::new(lines).block(card_block("substrate")), area);
}

fn render_activity_card(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    let lines = vec![
        Line::from(Span::styled(
            "  last 24h",
            theme::respect_no_color(theme::dim()),
        )),
        kv("retains", s.retains_24h.to_string()),
        kv("recalls", s.recalls_24h.to_string()),
        kv("formulates", s.formulates_24h.to_string()),
    ];
    frame.render_widget(Paragraph::new(lines).block(card_block("activity")), area);
}

fn render_latency_card(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    let recall = match s.mean_recall_ms_24h {
        Some(ms) => format_ms(ms),
        None => "—".into(),
    };
    let retain = match s.mean_retain_ms_24h {
        Some(ms) => format_ms(ms),
        None => "—".into(),
    };
    let total_llm = format_total_ms(s.total_llm_ms_24h);
    let lines = vec![
        Line::from(Span::styled(
            "  last 24h",
            theme::respect_no_color(theme::dim()),
        )),
        kv("mean recall", recall),
        kv("mean retain", retain),
        kv("Σ llm time", total_llm),
    ];
    frame.render_widget(
        Paragraph::new(lines).block(card_block("latency · llm-time")),
        area,
    );
}

fn render_health_card(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    // Formulate parse_fallback rate is the load-bearing P18 health signal.
    let fallback_str = match s.formulate_fallback_rate_24h {
        Some(r) => format!(
            "{:.1}%  ({}/{})",
            r * 100.0,
            (r * s.formulates_24h as f64).round() as i64,
            s.formulates_24h
        ),
        None => "— (no formulates in 24h)".into(),
    };
    let fallback_style = match s.formulate_fallback_rate_24h {
        Some(r) if r >= 0.10 => theme::error(),
        Some(r) if r > 0.0 => Style::default().fg(Color::LightYellow),
        _ => Style::default().fg(Color::LightGreen),
    };

    let total_calls: i64 = s.llm_calls.iter().map(|c| c.calls).sum();
    let total_errors: i64 = s.llm_calls.iter().map(|c| c.errors).sum();
    let err_str = if total_calls == 0 {
        "—".into()
    } else {
        format!(
            "{}/{} ({:.1}%)",
            total_errors,
            total_calls,
            total_errors as f64 / total_calls as f64 * 100.0
        )
    };
    let err_style = if total_errors > 0 {
        theme::error()
    } else {
        Style::default().fg(Color::LightGreen)
    };

    let lines = vec![
        Line::from(Span::styled(
            "  last 24h · P18 safety net",
            theme::respect_no_color(theme::dim()),
        )),
        kv_styled("parse_fallback", fallback_str, fallback_style),
        kv_styled("llm errors", err_str, err_style),
    ];
    frame.render_widget(Paragraph::new(lines).block(card_block("health")), area);
}

fn render_llm_table(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    if s.llm_calls.is_empty() {
        let block = card_block("llm calls · last 24h");
        let body = Line::from(Span::styled(
            "  (no llm_calls in the last 24 hours)",
            theme::respect_no_color(theme::dim()),
        ));
        frame.render_widget(Paragraph::new(body).block(block), area);
        return;
    }

    let rows: Vec<Row> = s
        .llm_calls
        .iter()
        .map(|c: &LlmCallStat| {
            let err_marker = if c.errors > 0 {
                Span::styled("!", theme::respect_no_color(theme::error()))
            } else {
                Span::raw(" ")
            };
            Row::new(vec![
                Line::from(c.prompt_name.clone()),
                Line::from(c.calls.to_string()),
                Line::from(format_ms(c.avg_ms)),
                Line::from(format!("{}ms", c.max_ms)),
                Line::from(vec![
                    err_marker,
                    Span::raw(" "),
                    Span::raw(c.errors.to_string()),
                ]),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(14),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Min(8),
    ];

    let header = Row::new(vec!["prompt", "calls", "avg", "max", "errors"])
        .style(theme::respect_no_color(theme::table_header()));

    let table = Table::new(rows, widths)
        .header(header)
        .block(card_block("llm calls · last 24h"));
    frame.render_widget(table, area);
}

// --- helpers ---

fn format_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

fn format_total_ms(ms: i64) -> String {
    if ms >= 60_000 {
        format!("{:.1}m", ms as f64 / 60_000.0)
    } else if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{}ms", ms)
    }
}
