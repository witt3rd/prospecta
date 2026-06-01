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
    db::{DashboardStats, LlmCallStat, SweepStat},
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

    // Split: 2x2 card grid on top (~14 rows), then a bottom region holding
    // the llm_calls table and the sweep-status table side by side.
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(14), Constraint::Min(5)])
        .split(area);

    render_cards(frame, outer[0], stats);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(outer[1]);

    render_llm_table(frame, bottom[0], stats);
    render_sweep_table(frame, bottom[1], stats);
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

fn render_sweep_table(frame: &mut Frame, area: Rect, s: &DashboardStats) {
    if s.sweeps.is_empty() {
        let block = card_block("sweep status");
        let body = Line::from(Span::styled(
            "  (this bank has never been swept)",
            theme::respect_no_color(theme::dim()),
        ));
        frame.render_widget(Paragraph::new(body).block(block), area);
        return;
    }

    let rows: Vec<Row> = s
        .sweeps
        .iter()
        .map(|sw: &SweepStat| {
            // A pass with a fatal error or any per-file errors is unhealthy.
            let unhealthy = sw.error.is_some() || sw.errors_count > 0;
            // Mid-flight pass (started, never ended) is worth flagging too.
            let in_flight = sw.ended_at.is_none();

            let status = if in_flight {
                Span::styled("running", theme::respect_no_color(theme::accent()))
            } else if unhealthy {
                Span::styled("error", theme::respect_no_color(theme::error()))
            } else {
                Span::styled("ok", Style::default().fg(Color::LightGreen))
            };

            let errors_cell = if sw.errors_count > 0 {
                Span::styled(
                    sw.errors_count.to_string(),
                    theme::respect_no_color(theme::error()),
                )
            } else {
                Span::raw(sw.errors_count.to_string())
            };

            Row::new(vec![
                Line::from(shorten_path(&sw.corpus_path)),
                Line::from(status),
                Line::from(format!("{}/{}", sw.files_indexed, sw.files_seen)),
                Line::from(errors_cell),
                Line::from(Span::styled(
                    format_age(sw.started_at),
                    theme::respect_no_color(theme::dim()),
                )),
            ])
        })
        .collect();

    let widths = [
        Constraint::Min(12),
        Constraint::Length(8),
        Constraint::Length(9),
        Constraint::Length(6),
        Constraint::Length(8),
    ];

    let header = Row::new(vec!["corpus", "status", "idx/seen", "errs", "age"])
        .style(theme::respect_no_color(theme::table_header()));

    let table = Table::new(rows, widths)
        .header(header)
        .block(card_block("sweep status · latest pass per corpus"));
    frame.render_widget(table, area);
}

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

/// Tail a long corpus path so the table cell stays legible:
/// `/home/dt/notes` → `notes`, `/a/b/c/d` → `…/c/d`.
fn shorten_path(path: &str) -> String {
    let parts: Vec<&str> = path
        .trim_end_matches('/')
        .split('/')
        .filter(|p| !p.is_empty())
        .collect();
    match parts.len() {
        0 => path.to_string(),
        1 => parts[0].to_string(),
        _ => {
            let tail = &parts[parts.len().saturating_sub(2)..];
            format!("…/{}", tail.join("/"))
        }
    }
}

/// Coarse "time since" for the sweep age column. The load-bearing signal is
/// "did the sweeper stop running?" — so a stale last-pass should read loud:
/// `12s`, `5m`, `3h`, `4d`.
fn format_age(ts: chrono::DateTime<chrono::Utc>) -> String {
    let secs = (chrono::Utc::now() - ts).num_seconds().max(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86_400)
    }
}
