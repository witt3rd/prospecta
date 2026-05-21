//! Recall-thread view — the load-bearing observability surface.
//!
//! Shows one recall_event end-to-end: the preceding formulate_event (verbatim
//! user message + raw JSON expansion + parse_fallback diagnostics), the recall
//! itself (queries + per-chunk results with per-channel scores + synthesis),
//! and the llm_calls in the ±10s window around it (verbatim prompts/responses).

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use serde_json::Value;

use crate::{
    db::{FormulateRow, LlmCallRow, RecallRow, RecallThread},
    theme,
};

pub struct RecallThreadState {
    pub thread: Option<RecallThread>,
    pub error: Option<String>,
    pub scroll: u16,
}

impl RecallThreadState {
    pub fn new() -> Self {
        Self {
            thread: None,
            error: None,
            scroll: 0,
        }
    }

    pub fn scroll_down(&mut self) {
        self.scroll = self.scroll.saturating_add(2);
    }

    pub fn scroll_up(&mut self) {
        self.scroll = self.scroll.saturating_sub(2);
    }

    pub fn reset_scroll(&mut self) {
        self.scroll = 0;
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &mut RecallThreadState) {
    if let Some(err) = state.error.as_deref() {
        let block = Block::default()
            .title(" recall thread · load error ")
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

    let Some(thread) = state.thread.as_ref() else {
        empty_panel(
            frame,
            area,
            " recall thread ",
            "press Enter on a recall row in the events tab to load its thread",
        );
        return;
    };

    // Build all lines into a single paragraph and let Ratatui scroll.
    // This keeps the multi-section layout simple and predictable.
    let mut lines: Vec<Line> = Vec::new();
    push_recall_header(&mut lines, &thread.recall);
    push_blank(&mut lines);
    push_formulate_section(&mut lines, thread.formulate.as_ref());
    push_blank(&mut lines);
    push_recall_body(&mut lines, &thread.recall);
    push_blank(&mut lines);
    push_llm_calls_section(&mut lines, &thread.llm_calls);

    let title = format!(
        " recall thread · id={} · bank={} ",
        thread.recall.id, thread.recall.bank_id
    );

    let para = Paragraph::new(lines)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(theme::respect_no_color(theme::dim())),
        )
        .wrap(Wrap { trim: false })
        .scroll((state.scroll, 0));

    frame.render_widget(para, area);
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

fn section_header(label: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("── {} ", label),
            theme::respect_no_color(theme::accent()).add_modifier(Modifier::BOLD),
        ),
        Span::styled("─".repeat(40), theme::respect_no_color(theme::dim())),
    ])
}

fn kv_line(key: &str, val: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key}: "), theme::respect_no_color(theme::dim())),
        Span::raw(val),
    ])
}

fn kv_line_styled(key: &str, val: String, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key}: "), theme::respect_no_color(theme::dim())),
        Span::styled(val, theme::respect_no_color(style)),
    ])
}

fn push_blank(lines: &mut Vec<Line<'static>>) {
    lines.push(Line::from(""));
}

fn push_recall_header(lines: &mut Vec<Line<'static>>, r: &RecallRow) {
    lines.push(section_header("recall"));
    lines.push(kv_line(
        "created_at",
        r.created_at.format("%Y-%m-%d %H:%M:%S%.3f %Z").to_string(),
    ));
    lines.push(kv_line("mode", r.mode.clone()));
    lines.push(kv_line("n_results", r.n_results.to_string()));
    lines.push(kv_line("duration", format_duration(r.duration_ms)));
}

fn push_formulate_section(lines: &mut Vec<Line<'static>>, f: Option<&FormulateRow>) {
    lines.push(section_header("formulate (preceding)"));
    let Some(f) = f else {
        lines.push(Line::from(Span::styled(
            "  (no formulate_event found within the bank/time window)",
            theme::respect_no_color(theme::dim()),
        )));
        return;
    };
    lines.push(kv_line("id", f.id.to_string()));
    lines.push(kv_line(
        "created_at",
        f.created_at.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
    ));
    lines.push(kv_line("message", f.message.clone()));
    lines.push(kv_line("n_queries_out", f.n_queries_out.to_string()));
    lines.push(kv_line("json_mode", f.json_mode_used.to_string()));
    let fallback_str = if f.parse_fallback {
        format!("YES ({})", f.error_kind.as_deref().unwrap_or("?"))
    } else {
        "no".into()
    };
    lines.push(kv_line_styled(
        "parse_fallback",
        fallback_str,
        if f.parse_fallback {
            theme::error()
        } else {
            Style::default()
        },
    ));
    lines.push(kv_line("duration", format_duration(f.duration_ms)));
    lines.push(Line::from(Span::styled(
        "  raw_response:",
        theme::respect_no_color(theme::dim()),
    )));
    for chunk in chunk_lines(&f.raw_response, 96) {
        lines.push(Line::from(format!("    {chunk}")));
    }
}

fn push_recall_body(lines: &mut Vec<Line<'static>>, r: &RecallRow) {
    lines.push(section_header("recall body"));
    lines.push(Line::from(Span::styled(
        "  queries:",
        theme::respect_no_color(theme::dim()),
    )));
    if let Value::Array(arr) = &r.queries {
        for (i, q) in arr.iter().enumerate() {
            let s = q
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| q.to_string());
            lines.push(Line::from(format!("    {}. {}", i + 1, s)));
        }
    } else {
        lines.push(Line::from(format!("    {}", r.queries)));
    }

    push_blank(lines);
    lines.push(Line::from(Span::styled(
        "  results:",
        theme::respect_no_color(theme::dim()),
    )));
    match &r.results {
        Some(Value::Array(arr)) if !arr.is_empty() => {
            for elem in arr {
                let rank = elem.get("rank").and_then(|v| v.as_i64()).unwrap_or(0);
                let source = elem.get("source").and_then(|v| v.as_str()).unwrap_or("—");
                let doc_id = elem
                    .get("document_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("—");
                let preview = elem
                    .get("content_preview")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("    [{rank}] "),
                        theme::respect_no_color(theme::accent()),
                    ),
                    Span::raw(format!("{}  doc={}", source, short(doc_id, 8))),
                ]));
                if let Some(scores) = elem.get("scores") {
                    lines.push(Line::from(format!("        scores: {}", scores)));
                }
                if !preview.is_empty() {
                    for c in chunk_lines(preview, 88) {
                        lines.push(Line::from(format!("        {c}")));
                    }
                }
            }
        }
        Some(_) | None => {
            lines.push(Line::from(Span::styled(
                "    (no results JSONB on this recall — pre-0003 row or empty result set)",
                theme::respect_no_color(theme::dim()),
            )));
        }
    }

    push_blank(lines);
    lines.push(Line::from(Span::styled(
        "  synthesis:",
        theme::respect_no_color(theme::dim()),
    )));
    match r.synthesis.as_deref() {
        Some(s) if !s.is_empty() => {
            for c in chunk_lines(s, 96) {
                lines.push(Line::from(format!("    {c}")));
            }
        }
        _ => lines.push(Line::from(Span::styled(
            "    (no synthesis — plain recall(), not recall_synth())",
            theme::respect_no_color(theme::dim()),
        ))),
    }
}

fn push_llm_calls_section(lines: &mut Vec<Line<'static>>, calls: &[LlmCallRow]) {
    lines.push(section_header(&format!(
        "llm_calls (±10s window, {})",
        calls.len()
    )));
    if calls.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no llm_calls in the window)",
            theme::respect_no_color(theme::dim()),
        )));
        return;
    }
    for c in calls {
        let when = c.created_at.format("%H:%M:%S%.3f").to_string();
        let err_marker = if c.error.is_some() {
            Span::styled(" ! ", theme::respect_no_color(theme::error()))
        } else {
            Span::raw("   ")
        };
        lines.push(Line::from(vec![
            err_marker,
            Span::styled(
                format!("{:<10} ", c.prompt_name),
                theme::respect_no_color(theme::accent()).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(
                "id={}  {}  dur={}  json={}",
                c.id,
                when,
                format_duration(c.duration_ms),
                c.json_mode
            )),
        ]));
        if let Some(err) = c.error.as_deref() {
            lines.push(Line::from(Span::styled(
                format!("    error: {err}"),
                theme::respect_no_color(theme::error()),
            )));
        }
        if let Some(p) = c.prompt_text.as_deref() {
            lines.push(Line::from(Span::styled(
                "    prompt:",
                theme::respect_no_color(theme::dim()),
            )));
            for chunk in chunk_lines(p, 92) {
                lines.push(Line::from(format!("      {chunk}")));
            }
        }
        if let Some(r) = c.response_text.as_deref() {
            lines.push(Line::from(Span::styled(
                "    response:",
                theme::respect_no_color(theme::dim()),
            )));
            for chunk in chunk_lines(r, 92) {
                lines.push(Line::from(format!("      {chunk}")));
            }
        }
        if c.prompt_text.is_none() && c.response_text.is_none() && c.error.is_none() {
            lines.push(Line::from(Span::styled(
                "    (no prompt/response captured — persist_llm_text=false or pre-0003)",
                theme::respect_no_color(theme::dim()),
            )));
        }
        lines.push(Line::from(""));
    }
}

// --- helpers ---

fn format_duration(ms: i32) -> String {
    if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{}ms", ms)
    }
}

fn short(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Soft-wrap long strings on whitespace at ~width columns. Newlines in the
/// source are preserved as separate output lines.
fn chunk_lines(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split('\n') {
        if raw.len() <= width {
            out.push(raw.to_string());
            continue;
        }
        let mut cur = String::new();
        for word in raw.split(' ') {
            if cur.is_empty() {
                cur.push_str(word);
            } else if cur.len() + 1 + word.len() <= width {
                cur.push(' ');
                cur.push_str(word);
            } else {
                out.push(std::mem::take(&mut cur));
                cur.push_str(word);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}
