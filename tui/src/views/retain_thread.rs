//! Retain-thread view — end-to-end inspection of one retain_event.
//!
//! Sibling to recall_thread.rs. Shows the retain_event, its document and
//! source text, all memory_items produced, and the paired index_text llm_call
//! (the call that produced the question-shaped index_text strings, if LLM-authored).

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::{
    db::{DocumentRow, IndexTextCall, ItemRow, RetainRow, RetainThread},
    theme,
};

pub struct RetainThreadState {
    pub thread: Option<RetainThread>,
    pub error: Option<String>,
    pub scroll: u16,
}

impl RetainThreadState {
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

pub fn render(frame: &mut Frame, area: Rect, state: &mut RetainThreadState) {
    if let Some(err) = state.error.as_deref() {
        let block = Block::default()
            .title(" retain thread · load error ")
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
        let block = Block::default()
            .title(" retain thread ")
            .borders(Borders::ALL)
            .border_style(theme::respect_no_color(theme::dim()));
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "press Enter on a retain row in the events tab to load its thread",
                theme::respect_no_color(theme::dim()),
            )))
            .block(block)
            .wrap(Wrap { trim: false }),
            area,
        );
        return;
    };

    let mut lines: Vec<Line> = Vec::new();
    push_retain_header(&mut lines, &thread.retain);
    push_blank(&mut lines);
    push_document_section(&mut lines, thread.document.as_ref());
    push_blank(&mut lines);
    push_items_section(&mut lines, &thread.items);
    push_blank(&mut lines);
    push_index_text_section(&mut lines, &thread.retain, thread.index_text_call.as_ref());

    let title = format!(
        " retain thread · id={} · bank={} ",
        thread.retain.id, thread.retain.bank_id
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

// --- sections ---

fn section_header(label: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("── {} ", label),
            theme::respect_no_color(theme::accent()).add_modifier(Modifier::BOLD),
        ),
        Span::styled("─".repeat(40), theme::respect_no_color(theme::dim())),
    ])
}

fn kv(key: &str, val: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key}: "), theme::respect_no_color(theme::dim())),
        Span::raw(val),
    ])
}

fn kv_styled(key: &str, val: String, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key}: "), theme::respect_no_color(theme::dim())),
        Span::styled(val, theme::respect_no_color(style)),
    ])
}

fn push_blank(lines: &mut Vec<Line<'static>>) {
    lines.push(Line::from(""));
}

fn push_retain_header(lines: &mut Vec<Line<'static>>, r: &RetainRow) {
    lines.push(section_header("retain"));
    lines.push(kv(
        "created_at",
        r.created_at.format("%Y-%m-%d %H:%M:%S%.3f %Z").to_string(),
    ));
    lines.push(kv("items_count", r.items_count.to_string()));
    lines.push(kv(
        "index_text_caller_supplied",
        r.index_text_caller_supplied.to_string(),
    ));
    lines.push(kv("duration", format_duration(r.duration_ms)));
    if let Some(err) = r.error.as_deref() {
        lines.push(kv_styled("error", err.to_string(), theme::error()));
    }
}

fn push_document_section(lines: &mut Vec<Line<'static>>, doc: Option<&DocumentRow>) {
    lines.push(section_header("document"));
    let Some(d) = doc else {
        lines.push(Line::from(Span::styled(
            "  (no document linked — retain pre-dated documents column or doc was deleted)",
            theme::respect_no_color(theme::dim()),
        )));
        return;
    };
    lines.push(kv("id", short_uuid(&d.id)));
    lines.push(kv("source", d.source.clone().unwrap_or_else(|| "—".into())));
    lines.push(kv("content_hash", d.content_hash.clone()));
    if !d.tags.is_empty() {
        lines.push(kv("tags", d.tags.join(",")));
    }
    lines.push(kv(
        "created_at",
        d.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
    ));
    lines.push(Line::from(Span::styled(
        "  original_text:",
        theme::respect_no_color(theme::dim()),
    )));
    for c in chunk_lines(&d.original_text, 92) {
        lines.push(Line::from(format!("    {c}")));
    }
}

fn push_items_section(lines: &mut Vec<Line<'static>>, items: &[ItemRow]) {
    lines.push(section_header(&format!(
        "memory_items ({} index_text strings)",
        items.len()
    )));
    if items.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no memory_items linked to this document)",
            theme::respect_no_color(theme::dim()),
        )));
        return;
    }
    for (i, it) in items.iter().enumerate() {
        let src_marker = if it.llm_generated { "llm   " } else { "caller" };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  [{:>2}] ", i + 1),
                theme::respect_no_color(theme::accent()),
            ),
            Span::styled(
                format!("{src_marker} "),
                theme::respect_no_color(theme::dim()),
            ),
            Span::raw(it.content.clone()),
        ]));
        if !it.tags.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("        tags: {}", it.tags.join(",")),
                theme::respect_no_color(theme::dim()),
            )));
        }
    }
}

fn push_index_text_section(
    lines: &mut Vec<Line<'static>>,
    retain: &RetainRow,
    call: Option<&IndexTextCall>,
) {
    lines.push(section_header("index_text llm_call (±10s)"));

    if retain.index_text_caller_supplied {
        lines.push(Line::from(Span::styled(
            "  (skipped — index_text was caller-supplied; no LLM call expected)",
            theme::respect_no_color(theme::dim()),
        )));
        if let Some(gen) = retain.index_text_generated.as_ref() {
            if !gen.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!(
                        "  (but {} index_text_generated entries on the retain row — inspect for drift)",
                        gen.len()
                    ),
                    theme::respect_no_color(theme::dim()),
                )));
            }
        }
        return;
    }

    let Some(c) = call else {
        lines.push(Line::from(Span::styled(
            "  (no index_text llm_call found in the bank/time window)",
            theme::respect_no_color(theme::dim()),
        )));
        if let Some(raw) = retain.raw_llm_response.as_deref() {
            lines.push(Line::from(Span::styled(
                "  raw_llm_response (from retain row):",
                theme::respect_no_color(theme::dim()),
            )));
            for c in chunk_lines(raw, 92) {
                lines.push(Line::from(format!("    {c}")));
            }
        }
        return;
    };

    lines.push(kv("id", c.id.to_string()));
    lines.push(kv(
        "created_at",
        c.created_at.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
    ));
    lines.push(kv("duration", format_duration(c.duration_ms)));
    lines.push(kv("json_mode", c.json_mode.to_string()));
    if let Some(err) = c.error.as_deref() {
        lines.push(kv_styled("error", err.to_string(), theme::error()));
    }
    if let Some(p) = c.prompt_text.as_deref() {
        lines.push(Line::from(Span::styled(
            "  prompt:",
            theme::respect_no_color(theme::dim()),
        )));
        for chunk in chunk_lines(p, 92) {
            lines.push(Line::from(format!("    {chunk}")));
        }
    }
    if let Some(r) = c.response_text.as_deref() {
        lines.push(Line::from(Span::styled(
            "  response:",
            theme::respect_no_color(theme::dim()),
        )));
        for chunk in chunk_lines(r, 92) {
            lines.push(Line::from(format!("    {chunk}")));
        }
    }
    if c.prompt_text.is_none() && c.response_text.is_none() && c.error.is_none() {
        lines.push(Line::from(Span::styled(
            "  (no prompt/response captured — persist_llm_text=false or pre-0003)",
            theme::respect_no_color(theme::dim()),
        )));
    }

    if let Some(gen) = retain.index_text_generated.as_ref() {
        push_blank(lines);
        lines.push(Line::from(Span::styled(
            format!("  index_text_generated (verbatim, {} strings):", gen.len()),
            theme::respect_no_color(theme::dim()),
        )));
        for (i, q) in gen.iter().enumerate() {
            lines.push(Line::from(format!("    {}. {}", i + 1, q)));
        }
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

fn short_uuid(id: &uuid::Uuid) -> String {
    id.to_string().chars().take(8).collect()
}

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
