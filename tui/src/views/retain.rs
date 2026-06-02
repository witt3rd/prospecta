//! Retain form — manual write path via shell-out to `prospecta retain`.
//!
//! A modal overlay (opened from the search tab with `R`) with four text
//! fields and a submit action. On submit the app shells out to the Python
//! CLI; this view renders the honest outcome — the document UUID on success
//! or the verbatim CLI stderr on failure. It never fabricates success.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::theme;

/// Which field has focus. Order matches Tab traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainField {
    Content,
    Source,
    Tags,
    IndexText,
}

impl RetainField {
    pub fn next(self) -> Self {
        match self {
            RetainField::Content => RetainField::Source,
            RetainField::Source => RetainField::Tags,
            RetainField::Tags => RetainField::IndexText,
            RetainField::IndexText => RetainField::Content,
        }
    }
    pub fn prev(self) -> Self {
        match self {
            RetainField::Content => RetainField::IndexText,
            RetainField::Source => RetainField::Content,
            RetainField::Tags => RetainField::Source,
            RetainField::IndexText => RetainField::Tags,
        }
    }
}

/// Outcome banner state after a submit.
#[derive(Debug, Clone)]
pub enum RetainStatus {
    /// Nothing submitted yet.
    Idle,
    /// Submit in flight (the shell-out is awaiting).
    Running,
    /// CLI returned success; carries the document UUID.
    Ok(String),
    /// CLI failed; carries a short reason (verbatim stderr tail).
    Failed(String),
}

pub struct RetainState {
    pub bank_id: Option<String>,
    pub content: String,
    pub source: String,
    pub tags: String,
    /// Single index_text line for v0.1 (the offline-skip-LLM path). Empty =
    /// let the CLI generate via LLM (needs litellm + a key).
    pub index_text: String,
    pub field: RetainField,
    pub status: RetainStatus,
}

impl RetainState {
    pub fn new() -> Self {
        Self {
            bank_id: None,
            content: String::new(),
            source: String::new(),
            tags: String::new(),
            index_text: String::new(),
            field: RetainField::Content,
            status: RetainStatus::Idle,
        }
    }

    /// Reset the editable fields for a fresh form, preserving the bound bank.
    pub fn reset_fields(&mut self) {
        self.content.clear();
        self.source.clear();
        self.tags.clear();
        self.index_text.clear();
        self.field = RetainField::Content;
        self.status = RetainStatus::Idle;
    }

    pub fn focused_mut(&mut self) -> &mut String {
        match self.field {
            RetainField::Content => &mut self.content,
            RetainField::Source => &mut self.source,
            RetainField::Tags => &mut self.tags,
            RetainField::IndexText => &mut self.index_text,
        }
    }

    pub fn push_char(&mut self, c: char) {
        self.focused_mut().push(c);
    }

    pub fn backspace(&mut self) {
        self.focused_mut().pop();
    }

    /// Content is the only required field — block submit on empty content.
    pub fn can_submit(&self) -> bool {
        !self.content.trim().is_empty() && !matches!(self.status, RetainStatus::Running)
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &RetainState) {
    // Centered modal.
    let popup = centered_rect(80, 80, area);
    frame.render_widget(Clear, popup);

    let bank = state.bank_id.as_deref().unwrap_or("(no bank)");
    let outer = Block::default()
        .title(format!(" retain · {bank} "))
        .borders(Borders::ALL)
        .border_style(theme::respect_no_color(theme::accent()));
    let inner = outer.inner(popup);
    frame.render_widget(outer, popup);

    // content(5) · source(3) · tags(3) · index_text(3) · status(rest) · hint(1)
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(inner);

    field_box(
        frame,
        rows[0],
        "content (required)",
        &state.content,
        state.field == RetainField::Content,
        true,
    );
    field_box(
        frame,
        rows[1],
        "source",
        &state.source,
        state.field == RetainField::Source,
        false,
    );
    field_box(
        frame,
        rows[2],
        "tags (comma-separated)",
        &state.tags,
        state.field == RetainField::Tags,
        false,
    );
    field_box(
        frame,
        rows[3],
        "index_text (empty → LLM-generated)",
        &state.index_text,
        state.field == RetainField::IndexText,
        false,
    );

    render_status(frame, rows[4], state);

    let hint = Line::from(vec![Span::styled(
        " Tab/↑↓ field · Enter newline (content) · Ctrl-S submit · Esc cancel",
        theme::respect_no_color(theme::dim()),
    )]);
    frame.render_widget(Paragraph::new(hint), rows[5]);
}

fn field_box(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    value: &str,
    focused: bool,
    multiline: bool,
) {
    let border_style = if focused {
        theme::respect_no_color(theme::accent())
    } else {
        theme::respect_no_color(theme::dim())
    };
    let title = if focused {
        format!(" {label} ◂ ")
    } else {
        format!(" {label} ")
    };
    // Cursor block on the focused field.
    let body = if focused {
        format!("{value}▏")
    } else {
        value.to_string()
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(border_style);
    let p = Paragraph::new(body).block(block);
    let p = if multiline {
        p.wrap(Wrap { trim: false })
    } else {
        p
    };
    frame.render_widget(p, area);
}

fn render_status(frame: &mut Frame, area: Rect, state: &RetainState) {
    let (title, body, style) = match &state.status {
        RetainStatus::Idle => (
            " result ",
            "fill content, then Ctrl-S to retain via the prospecta CLI".to_string(),
            theme::dim(),
        ),
        RetainStatus::Running => (
            " result · running ",
            "shelling out to `prospecta retain`…".to_string(),
            theme::accent(),
        ),
        RetainStatus::Ok(id) => (
            " result · ok ",
            format!("retained · document_id={id}"),
            Style::default().fg(ratatui::style::Color::LightGreen),
        ),
        RetainStatus::Failed(msg) => (" result · failed ", msg.clone(), theme::error()),
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(theme::respect_no_color(style));
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            body,
            theme::respect_no_color(style).add_modifier(Modifier::BOLD),
        )))
        .block(block)
        .wrap(Wrap { trim: false }),
        area,
    );
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let v = Layout::default()
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
        .split(v[1])[1]
}
