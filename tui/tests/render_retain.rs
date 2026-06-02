//! Headless render tests for the retain modal.

use prospecta_tui::views::retain::{self, RetainField, RetainState, RetainStatus};
use ratatui::{backend::TestBackend, Terminal};

fn buf_to_string(t: &Terminal<TestBackend>) -> String {
    let buf = t.backend().buffer().clone();
    let mut s = String::new();
    for cell in buf.content.iter() {
        s.push_str(cell.symbol());
    }
    s
}

fn render(state: &RetainState) -> String {
    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| retain::render(f, f.area(), state))
        .unwrap();
    buf_to_string(&terminal)
}

#[test]
fn retain_renders_all_fields_and_bank() {
    let mut state = RetainState::new();
    state.bank_id = Some("tui-st".into());
    let s = render(&state);
    assert!(s.contains("retain · tui-st"), "title with bank missing");
    assert!(s.contains("content"), "content field missing");
    assert!(s.contains("source"), "source field missing");
    assert!(s.contains("tags"), "tags field missing");
    assert!(s.contains("index_text"), "index_text field missing");
    assert!(s.contains("Ctrl-S"), "submit hint missing");
}

#[test]
fn retain_shows_typed_content() {
    let mut state = RetainState::new();
    state.bank_id = Some("tui-st".into());
    state.content = "Kelly loves Rick Springfield".into();
    let s = render(&state);
    assert!(
        s.contains("Kelly loves Rick Springfield"),
        "content text missing"
    );
}

#[test]
fn retain_ok_status_shows_document_id() {
    let mut state = RetainState::new();
    state.bank_id = Some("tui-st".into());
    state.status = RetainStatus::Ok("ead4453b-2f4b-41f7-9b1b-8961183df4eb".into());
    let s = render(&state);
    assert!(s.contains("ok"), "ok banner missing");
    assert!(s.contains("ead4453b"), "document_id missing from ok banner");
}

#[test]
fn retain_failed_status_shows_reason() {
    let mut state = RetainState::new();
    state.bank_id = Some("tui-st".into());
    state.status = RetainStatus::Failed("error: retain failed: foreign key constraint".into());
    let s = render(&state);
    assert!(s.contains("failed"), "failed banner missing");
    assert!(s.contains("foreign key"), "failure reason missing");
}

#[test]
fn retain_running_status_banner() {
    let mut state = RetainState::new();
    state.bank_id = Some("tui-st".into());
    state.status = RetainStatus::Running;
    let s = render(&state);
    assert!(s.contains("running"), "running banner missing");
}

#[test]
fn retain_field_focus_cycles() {
    assert_eq!(RetainField::Content.next(), RetainField::Source);
    assert_eq!(RetainField::Source.next(), RetainField::Tags);
    assert_eq!(RetainField::Tags.next(), RetainField::IndexText);
    assert_eq!(RetainField::IndexText.next(), RetainField::Content);
    // prev is the inverse
    assert_eq!(RetainField::Content.prev(), RetainField::IndexText);
    assert_eq!(RetainField::Source.prev(), RetainField::Content);
}

#[test]
fn retain_can_submit_requires_content() {
    let mut state = RetainState::new();
    assert!(!state.can_submit(), "empty content should block submit");
    state.content = "   ".into();
    assert!(
        !state.can_submit(),
        "whitespace-only content should block submit"
    );
    state.content = "real content".into();
    assert!(state.can_submit(), "non-empty content should allow submit");
    // running blocks resubmit
    state.status = RetainStatus::Running;
    assert!(!state.can_submit(), "running should block resubmit");
}

#[test]
fn retain_focused_mut_targets_active_field() {
    let mut state = RetainState::new();
    state.field = RetainField::Source;
    state.push_char('x');
    assert_eq!(state.source, "x");
    assert_eq!(state.content, "");
    state.field = RetainField::Content;
    state.push_char('y');
    assert_eq!(state.content, "y");
}

#[test]
fn retain_reset_fields_clears_but_keeps_bank() {
    let mut state = RetainState::new();
    state.bank_id = Some("tui-st".into());
    state.content = "stuff".into();
    state.source = "src".into();
    state.status = RetainStatus::Ok("id".into());
    state.reset_fields();
    assert_eq!(state.content, "");
    assert_eq!(state.source, "");
    assert!(matches!(state.status, RetainStatus::Idle));
    assert_eq!(
        state.bank_id.as_deref(),
        Some("tui-st"),
        "bank should survive reset"
    );
}
