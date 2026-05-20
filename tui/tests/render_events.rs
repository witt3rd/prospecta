//! Headless render tests for the events view.

use chrono::{Duration, Utc};
use prospecta_tui::db::{Event, EventKind};
use prospecta_tui::views::{common, observability};
use ratatui::{backend::TestBackend, Terminal};

fn dummy_events() -> Vec<Event> {
    let now = Utc::now();
    vec![
        Event {
            kind: EventKind::Recall,
            id: 17,
            bank_id: Some("default".into()),
            created_at: now,
            duration_ms: Some(187),
            has_error: false,
            summary: "mode=hybrid n=5 +synth".into(),
        },
        Event {
            kind: EventKind::Formulate,
            id: 9,
            bank_id: Some("default".into()),
            created_at: now - Duration::seconds(2),
            duration_ms: Some(89),
            has_error: true,
            summary: "n_out=2 FALLBACK=malformed_json".into(),
        },
        Event {
            kind: EventKind::LlmCall,
            id: 42,
            bank_id: Some("default".into()),
            created_at: now - Duration::seconds(3),
            duration_ms: Some(1100),
            has_error: false,
            summary: "synthesize".into(),
        },
        Event {
            kind: EventKind::Retain,
            id: 4,
            bank_id: Some("default".into()),
            created_at: now - Duration::seconds(5),
            duration_ms: Some(980),
            has_error: false,
            summary: "items=4 caller_supplied=false".into(),
        },
        Event {
            kind: EventKind::SweepPass,
            id: 2,
            bank_id: Some("default".into()),
            created_at: now - Duration::minutes(60),
            duration_ms: Some(6044),
            has_error: true,
            summary: "/home/dt/notes seen=142 idx=0 pruned=0 err=1".into(),
        },
    ]
}

fn buffer_to_string(terminal: &Terminal<TestBackend>) -> String {
    let buf = terminal.backend().buffer().clone();
    let mut s = String::new();
    for cell in buf.content.iter() {
        s.push_str(cell.symbol());
    }
    s
}

#[test]
fn event_stream_renders_all_kinds() {
    let events = dummy_events();
    let mut state = observability::EventStreamState::new();

    let backend = TestBackend::new(140, 22);
    let mut terminal = Terminal::new(backend).expect("init test terminal");

    terminal
        .draw(|frame| {
            let area = frame.area();
            let [_, body, _] = common::chrome_layout(area);
            observability::render(frame, body, &events, &mut state, None, None);
        })
        .expect("draw events");

    let s = buffer_to_string(&terminal);
    // All five event kinds painted.
    assert!(s.contains("retain"), "retain tag missing");
    assert!(s.contains("recall"), "recall tag missing");
    assert!(s.contains("formulate"), "formulate tag missing");
    assert!(s.contains("llm_call"), "llm_call tag missing");
    assert!(s.contains("sweep"), "sweep tag missing");
    // Header rendered.
    assert!(s.contains("time"), "table header time missing");
    assert!(s.contains("kind"), "table header kind missing");
    // Auto-scroll marker visible in title.
    assert!(s.contains("▼"), "auto-scroll marker missing");
    // Duration formatting: 1100 → 1.1s, 187 → 187ms
    assert!(
        s.contains("1.1s"),
        "duration humanization missing (seconds)"
    );
    assert!(s.contains("187ms"), "duration humanization missing (ms)");
}

#[test]
fn event_stream_empty_state() {
    let events: Vec<Event> = Vec::new();
    let mut state = observability::EventStreamState::new();

    let backend = TestBackend::new(120, 10);
    let mut terminal = Terminal::new(backend).expect("init test terminal");

    terminal
        .draw(|frame| {
            let area = frame.area();
            let [_, body, _] = common::chrome_layout(area);
            observability::render(frame, body, &events, &mut state, None, None);
        })
        .expect("draw empty events");

    let s = buffer_to_string(&terminal);
    assert!(s.contains("no events yet"), "empty-state copy missing");
}

#[test]
fn event_stream_load_error() {
    let events: Vec<Event> = Vec::new();
    let mut state = observability::EventStreamState::new();

    let backend = TestBackend::new(120, 10);
    let mut terminal = Terminal::new(backend).expect("init test terminal");

    terminal
        .draw(|frame| {
            let area = frame.area();
            let [_, body, _] = common::chrome_layout(area);
            observability::render(frame, body, &events, &mut state, None, Some("ECONNREFUSED"));
        })
        .expect("draw error");

    let s = buffer_to_string(&terminal);
    assert!(s.contains("ECONNREFUSED"), "error text missing");
}

#[test]
fn event_state_auto_scroll_lifecycle() {
    let mut state = observability::EventStreamState::new();
    assert!(state.auto_scroll, "should start with auto-scroll on");
    assert_eq!(state.table.selected(), Some(0));

    // Manual selection turns auto-scroll off.
    state.select_next(5);
    assert!(!state.auto_scroll, "manual nav should disable auto-scroll");
    assert_eq!(state.table.selected(), Some(1));

    // Maintaining auto-scroll when off does nothing.
    state.maintain_auto_scroll(5);
    assert_eq!(state.table.selected(), Some(1));

    // jump_top re-enables auto-scroll.
    state.jump_top(5);
    assert!(state.auto_scroll, "jump_top should re-enable auto-scroll");
    assert_eq!(state.table.selected(), Some(0));

    // Now maintain_auto_scroll keeps us pinned to row 0 across refreshes.
    state.table.select(Some(3)); // simulate someone else messed with it
    state.maintain_auto_scroll(5);
    assert_eq!(
        state.table.selected(),
        Some(0),
        "auto-scroll should pin to row 0"
    );
}
