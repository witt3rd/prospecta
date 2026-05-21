//! Headless render tests for the recall-thread view.

use chrono::{Duration, Utc};
use prospecta_tui::db::{FormulateRow, LlmCallRow, RecallRow, RecallThread};
use prospecta_tui::views::{common, recall_thread};
use ratatui::{backend::TestBackend, Terminal};
use serde_json::json;

fn dummy_thread() -> RecallThread {
    let now = Utc::now();
    RecallThread {
        recall: RecallRow {
            id: 17,
            bank_id: "default".into(),
            created_at: now,
            queries: json!(["when is kelly's birthday?", "kelly birthday date"]),
            mode: "hybrid".into(),
            n_results: 5,
            duration_ms: 187,
            synthesis: Some("Kelly's birthday is March 14, per ~/forge/notes/kelly.md.".into()),
            results: Some(json!([
                {
                    "source": "~/forge/notes/kelly.md",
                    "document_id": "abcdef12-3456-7890-abcd-ef1234567890",
                    "rank": 1,
                    "scores": {"semantic": 0.81, "lexical_content": 0.42, "lexical_body": 0.30, "rrf": 0.92},
                    "content_preview": "Kelly's birthday is March 14..."
                },
                {
                    "source": "design/calendar",
                    "document_id": "deadbeef-1111-2222-3333-444455556666",
                    "rank": 2,
                    "scores": {"semantic": 0.72},
                    "content_preview": "Birthdays tracked in the calendar..."
                }
            ])),
        },
        formulate: Some(FormulateRow {
            id: 9,
            created_at: now - Duration::milliseconds(500),
            message: "when is kelly's birthday?".into(),
            n_queries_out: 2,
            json_mode_used: true,
            parse_fallback: false,
            raw_response: r#"{"queries":["when is kelly's birthday?","kelly birthday date"]}"#
                .into(),
            duration_ms: 412,
            error_kind: None,
        }),
        llm_calls: vec![
            LlmCallRow {
                id: 1,
                created_at: now - Duration::milliseconds(500),
                prompt_name: "formulate".into(),
                duration_ms: 412,
                json_mode: true,
                error: None,
                prompt_text: Some("<system>expand queries...</system>".into()),
                response_text: Some(r#"{"queries":[...]}"#.into()),
            },
            LlmCallRow {
                id: 2,
                created_at: now + Duration::milliseconds(100),
                prompt_name: "synthesize".into(),
                duration_ms: 1100,
                json_mode: false,
                error: None,
                prompt_text: Some("<system>synthesize from chunks</system>".into()),
                response_text: Some("Kelly's birthday is March 14.".into()),
            },
        ],
    }
}

fn buf_to_string(t: &Terminal<TestBackend>) -> String {
    let buf = t.backend().buffer().clone();
    let mut s = String::new();
    for cell in buf.content.iter() {
        s.push_str(cell.symbol());
    }
    s
}

#[test]
fn recall_thread_empty_state_when_no_thread_loaded() {
    let mut state = recall_thread::RecallThreadState::new();
    let backend = TestBackend::new(140, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            recall_thread::render(f, body, &mut state);
        })
        .unwrap();

    let s = buf_to_string(&terminal);
    assert!(
        s.contains("press Enter on a recall row"),
        "empty hint missing"
    );
}

#[test]
fn recall_thread_load_error_renders() {
    let mut state = recall_thread::RecallThreadState::new();
    state.error = Some("ECONNREFUSED".into());

    let backend = TestBackend::new(140, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            recall_thread::render(f, body, &mut state);
        })
        .unwrap();

    let s = buf_to_string(&terminal);
    assert!(s.contains("ECONNREFUSED"), "error not painted");
}

#[test]
fn recall_thread_full_render_paints_all_sections() {
    let mut state = recall_thread::RecallThreadState::new();
    state.thread = Some(dummy_thread());

    // Tall buffer so all sections fit without scrolling.
    let backend = TestBackend::new(160, 60);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            recall_thread::render(f, body, &mut state);
        })
        .unwrap();

    let s = buf_to_string(&terminal);
    // Section headers.
    assert!(s.contains("recall"), "recall section header missing");
    assert!(s.contains("formulate"), "formulate section header missing");
    assert!(
        s.contains("recall body"),
        "recall body section header missing"
    );
    assert!(s.contains("llm_calls"), "llm_calls section header missing");

    // Recall metadata + queries.
    assert!(s.contains("hybrid"), "mode missing");
    assert!(s.contains("when is kelly"), "user message missing");
    assert!(
        s.contains("1. when is kelly"),
        "first formulated query missing"
    );

    // Per-chunk results with rank, source, doc id, scores, preview.
    assert!(s.contains("[1]"), "result rank 1 missing");
    assert!(
        s.contains("~/forge/notes/kelly.md"),
        "result source missing"
    );
    assert!(s.contains("abcdef12"), "result short doc-id missing");
    assert!(s.contains("scores:"), "scores label missing");
    assert!(s.contains("rrf"), "rrf channel missing in scores");
    assert!(
        s.contains("Kelly's birthday is March 14"),
        "preview missing"
    );

    // Synthesis text.
    assert!(
        s.contains("Kelly's birthday is March 14"),
        "synthesis missing"
    );

    // llm_calls section content.
    assert!(s.contains("formulate"), "formulate llm_call missing");
    assert!(s.contains("synthesize"), "synthesize llm_call missing");
    assert!(s.contains("prompt:"), "prompt label missing");
    assert!(s.contains("response:"), "response label missing");
}

#[test]
fn recall_thread_handles_missing_formulate() {
    let mut state = recall_thread::RecallThreadState::new();
    let mut thread = dummy_thread();
    thread.formulate = None;
    state.thread = Some(thread);

    let backend = TestBackend::new(160, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            recall_thread::render(f, body, &mut state);
        })
        .unwrap();

    let s = buf_to_string(&terminal);
    assert!(
        s.contains("no formulate_event found"),
        "missing-formulate hint not painted"
    );
}

#[test]
fn recall_thread_handles_no_results_and_no_synthesis() {
    let mut state = recall_thread::RecallThreadState::new();
    let mut thread = dummy_thread();
    thread.recall.results = None;
    thread.recall.synthesis = None;
    state.thread = Some(thread);

    let backend = TestBackend::new(160, 50);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            recall_thread::render(f, body, &mut state);
        })
        .unwrap();

    let s = buf_to_string(&terminal);
    assert!(s.contains("no results JSONB"), "no-results hint missing");
    assert!(s.contains("no synthesis"), "no-synthesis hint missing");
}

#[test]
fn recall_thread_scroll_state() {
    let mut state = recall_thread::RecallThreadState::new();
    assert_eq!(state.scroll, 0);
    state.scroll_down();
    assert_eq!(state.scroll, 2);
    state.scroll_down();
    assert_eq!(state.scroll, 4);
    state.scroll_up();
    assert_eq!(state.scroll, 2);
    // Saturating at 0
    for _ in 0..10 {
        state.scroll_up();
    }
    assert_eq!(state.scroll, 0);
    state.scroll_down();
    state.reset_scroll();
    assert_eq!(state.scroll, 0);
}
