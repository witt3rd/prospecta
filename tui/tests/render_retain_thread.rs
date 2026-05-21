//! Headless render tests for the retain-thread view.

use chrono::{Duration, Utc};
use prospecta_tui::db::{DocumentRow, IndexTextCall, ItemRow, RetainRow, RetainThread};
use prospecta_tui::views::{common, retain_thread};
use ratatui::{backend::TestBackend, Terminal};
use uuid::Uuid;

fn dummy_thread() -> RetainThread {
    let now = Utc::now();
    let doc_id = Uuid::parse_str("b6f6d2a6-c665-4a55-937e-c37e0e01ce18").unwrap();
    RetainThread {
        retain: RetainRow {
            id: 7,
            bank_id: "default".into(),
            created_at: now,
            document_id: Some(doc_id),
            items_count: 3,
            index_text_caller_supplied: false,
            index_text_generated: Some(vec![
                "What is prospecta?".into(),
                "How does bilateral synthesis work?".into(),
                "Why anticipated questions?".into(),
            ]),
            duration_ms: 980,
            raw_llm_response: None,
            error: None,
        },
        document: Some(DocumentRow {
            id: doc_id,
            source: Some("~/forge/notes/handoff.md".into()),
            content_hash: "handoff_hash_001".into(),
            tags: vec!["handoff".into(), "tui".into()],
            created_at: now - Duration::seconds(1),
            original_text: "Prospecta is the bilateral memory library.".into(),
        }),
        items: vec![
            ItemRow {
                id: Uuid::parse_str("11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa").unwrap(),
                content: "What is prospecta?".into(),
                llm_generated: true,
                tags: vec![],
            },
            ItemRow {
                id: Uuid::parse_str("22222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb").unwrap(),
                content: "How does bilateral synthesis work?".into(),
                llm_generated: true,
                tags: vec!["q".into()],
            },
            ItemRow {
                id: Uuid::parse_str("33333333-cccc-cccc-cccc-cccccccccccc").unwrap(),
                content: "Why anticipated questions?".into(),
                llm_generated: true,
                tags: vec![],
            },
        ],
        index_text_call: Some(IndexTextCall {
            id: 3,
            created_at: now - Duration::milliseconds(200),
            duration_ms: 290,
            json_mode: true,
            error: None,
            prompt_text: Some("<system>generate index_text...</system>".into()),
            response_text: Some(
                r#"{"questions":["What is prospecta?","How does bilateral synthesis work?","Why anticipated questions?"]}"#
                    .into(),
            ),
        }),
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
fn retain_thread_empty_state() {
    let mut state = retain_thread::RetainThreadState::new();
    let backend = TestBackend::new(140, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            retain_thread::render(f, body, &mut state);
        })
        .unwrap();

    let s = buf_to_string(&terminal);
    assert!(
        s.contains("press Enter on a retain row"),
        "empty hint missing"
    );
}

#[test]
fn retain_thread_load_error() {
    let mut state = retain_thread::RetainThreadState::new();
    state.error = Some("ECONNREFUSED".into());
    let backend = TestBackend::new(140, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            retain_thread::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);
    assert!(s.contains("ECONNREFUSED"));
}

#[test]
fn retain_thread_full_render() {
    let mut state = retain_thread::RetainThreadState::new();
    state.thread = Some(dummy_thread());
    let backend = TestBackend::new(160, 60);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            retain_thread::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);

    // Section headers
    assert!(s.contains("retain"), "retain header missing");
    assert!(s.contains("document"), "document header missing");
    assert!(s.contains("memory_items"), "items header missing");
    assert!(
        s.contains("index_text llm_call"),
        "index_text header missing"
    );

    // Retain metadata
    assert!(s.contains("items_count"), "items_count field missing");
    assert!(
        s.contains("index_text_caller_supplied"),
        "caller_supplied field missing"
    );
    assert!(s.contains("980ms"), "duration missing");

    // Document section
    assert!(s.contains("~/forge/notes/handoff.md"), "doc source missing");
    assert!(s.contains("b6f6d2a6"), "doc short uuid missing");
    assert!(s.contains("handoff"), "doc tag missing");
    assert!(
        s.contains("Prospecta is the bilateral memory library."),
        "doc body missing"
    );

    // Items section
    assert!(s.contains("What is prospecta?"), "item 1 missing");
    assert!(s.contains("Why anticipated questions?"), "item 3 missing");
    assert!(s.contains("llm"), "llm marker missing");

    // Index text call
    assert!(s.contains("290ms"), "index_text call duration missing");
    assert!(s.contains("prompt:"), "prompt label missing");
    assert!(s.contains("response:"), "response label missing");
    assert!(
        s.contains("index_text_generated"),
        "verbatim list label missing"
    );
}

#[test]
fn retain_thread_caller_supplied_skips_llm_section() {
    let mut state = retain_thread::RetainThreadState::new();
    let mut thread = dummy_thread();
    thread.retain.index_text_caller_supplied = true;
    thread.retain.index_text_generated = None;
    thread.index_text_call = None;
    state.thread = Some(thread);

    let backend = TestBackend::new(160, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            retain_thread::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);
    assert!(s.contains("skipped"), "caller-supplied skip hint missing");
}

#[test]
fn retain_thread_missing_document_hint() {
    let mut state = retain_thread::RetainThreadState::new();
    let mut thread = dummy_thread();
    thread.document = None;
    thread.retain.document_id = None;
    thread.items.clear();
    state.thread = Some(thread);

    let backend = TestBackend::new(160, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            retain_thread::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);
    assert!(
        s.contains("no document linked"),
        "missing-doc hint not painted"
    );
}

#[test]
fn retain_thread_scroll_state() {
    let mut state = retain_thread::RetainThreadState::new();
    assert_eq!(state.scroll, 0);
    state.scroll_down();
    state.scroll_down();
    assert_eq!(state.scroll, 4);
    state.scroll_up();
    assert_eq!(state.scroll, 2);
    state.reset_scroll();
    assert_eq!(state.scroll, 0);
}
