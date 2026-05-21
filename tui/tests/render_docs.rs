//! Headless render tests for the docs (drill-down) view.

use chrono::Utc;
use prospecta_tui::db::{Document, MemoryItem};
use prospecta_tui::views::{common, docs};
use ratatui::{backend::TestBackend, Terminal};
use uuid::Uuid;

fn dummy_documents() -> Vec<Document> {
    let now = Utc::now();
    let id_a = Uuid::parse_str("aaaaaaaa-1111-1111-1111-111111111111").unwrap();
    let id_b = Uuid::parse_str("bbbbbbbb-2222-2222-2222-222222222222").unwrap();
    vec![
        Document {
            id: id_a,
            bank_id: "default".into(),
            source: Some("~/forge/notes/handoff.md".into()),
            content_hash: "h1".into(),
            tags: vec!["handoff".into(), "tui".into()],
            created_at: now,
            item_count: 6,
        },
        Document {
            id: id_b,
            bank_id: "default".into(),
            source: Some("forge/soul".into()),
            content_hash: "h2".into(),
            tags: vec!["identity".into()],
            created_at: now - chrono::Duration::seconds(60),
            item_count: 3,
        },
    ]
}

fn dummy_items(document_id: Uuid) -> Vec<MemoryItem> {
    let now = Utc::now();
    vec![
        MemoryItem {
            id: Uuid::parse_str("11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa").unwrap(),
            bank_id: "default".into(),
            document_id,
            content: "What is prospecta?".into(),
            original_chunk: "Prospecta is the bilateral memory library.".into(),
            llm_generated: true,
            tags: vec!["q".into()],
            created_at: now,
        },
        MemoryItem {
            id: Uuid::parse_str("22222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb").unwrap(),
            bank_id: "default".into(),
            document_id,
            content: "How does bilateral synthesis work?".into(),
            original_chunk: "Both writer and reader speak through anticipated questions.".into(),
            llm_generated: false,
            tags: vec![],
            created_at: now,
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
fn docs_no_bank_selected_shows_hint() {
    let mut state = docs::DocsState::new();
    let backend = TestBackend::new(120, 12);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            docs::render(f, body, &mut state);
        })
        .unwrap();

    let s = buffer_to_string(&terminal);
    assert!(s.contains("pick a bank"), "no-bank hint missing: {s:?}");
}

#[test]
fn docs_documents_list_renders() {
    let mut state = docs::DocsState::new();
    state.bank_id = Some("default".into());
    state.documents = dummy_documents();
    state.documents_table.select(Some(0));

    let backend = TestBackend::new(140, 15);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            docs::render(f, body, &mut state);
        })
        .unwrap();

    let s = buffer_to_string(&terminal);
    assert!(s.contains("documents"), "view title missing");
    assert!(s.contains("default"), "bank name missing in title");
    assert!(s.contains("handoff"), "first doc tag missing");
    assert!(s.contains("forge/soul"), "second doc source missing");
    // Short uuid prefix from each doc.
    assert!(s.contains("aaaaaaaa"), "first short-uuid missing");
    assert!(s.contains("bbbbbbbb"), "second short-uuid missing");
}

#[test]
fn docs_items_level_renders_after_descent() {
    let mut state = docs::DocsState::new();
    state.bank_id = Some("default".into());
    state.documents = dummy_documents();
    state.documents_table.select(Some(0));

    let doc_id = state.documents[0].id;
    state.level = docs::DocsLevel::Items;
    state.focused_document = Some(doc_id);
    state.items = dummy_items(doc_id);
    state.items_table.select(Some(0));

    let backend = TestBackend::new(140, 15);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            docs::render(f, body, &mut state);
        })
        .unwrap();

    let s = buffer_to_string(&terminal);
    assert!(s.contains("items"), "items title missing");
    assert!(
        s.contains("What is prospecta?"),
        "first item content missing"
    );
    assert!(
        s.contains("How does bilateral synthesis work?"),
        "second item content missing"
    );
    assert!(s.contains("llm"), "llm-generated marker missing");
    assert!(s.contains("caller"), "caller-supplied marker missing");
}

#[test]
fn docs_navigation_wraps_within_level() {
    let mut state = docs::DocsState::new();
    state.bank_id = Some("default".into());
    state.documents = dummy_documents();
    state.documents_table.select(Some(0));

    // Documents level: 2 items, wrap test.
    state.select_next();
    assert_eq!(state.documents_table.selected(), Some(1));
    state.select_next();
    assert_eq!(state.documents_table.selected(), Some(0));
    state.select_prev();
    assert_eq!(state.documents_table.selected(), Some(1));

    // Switch to items level — independent selection state.
    state.level = docs::DocsLevel::Items;
    state.items = dummy_items(state.documents[0].id);
    state.items_table.select(Some(0));
    state.select_next();
    assert_eq!(state.items_table.selected(), Some(1));
    // Documents selection untouched by item navigation.
    assert_eq!(state.documents_table.selected(), Some(1));
}

#[test]
fn docs_empty_bank_shows_hint() {
    let mut state = docs::DocsState::new();
    state.bank_id = Some("empty-bank".into());
    state.documents = Vec::new();

    let backend = TestBackend::new(120, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            docs::render(f, body, &mut state);
        })
        .unwrap();

    let s = buffer_to_string(&terminal);
    assert!(
        s.contains("no documents in this bank"),
        "empty-bank hint missing"
    );
}
