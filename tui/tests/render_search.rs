//! Headless render tests for the search view.

use chrono::Utc;
use prospecta_tui::db::SearchHit;
use prospecta_tui::views::search::{self, SearchMode, SearchState};
use ratatui::{backend::TestBackend, Terminal};
use uuid::Uuid;

fn hit(
    content: &str,
    content_hit: bool,
    body_hit: bool,
    content_rank: f32,
    body_rank: f32,
) -> SearchHit {
    SearchHit {
        item_id: Uuid::new_v4(),
        document_id: Uuid::new_v4(),
        source: Some("design/tui-v0.1".into()),
        content: content.into(),
        original_chunk: "the source body chunk this index_text indexes".into(),
        llm_generated: true,
        content_rank,
        body_rank,
        content_hit,
        body_hit,
        created_at: Utc::now(),
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

fn render(state: &mut SearchState) -> String {
    let backend = TestBackend::new(160, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            search::render(f, area, state);
        })
        .unwrap();
    buf_to_string(&terminal)
}

#[test]
fn search_prompt_before_first_query() {
    let mut state = SearchState::new();
    state.bank_id = Some("default".into());
    let s = render(&mut state);
    assert!(s.contains("search · default"), "input title missing bank");
    assert!(
        s.contains("type a query and press Enter"),
        "pre-search hint missing"
    );
}

#[test]
fn search_no_results_after_searching() {
    let mut state = SearchState::new();
    state.bank_id = Some("default".into());
    state.query = "zzzz".into();
    state.searched = true;
    let s = render(&mut state);
    assert!(
        s.contains("no matches in either lexical channel"),
        "empty-result hint missing"
    );
}

#[test]
fn search_renders_channel_labels_and_ranks() {
    let mut state = SearchState::new();
    state.bank_id = Some("default".into());
    state.query = "bilateral".into();
    state.searched = true;
    state.mode = SearchMode::Browsing;
    state.hits = vec![
        hit(
            "How does bilateral synthesis work?",
            true,
            true,
            0.099,
            0.099,
        ),
        hit("What problem does memory solve?", false, true, 0.0, 0.099),
        hit(
            "How does compile-time checking work?",
            true,
            false,
            0.461,
            0.0,
        ),
    ];
    state.table.select(Some(0));
    let s = render(&mut state);

    // Header + counts
    assert!(s.contains("3 hits"), "hit count missing");
    assert!(s.contains("channel"), "results header missing");
    assert!(s.contains("matched index_text"), "results header missing");
    // Channel labels for all three states
    assert!(s.contains("both"), "both-channel label missing");
    assert!(s.contains("body"), "body-channel label missing");
    assert!(s.contains("question"), "question-channel label missing");
    // A rank renders
    assert!(s.contains("0.461") || s.contains("0.099"), "rank missing");
    // The matched content text shows
    assert!(s.contains("bilateral synthesis"), "matched content missing");
}

#[test]
fn search_preview_shows_selected_item() {
    let mut state = SearchState::new();
    state.bank_id = Some("default".into());
    state.searched = true;
    state.mode = SearchMode::Browsing;
    state.hits = vec![hit(
        "How does bilateral synthesis work?",
        true,
        true,
        0.5,
        0.3,
    )];
    state.table.select(Some(0));
    let s = render(&mut state);

    // Preview pane labels + the body chunk
    assert!(s.contains("preview"), "preview pane title missing");
    assert!(s.contains("index_text"), "preview index_text label missing");
    assert!(
        s.contains("source body chunk"),
        "preview body content missing"
    );
    assert!(
        s.contains("question=0.5000") || s.contains("question=0.500"),
        "preview rank breakdown missing"
    );
}

#[test]
fn search_editing_mode_shows_cursor() {
    let mut state = SearchState::new();
    state.bank_id = Some("default".into());
    state.mode = SearchMode::Editing;
    state.query = "bilat".into();
    let s = render(&mut state);
    // The typed query renders in the input box.
    assert!(s.contains("bilat"), "query text missing from input box");
}

#[test]
fn search_state_navigation_clamps() {
    let mut state = SearchState::new();
    state.hits = vec![
        hit("a", true, false, 0.1, 0.0),
        hit("b", true, false, 0.1, 0.0),
    ];
    state.table.select(Some(0));
    // prev at top stays at 0
    state.select_prev();
    assert_eq!(state.table.selected(), Some(0));
    // next advances
    state.select_next();
    assert_eq!(state.table.selected(), Some(1));
    // next at bottom stays at last
    state.select_next();
    assert_eq!(state.table.selected(), Some(1));
}
