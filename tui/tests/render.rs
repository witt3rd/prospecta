//! Headless rendering smoke test — proves the bank-list view paints something
//! sensible against a TestBackend, with no real terminal and no live DB.

use chrono::Utc;
use prospecta_tui::db::{Bank, BankSummary};
use prospecta_tui::views::{common, manage};
use ratatui::{backend::TestBackend, Terminal};

fn dummy_banks() -> Vec<BankSummary> {
    vec![
        BankSummary {
            bank: Bank {
                bank_id: "default".into(),
                embedding_dim: 32,
                mission: None,
                embedding_model_id: None,
                created_at: Utc::now(),
            },
            documents: 12,
            memory_items: 47,
            retain_events: 9,
            recall_events: 4,
        },
        BankSummary {
            bank: Bank {
                bank_id: "scratch".into(),
                embedding_dim: 64,
                mission: Some("scratchpad bank for TUI dev".into()),
                embedding_model_id: None,
                created_at: Utc::now(),
            },
            documents: 0,
            memory_items: 0,
            retain_events: 0,
            recall_events: 0,
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
fn bank_list_renders_to_test_backend() {
    let banks = dummy_banks();
    let mut state = manage::BankListState::new();

    let backend = TestBackend::new(120, 20);
    let mut terminal = Terminal::new(backend).expect("init test terminal");

    terminal
        .draw(|frame| {
            let area = frame.area();
            let [header, body, footer] = common::chrome_layout(area);
            common::render_header(
                frame,
                header,
                "banks",
                "postgres://prospecta:***@localhost:5432/prospecta",
            );
            manage::render(frame, body, &banks, &mut state, None);
            common::render_footer(frame, footer, &[("q", "quit")]);
        })
        .expect("draw frame");

    let s = buffer_to_string(&terminal);
    assert!(s.contains("prospecta-tui"), "header not painted");
    assert!(s.contains("banks"), "view title not painted");
    assert!(s.contains("default"), "default bank row missing");
    assert!(s.contains("scratch"), "scratch bank row missing");
    assert!(
        s.contains("scratchpad bank for TUI dev"),
        "mission column missing"
    );
    assert!(s.contains("quit"), "footer keybinding missing");
}

#[test]
fn empty_banks_render_safely() {
    let banks: Vec<BankSummary> = Vec::new();
    let mut state = manage::BankListState::new();

    let backend = TestBackend::new(120, 20);
    let mut terminal = Terminal::new(backend).expect("init test terminal");

    terminal
        .draw(|frame| {
            let area = frame.area();
            let [_, body, _] = common::chrome_layout(area);
            manage::render(frame, body, &banks, &mut state, None);
        })
        .expect("draw empty");

    let s = buffer_to_string(&terminal);
    assert!(s.contains("no banks yet"), "empty-state copy missing");
}

#[test]
fn load_error_renders_safely() {
    let banks: Vec<BankSummary> = Vec::new();
    let mut state = manage::BankListState::new();

    let backend = TestBackend::new(120, 20);
    let mut terminal = Terminal::new(backend).expect("init test terminal");

    terminal
        .draw(|frame| {
            let area = frame.area();
            let [_, body, _] = common::chrome_layout(area);
            manage::render(frame, body, &banks, &mut state, Some("connection refused"));
        })
        .expect("draw error");

    let s = buffer_to_string(&terminal);
    assert!(
        s.contains("connection refused"),
        "error message not painted"
    );
}

#[test]
fn bank_list_state_navigation_wraps() {
    let mut state = manage::BankListState::new();
    assert_eq!(state.table.selected(), Some(0));

    // 3-item list: next from 0 → 1 → 2 → 0 (wrap)
    state.select_next(3);
    assert_eq!(state.table.selected(), Some(1));
    state.select_next(3);
    assert_eq!(state.table.selected(), Some(2));
    state.select_next(3);
    assert_eq!(state.table.selected(), Some(0));

    // prev from 0 → 2 (wrap backwards)
    state.select_prev(3);
    assert_eq!(state.table.selected(), Some(2));

    // empty list clears selection
    state.select_next(0);
    assert_eq!(state.table.selected(), None);
}
