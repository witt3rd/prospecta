//! Headless render tests for the dashboard view.

use prospecta_tui::db::{DashboardStats, LlmCallStat};
use prospecta_tui::views::{common, dashboard};
use ratatui::{backend::TestBackend, Terminal};

fn dummy_stats() -> DashboardStats {
    DashboardStats {
        bank_id: "default".into(),
        embedding_dim: 32,
        documents: 12,
        memory_items: 47,
        retains_24h: 9,
        recalls_24h: 4,
        mean_recall_ms_24h: Some(187.5),
        mean_retain_ms_24h: Some(980.0),
        formulate_fallback_rate_24h: Some(0.15),
        formulates_24h: 20,
        llm_calls: vec![
            LlmCallStat {
                prompt_name: "formulate".into(),
                calls: 20,
                avg_ms: 412.0,
                max_ms: 890,
                errors: 1,
            },
            LlmCallStat {
                prompt_name: "synthesize".into(),
                calls: 4,
                avg_ms: 1100.0,
                max_ms: 2300,
                errors: 0,
            },
        ],
        total_llm_ms_24h: 13_240,
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
fn dashboard_empty_state() {
    let mut state = dashboard::DashboardState::new();
    let backend = TestBackend::new(140, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            dashboard::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);
    assert!(s.contains("pick a bank"), "empty hint missing");
}

#[test]
fn dashboard_load_error() {
    let mut state = dashboard::DashboardState::new();
    state.error = Some("ECONNREFUSED".into());
    let backend = TestBackend::new(140, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            dashboard::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);
    assert!(s.contains("ECONNREFUSED"));
}

#[test]
fn dashboard_full_render_paints_all_cards() {
    let mut state = dashboard::DashboardState::new();
    state.bank_id = Some("default".into());
    state.stats = Some(dummy_stats());

    let backend = TestBackend::new(160, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            dashboard::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);

    // Cards
    assert!(s.contains("substrate"), "substrate card missing");
    assert!(s.contains("activity"), "activity card missing");
    assert!(s.contains("latency"), "latency card missing");
    assert!(s.contains("health"), "health card missing");
    // Values
    assert!(s.contains("default"), "bank_id missing");
    assert!(s.contains("12"), "doc count missing");
    assert!(s.contains("47"), "items count missing");
    assert!(
        s.contains("188ms") || s.contains("187ms") || s.contains("187"),
        "mean recall missing in buffer"
    );
    assert!(s.contains("15.0%"), "parse_fallback rate missing");
    // LLM table
    assert!(s.contains("formulate"), "formulate prompt missing");
    assert!(s.contains("synthesize"), "synthesize prompt missing");
    assert!(s.contains("prompt"), "llm header missing");
    assert!(s.contains("max"), "llm header missing");
}

#[test]
fn dashboard_no_activity_shows_dashes() {
    let mut state = dashboard::DashboardState::new();
    state.bank_id = Some("scratch".into());
    let mut stats = dummy_stats();
    stats.bank_id = "scratch".into();
    stats.retains_24h = 0;
    stats.recalls_24h = 0;
    stats.formulates_24h = 0;
    stats.mean_recall_ms_24h = None;
    stats.mean_retain_ms_24h = None;
    stats.formulate_fallback_rate_24h = None;
    stats.llm_calls = vec![];
    stats.total_llm_ms_24h = 0;
    state.stats = Some(stats);

    let backend = TestBackend::new(160, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            let area = f.area();
            let [_, body, _] = common::chrome_layout(area);
            dashboard::render(f, body, &mut state);
        })
        .unwrap();
    let s = buf_to_string(&terminal);
    assert!(s.contains("no formulates"), "no-formulates hint missing");
    assert!(
        s.contains("no llm_calls in the last 24 hours"),
        "empty llm-table hint missing"
    );
    assert!(s.contains("scratch"), "bank id missing in render");
}
