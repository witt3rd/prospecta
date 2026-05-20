//! Config resolution — DATABASE_URL precedence, password redaction for display.

use color_eyre::eyre::{eyre, Result};
use url::Url;

/// Redact the password component of a DATABASE_URL for display.
/// Mirrors the Python library's `redact_database_url` so the TUI and the CLI
/// agree on what a "shown" connection string looks like.
pub fn redact_database_url(s: &str) -> String {
    match Url::parse(s) {
        Ok(mut u) => {
            if u.password().is_some() {
                let _ = u.set_password(Some("***"));
            }
            u.to_string()
        }
        Err(_) => "<unparseable DATABASE_URL>".to_string(),
    }
}

/// Validate the URL parses and uses a postgres-family scheme.
pub fn validate(url: &str) -> Result<()> {
    let u = Url::parse(url).map_err(|e| eyre!("DATABASE_URL parse error: {e}"))?;
    let scheme = u.scheme();
    if !(scheme == "postgres" || scheme == "postgresql") {
        return Err(eyre!(
            "DATABASE_URL scheme must be postgres:// or postgresql://, got {scheme}://"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_password() {
        let s = "postgres://prospecta:secret@localhost:5432/prospecta";
        assert_eq!(
            redact_database_url(s),
            "postgres://prospecta:***@localhost:5432/prospecta"
        );
    }

    #[test]
    fn redact_handles_no_password() {
        let s = "postgres://prospecta@localhost:5432/prospecta";
        // No password component; should pass through unchanged (parsed).
        let out = redact_database_url(s);
        assert!(out.starts_with("postgres://prospecta@"));
    }

    #[test]
    fn redact_handles_garbage() {
        assert_eq!(
            redact_database_url("not a url"),
            "<unparseable DATABASE_URL>"
        );
    }

    #[test]
    fn validate_accepts_postgres_schemes() {
        assert!(validate("postgres://u:p@h/d").is_ok());
        assert!(validate("postgresql://u:p@h/d").is_ok());
    }

    #[test]
    fn validate_rejects_other_schemes() {
        assert!(validate("mysql://u:p@h/d").is_err());
    }
}
