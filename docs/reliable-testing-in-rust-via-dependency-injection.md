# 🛡️ Reliable Testing in Rust via Dependency Injection

Writing robust, reliable, and parallelizable tests requires an intentional
approach to handling external dependencies such as environment variables, the
filesystem, or the system clock. Functions that directly call `std::env::var` or
`SystemTime::now()` are difficult to test because they depend on global,
non-deterministic state.

This leads to several problems:

- **Flaky Tests:** A test might pass or fail depending on the environment it
  runs in.
- **Parallel Execution Conflicts:** Tests that modify the same global
  environment variable (`std::env::set_var`) will interfere with each other
  when run with `cargo test`.
- **State Corruption:** A test that panics can fail to clean up its changes to
  the environment, poisoning subsequent tests.

The solution is a classic software design pattern: **Dependency Injection
(DI)**. Instead of a function reaching out to global state, its dependencies
are provided as arguments. The workspace uses `mockable` 3.0.0 for this
pattern. Its `Env` and `Clock` traits model common system interactions while
keeping the production implementation at the application boundary.

______________________________________________________________________

## ✨ Mocking Environment Variables

### 1. Add `mockable`

The workspace declares the approved version once at its root. A crate that uses
environment access inherits that declaration:

```toml
[dependencies]
mockable = { workspace = true }
```

Tests that use `MockEnv` enable its `mock` feature in their development
dependency:

```toml
[dev-dependencies]
mockable = { workspace = true, features = ["mock"] }
```

### 2. The Untestable Code (Before)

Directly calling `std::env` makes it hard to test all logic paths.

```rust
pub fn get_api_key() -> Option<String> {
    std::env::var("API_KEY").ok().filter(|key| !key.is_empty())
}
```

### 3. Refactoring for Testability (After)

The function is refactored to accept a generic type that implements the
`mockable::Env` trait.

```rust
use mockable::Env;

pub fn get_api_key(env: &impl Env) -> Option<String> {
    env.string("API_KEY").filter(|key| !key.is_empty())
}
```

The function's core logic remains unchanged, but its dependency on the
environment is now explicit and injectable.

### 4. Writing Isolated Unit Tests

Tests use `MockEnv`, the `mockall`-backed implementation supplied by
`mockable`'s `mock` feature, to simulate environmental conditions without
touching the process environment. Configure expectations for the method that
the code under test calls; do not populate or mutate the real environment.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mockable::{Env, MockEnv};

    #[test]
    fn test_get_api_key_present() {
        let mut env = MockEnv::new();
        env.expect_string().returning(|key| {
            assert_eq!(key, "API_KEY");
            Some("secret123".to_owned())
        });
        assert_eq!(get_api_key(&env), Some("secret123".to_string()));
    }

    #[test]
    fn test_get_api_key_missing() {
        let mut env = MockEnv::new();
        env.expect_string().returning(|key| {
            assert_eq!(key, "API_KEY");
            None
        });
        assert_eq!(get_api_key(&env), None);
    }

    #[test]
    fn test_get_api_key_present_but_empty() {
        let mut env = MockEnv::new();
        env.expect_string().returning(|key| {
            assert_eq!(key, "API_KEY");
            Some(String::new())
        });
        assert_eq!(get_api_key(&env), None);
    }
}
```

These tests are fast, completely isolated from each other, and will never fail
due to external state.

### 5. Environment value semantics

`Env` exposes several retrieval methods. Choose the narrowest one that
preserves the value the caller needs:

- `raw(key)` returns `Result<String, VarError>`, preserving the distinction
  between an absent variable and a present value that is not valid Unicode.
- `string(key)` returns `Option<String>` and maps absent or non-Unicode values
  to `None`.
- `os_string(key)` returns `Option<OsString>`, preserving non-Unicode values.
- `path_buf(key)` returns `Option<PathBuf>`, preserving non-Unicode executable
  and directory paths.

In production code, inject `DefaultEnv` at the application boundary. It is the
`mockable` 3.0.0 implementation that delegates to `std::env`; functions that
contain configuration logic should receive `&impl Env` instead.

```rust
use mockable::DefaultEnv;

fn main() {
    let env = DefaultEnv;
    if let Some(api_key) = get_api_key(&env) {
        println!("API Key found!");
    } else {
        println!("API Key not configured.");
    }
}
```

______________________________________________________________________

## 🔩 Handling Other Non-Deterministic Dependencies

This dependency injection pattern also applies to other non-deterministic
dependencies such as the system clock. With `mockable` 3.0.0's `clock` feature,
the `Clock` trait exposes `local()` and `utc()` methods. Enable both the
`clock` and `mock` features when tests use `MockClock`.

### Untestable Code

```rust
use chrono::{DateTime, Duration, Utc};

fn is_cache_entry_stale(creation_time: DateTime<Utc>) -> bool {
    Utc::now() >= creation_time + Duration::seconds(300)
}
```

### Testable Refactor

```rust
use chrono::{DateTime, Duration, Utc};
use mockable::Clock;

fn is_cache_entry_stale(creation_time: DateTime<Utc>, clock: &impl Clock) -> bool {
    clock.utc() >= creation_time + Duration::seconds(300)
}
```

### Testing with `MockClock`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use mockable::MockClock;

    #[test]
    fn test_cache_is_not_stale() {
        let creation_time = Utc::now();
        let mut clock = MockClock::new();
        clock
            .expect_utc()
            .return_const(creation_time + Duration::seconds(100));
        assert!(!is_cache_entry_stale(creation_time, &clock));
    }

    #[test]
    fn test_cache_is_stale() {
        let creation_time = Utc::now();
        let mut clock = MockClock::new();
        clock
            .expect_utc()
            .return_const(creation_time + Duration::seconds(301));
        assert!(is_cache_entry_stale(creation_time, &clock));
    }
}
```

In production, pass `DefaultClock` at the application boundary.

______________________________________________________________________

## 📌 Key Takeaways

- **The Problem is Non-Determinism:** Directly accessing global state like
  `std::env` or `SystemTime::now` makes code hard to test.
- **The Solution is Dependency Injection:** Pass dependencies into functions as
  arguments.
- **Use** `mockable` **Traits:** Abstract dependencies behind traits such as
  `impl Env` or `impl Clock`.
- **`Mock*` for Tests:** Use `MockEnv` and `MockClock` with `mockall`
  expectations for isolated, deterministic control.
- **`Default*` for Production:** Pass `DefaultEnv` and `DefaultClock` at the
  application boundary; keep configuration and time logic injectable.
- **Never mutate the process environment in tests:** Use `MockEnv` for
  in-process tests. For subprocess tests, configure the child command's
  environment explicitly.
