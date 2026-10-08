//! Reads the shell lines `make -n` prints: which commands a logical line runs, and which of them
//! compile or run code, so the contract can ask each one to assign `RUSTFLAGS`.

/// Cargo subcommands that inspect or install and compile nothing, so a recipe running
/// one needs no `RUSTFLAGS`. Every other Cargo subcommand, and Whitaker, compiles or runs
/// code, and a development or held-out recipe must assign the flags it takes there.
const INSPECTION_SUBCOMMANDS: &[&str] = &[
    "fmt", "metadata", "doc", "install", "binstall", "audit", "deny", "machete", "version", "help",
];

/// Returns whether a command runs a tool that compiles or runs code: Whitaker, or Cargo with a
/// subcommand outside the inspection list. A version probe (`--version`, `-V`) runs nothing.
///
/// ```text
/// compiles("cargo +nightly clippy --all-targets") -> true
/// compiles("whitaker --all")                      -> true
/// compiles("/home/u/.cargo/bin/cargo test")       -> true (Cargo exports its own path)
/// compiles("cargo nextest --version")             -> false
/// compiles("cargo fmt --all --check")             -> false
/// ```
pub fn compiles(command: &str) -> bool {
    let words: Vec<&str> = command.split_whitespace().collect();
    if words.iter().any(|word| matches!(*word, "--version" | "-V")) {
        return false;
    }
    let cargo_subcommand = words
        .iter()
        .position(|word| word.rsplit('/').next() == Some("cargo"))
        .and_then(|at| words.get(at + 1..))
        .and_then(|rest| {
            rest.iter()
                .find(|word| !word.starts_with('+') && !word.starts_with('-'))
        })
        .is_some_and(|sub| !INSPECTION_SUBCOMMANDS.contains(sub));
    cargo_subcommand
        || words
            .iter()
            .any(|word| word.rsplit('/').next() == Some("whitaker"))
}

/// Splits one logical line of `make -n` output into the shell commands it runs: at `;`, `&&`,
/// `||` and a newline, outside quotes, so a probe, a `cargo metadata` and the real command that
/// share a line are judged one by one.
///
/// ```text
/// shell_commands("a && b; c \"x;y\"") -> ["a ", " b", " c \"x;y\""]
/// ```
pub fn shell_commands(line: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (_, '\\') => {
                current.push(c);
                current.extend(chars.next());
            }
            (Some(open), _) => {
                current.push(c);
                if c == open {
                    quote = None;
                }
            }
            (None, '\'' | '"') => {
                current.push(c);
                quote = Some(c);
            }
            (None, ';') => commands.push(std::mem::take(&mut current)),
            (None, '&' | '|') if chars.peek() == Some(&c) => {
                chars.next();
                commands.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    commands.push(current);
    commands
}

/// Returns a command without the shell keywords that lead it: `then`, `else`, `do` and the like.
pub fn without_leading_keywords(command: &str) -> &str {
    let mut rest = command.trim();
    while let Some((word, after)) = rest.split_once(char::is_whitespace) {
        if matches!(
            word,
            "if" | "then" | "else" | "elif" | "do" | "while" | "until" | "!" | "{" | "("
        ) {
            rest = after.trim_start();
        } else {
            break;
        }
    }
    rest
}
