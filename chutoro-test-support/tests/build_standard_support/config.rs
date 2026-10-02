//! Readers for the Cargo configuration half of the build standard: the
//! toolchain pin, the flag lists, and the `rustflags` sources in
//! `.cargo/config.toml`. Everything is read as text, so the contract needs no
//! parser dependency.

pub const CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../.cargo/config.toml"
));
pub const TOOLCHAIN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rust-toolchain.toml"
));

/// The parallel-frontend flag every `rustflags` source carries on a nightly
/// pin.
pub const THREADS_FLAG: &str = "-Zthreads=8";
/// The linker flag the Linux source adds, normalized to one token.
pub const LINKER_FLAG: &str = "-Clink-arg=-fuse-ld=mold";

/// A list of complaints about the repository.
pub type Problems = Vec<String>;

/// The channel the toolchain file pins.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pin {
    Nightly,
    Stable,
}

impl Pin {
    /// Reads the pin from a `rust-toolchain.toml`.
    ///
    /// The channel must be named exactly once and be one the standard knows: a
    /// `nightly` (dated or not), `stable`, `beta`, or a numbered release.
    /// Anything else, and a missing or repeated `channel`, is an error rather
    /// than a guess that lets a malformed file pass as stable.
    ///
    /// ```text
    /// Pin::read("channel = \"nightly-2026-05-28\"") == Ok(Pin::Nightly)
    /// Pin::read("channel = \"1.94.0\"")             == Ok(Pin::Stable)
    /// Pin::read("[toolchain]")                       == Err(..)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the reason when the channel is missing, repeated or unsupported.
    pub fn read(toolchain: &str) -> Result<Self, String> {
        let channels = toolchain_channels(toolchain);
        match channels.as_slice() {
            [] => Err("rust-toolchain.toml names no channel".to_owned()),
            [channel] => Self::classify(channel),
            _ => Err(format!(
                "rust-toolchain.toml names more than one channel: {channels:?}"
            )),
        }
    }

    /// Classifies one channel name.
    fn classify(channel: &str) -> Result<Self, String> {
        let is_nightly = channel == "nightly" || channel.starts_with("nightly-");
        let is_release = channel.split('.').count() >= 2
            && channel
                .split('.')
                .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
        if is_nightly {
            Ok(Self::Nightly)
        } else if is_release || matches!(channel, "stable" | "beta") {
            Ok(Self::Stable)
        } else {
            Err(format!(
                "the channel `{channel}` is not one the standard knows"
            ))
        }
    }

    /// Returns whether the pin takes `-Zthreads`, which is a nightly flag.
    pub const fn takes_threads(self) -> bool {
        matches!(self, Self::Nightly)
    }
}

/// Returns the quoted value of a `channel = "..."` line, if the line is one.
fn channel_value(line: &str) -> Option<&str> {
    let (key, value) = line.split_once('=')?;
    if key.trim() != "channel" {
        return None;
    }
    value.trim().strip_prefix('"')?.split('"').next()
}

/// Returns every `channel` value under `[toolchain]`, skipping comments, so a
/// lookalike key, a commented line or a key in another table is not counted.
fn toolchain_channels(toolchain: &str) -> Vec<&str> {
    let mut table = "";
    let mut found = Vec::new();
    for line in toolchain
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
    {
        if line.starts_with('[') {
            table = line.trim_matches(|c| c == '[' || c == ']').trim();
        } else if table == "toolchain" {
            found.extend(channel_value(line));
        }
    }
    found
}

/// A list of compiler flags, with `-C value` pairs joined into `-Cvalue` so
/// both spellings compare equal.
#[derive(Debug, PartialEq, Eq)]
pub struct Flags(Vec<String>);

impl Flags {
    /// Reads a flag list from its words.
    ///
    /// ```text
    /// Flags::from_words(["-C", "link-arg=-fuse-ld=mold"]) == Flags::from_words(["-Clink-arg=-fuse-ld=mold"])
    /// ```
    pub fn from_words<'a>(words: impl IntoIterator<Item = &'a str>) -> Self {
        let mut joined: Vec<String> = Vec::new();
        for word in words {
            match joined.last_mut() {
                Some(last) if last == "-C" => *last = format!("-C{word}"),
                _ => joined.push(word.to_owned()),
            }
        }
        Self(joined)
    }

    /// Returns whether the list names one flag.
    fn names(&self, flag: &str) -> bool {
        self.0.iter().any(|candidate| candidate == flag)
    }

    /// Returns whether the list denies warnings, in either spelling.
    pub fn denies_warnings(&self) -> bool {
        self.0.iter().any(|word| word == "-Dwarnings")
            || self.0.windows(2).any(|pair| pair == ["-D", "warnings"])
    }

    /// Returns whether the list names the frontend flag.
    pub fn names_threads(&self) -> bool {
        self.names(THREADS_FLAG)
    }

    /// Returns whether the list names the linker flag.
    pub fn names_linker(&self) -> bool {
        self.names(LINKER_FLAG)
    }

    /// Returns the list without the linker flag, which is the one that may
    /// differ.
    fn without_linker_flag(&self) -> Vec<&String> {
        self.0.iter().filter(|flag| *flag != LINKER_FLAG).collect()
    }

    /// Checks the list against a pin and whether the linker flag is expected.
    ///
    /// ```text
    /// Flags::from_words(["-Zthreads=8"]).meets(Pin::Nightly, false) == Ok(())
    /// Flags::from_words([]).meets(Pin::Nightly, false).is_err()
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the reason when the frontend or linker flag is wrong.
    pub fn meets(&self, pin: Pin, takes_linker_flag: bool) -> Result<(), String> {
        if self.names_threads() != pin.takes_threads() {
            return Err(format!("gets {THREADS_FLAG} wrong: {:?}", self.0));
        }
        if self.names_linker() != takes_linker_flag {
            return Err(format!("gets mold wrong: {:?}", self.0));
        }
        Ok(())
    }
}

/// One `rustflags` source in a Cargo configuration.
struct Source {
    table: String,
    flags: Flags,
}

impl Source {
    /// Returns whether the table applies on Linux alone.
    fn is_linux(&self) -> bool {
        self.table.starts_with("target.") && self.table.contains("linux")
    }

    /// Returns whether the table selects every Linux target, not one triple.
    fn is_all_linux(&self) -> bool {
        self.table == ALL_LINUX_TABLE
    }

    /// Returns what is wrong with the source's flags for a pin: the frontend
    /// flag on a nightly pin only, and mold in a Linux table only.
    fn problem(&self, pin: Pin) -> Option<String> {
        let reason = self.flags.meets(pin, self.is_linux()).err()?;
        Some(format!("[{}] {reason}", self.table))
    }
}

/// The table name for `[target.'cfg(target_os = "linux")']`, which every Linux
/// target matches; a triple table covers one architecture alone.
const ALL_LINUX_TABLE: &str = "target.'cfg(target_os = \"linux\")'";

/// One line of a Cargo configuration, as far as the standard reads it.
enum Line {
    Table(String),
    Rustflags(Flags),
    Other,
}

/// Returns the quoted strings in one line, in order.
fn quoted(line: &str) -> Vec<&str> {
    line.split('"').skip(1).step_by(2).collect()
}

/// Returns a line up to a `#` that starts a comment, ignoring a `#` inside a
/// quoted string, and without trailing space.
fn without_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    for (index, c) in line.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(open), _) if c == open => quote = None,
            (None, '#') => return line.get(..index).unwrap_or(line).trim_end(),
            _ => {}
        }
    }
    line
}

/// Reads one configuration line. A `rustflags` entry is a one-line array of
/// strings, which is the shape the standard prescribes; an entry spread over
/// several lines is refused rather than half read.
///
/// ```text
/// read_line("[build]")                     -> Line::Table("build")
/// read_line("[build] # hosts")             -> Line::Table("build")
/// read_line("rustflags-extra = [\"x\"]")   -> Line::Other
/// read_line("rustflags = [\"-Zthreads=8\"]") -> Line::Rustflags(..)
/// ```
fn read_line(raw: &str) -> Result<Line, String> {
    let line = without_comment(raw);
    if line.starts_with('[') {
        return Ok(Line::Table(
            line.trim_matches(|c| c == '[' || c == ']')
                .trim()
                .to_owned(),
        ));
    }
    let Some(value) = line
        .strip_prefix("rustflags")
        .filter(|value| value.trim_start().starts_with('='))
    else {
        return Ok(Line::Other);
    };
    if !value.contains(']') {
        return Err("a `rustflags` array spans lines; keep it on one".to_owned());
    }
    Ok(Line::Rustflags(Flags::from_words(quoted(value))))
}

/// Returns every `rustflags` source in a Cargo configuration.
fn sources(config: &str) -> Result<Vec<Source>, String> {
    let mut table = String::new();
    let mut found = Vec::new();
    for line in config
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
    {
        match read_line(line)? {
            Line::Table(name) => table = name,
            Line::Rustflags(flags) => found.push(Source {
                table: table.clone(),
                flags,
            }),
            Line::Other => {}
        }
    }
    Ok(found)
}

/// Returns the complaints about which sources exist: there must be a Linux
/// table, and a nightly pin needs a `[build]` source for the other hosts.
fn shape_problems(found: &[Source], pin: Pin) -> Problems {
    let checks = [
        (found.is_empty(), "no rustflags source"),
        (
            !found.iter().any(Source::is_linux),
            "no Linux target table carries rustflags",
        ),
        (
            found.iter().any(Source::is_linux) && !found.iter().any(Source::is_all_linux),
            "no `cfg(target_os = \"linux\")` table carries rustflags, so mold reaches one Linux architecture only",
        ),
        (
            pin.takes_threads() && !found.iter().any(|source| source.table == "build"),
            "no [build] rustflags for non-Linux hosts",
        ),
    ];
    checks
        .into_iter()
        .filter(|(failed, _)| *failed)
        .map(|(_, text)| text.to_owned())
        .collect()
}

/// Returns a complaint when the sources differ in anything but the linker,
/// since Cargo applies one source rather than merging them.
fn drift_problem(found: &[Source]) -> Option<String> {
    let mut stripped: Vec<Vec<&String>> = found
        .iter()
        .map(|source| source.flags.without_linker_flag())
        .collect();
    stripped.dedup();
    (stripped.len() > 1).then(|| format!("sources differ beyond the linker: {stripped:?}"))
}

/// Returns every complaint about the configuration sources.
///
/// ```text
/// config_problems(CONFIG, Pin::read(TOOLCHAIN)) == Ok(vec![])   // a compliant repository
/// ```
///
/// # Errors
///
/// Returns the reason when the configuration cannot be read.
pub fn config_problems(config: &str, pin: Pin) -> Result<Problems, String> {
    let found = sources(config)?;
    let mut problems = shape_problems(&found, pin);
    problems.extend(found.iter().filter_map(|source| source.problem(pin)));
    problems.extend(drift_problem(&found));
    Ok(problems)
}
