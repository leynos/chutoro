//! Synthetic text generators for Levenshtein-distance benchmarking.

use crate::source::SyntheticError;
use chutoro_core::{DataSource, DataSourceError, MetricDescriptor};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use strsim::levenshtein;

/// Configuration for synthetic text corpus generation.
#[derive(Clone, Debug)]
pub struct SyntheticTextConfig {
    /// Number of strings to generate.
    pub item_count: usize,
    /// Minimum generated string length.
    pub min_length: usize,
    /// Maximum generated string length.
    pub max_length: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Character alphabet used for generation and mutation.
    pub alphabet: String,
    /// Optional template words used as mutation anchors.
    pub template_words: Vec<String>,
    /// Maximum edit operations applied per generated string.
    pub max_edits_per_item: usize,
}

/// A synthetic text [`DataSource`] using Levenshtein distance.
#[derive(Clone, Debug)]
pub struct SyntheticTextSource {
    data: Vec<String>,
    name: &'static str,
}

impl SyntheticTextSource {
    /// Generates text strings from the supplied configuration.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when the configuration is invalid.
    pub fn generate(config: &SyntheticTextConfig) -> Result<Self, SyntheticError> {
        validate_text_config(config)?;
        let alphabet = alphabet_chars(&config.alphabet)?;
        let mut rng = SmallRng::seed_from_u64(config.seed);
        let templates = resolved_templates(config, &alphabet, &mut rng)?;

        let mut data = Vec::with_capacity(config.item_count);
        for _ in 0..config.item_count {
            let template_index = rng.gen_range(0..templates.len());
            let template =
                templates
                    .get(template_index)
                    .ok_or(SyntheticError::InvalidTemplateIndex {
                        index: template_index,
                        template_count: templates.len(),
                    })?;
            let mut current = template.chars().collect::<Vec<char>>();
            let edits = rng.gen_range(0..=config.max_edits_per_item);
            for _ in 0..edits {
                apply_edit(&mut current, &alphabet, &mut rng);
            }
            enforce_length_bounds(&mut current, config, &alphabet, &mut rng);
            data.push(current.into_iter().collect());
        }

        Ok(Self {
            data,
            name: "synthetic-text",
        })
    }

    /// Returns a read-only view of generated strings.
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.data
    }
}

impl DataSource for SyntheticTextSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        self.name
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("levenshtein")
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "Levenshtein distances are converted to f32 for DataSource"
    )]
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let lhs = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let rhs = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok(levenshtein(lhs, rhs) as f32)
    }
}

const fn validate_text_config(config: &SyntheticTextConfig) -> Result<(), SyntheticError> {
    if config.item_count == 0 {
        return Err(SyntheticError::ZeroTextItems);
    }
    if config.min_length == 0 {
        return Err(SyntheticError::ZeroTextLength);
    }
    if config.min_length > config.max_length {
        return Err(SyntheticError::InvalidTextLengthRange {
            min_length: config.min_length,
            max_length: config.max_length,
        });
    }
    if config.alphabet.is_empty() {
        return Err(SyntheticError::EmptyAlphabet);
    }
    Ok(())
}

fn alphabet_chars(alphabet: &str) -> Result<Vec<char>, SyntheticError> {
    let chars: Vec<char> = alphabet.chars().collect();
    if chars.is_empty() {
        return Err(SyntheticError::EmptyAlphabet);
    }
    Ok(chars)
}

fn resolved_templates(
    config: &SyntheticTextConfig,
    alphabet: &[char],
    rng: &mut SmallRng,
) -> Result<Vec<String>, SyntheticError> {
    if config.template_words.is_empty() {
        let count = config.item_count.min(8);
        return (0..count)
            .map(|_| random_word(config.min_length, config.max_length, alphabet, rng))
            .collect();
    }

    let templates = config
        .template_words
        .iter()
        .map(|template| {
            let mut chars: Vec<char> = template.chars().collect();
            enforce_length_bounds(&mut chars, config, alphabet, rng);
            chars.into_iter().collect::<String>()
        })
        .collect::<Vec<String>>();

    if templates.is_empty() {
        return Err(SyntheticError::ZeroTextItems);
    }
    Ok(templates)
}

fn random_word(
    min_length: usize,
    max_length: usize,
    alphabet: &[char],
    rng: &mut SmallRng,
) -> Result<String, SyntheticError> {
    let length = rng.gen_range(min_length..=max_length);
    let mut chars = Vec::with_capacity(length);
    for _ in 0..length {
        let character = random_alphabet_char(alphabet, rng).ok_or(SyntheticError::EmptyAlphabet)?;
        chars.push(character);
    }
    Ok(chars.into_iter().collect())
}

#[derive(Clone, Copy)]
enum EditOperation {
    Insert,
    Delete,
    Substitute,
}

fn apply_edit(chars: &mut Vec<char>, alphabet: &[char], rng: &mut SmallRng) {
    let operation = choose_edit_operation(rng);
    apply_selected_edit(chars, alphabet, rng, operation);
}

fn choose_edit_operation(rng: &mut SmallRng) -> EditOperation {
    let operation = rng.gen_range(0..3);
    match operation {
        0 => EditOperation::Insert,
        1 => EditOperation::Delete,
        _ => EditOperation::Substitute,
    }
}

fn apply_selected_edit(
    chars: &mut Vec<char>,
    alphabet: &[char],
    rng: &mut SmallRng,
    operation: EditOperation,
) {
    match operation {
        EditOperation::Insert => insert_char(chars, alphabet, rng),
        EditOperation::Delete => delete_char(chars, rng),
        EditOperation::Substitute => substitute_char(chars, alphabet, rng),
    }
}

fn insert_char(chars: &mut Vec<char>, alphabet: &[char], rng: &mut SmallRng) {
    let insert_index = rng.gen_range(0..=chars.len());
    if let Some(character) = random_alphabet_char(alphabet, rng) {
        chars.insert(insert_index, character);
    }
}

fn delete_char(chars: &mut Vec<char>, rng: &mut SmallRng) {
    if chars.is_empty() {
        return;
    }
    let delete_index = rng.gen_range(0..chars.len());
    chars.remove(delete_index);
}

fn substitute_char(chars: &mut Vec<char>, alphabet: &[char], rng: &mut SmallRng) {
    if chars.is_empty() {
        insert_char(chars, alphabet, rng);
        return;
    }
    let replace_index = rng.gen_range(0..chars.len());
    if let Some(character) = random_alphabet_char(alphabet, rng)
        && let Some(slot) = chars.get_mut(replace_index)
    {
        *slot = character;
    }
}

fn enforce_length_bounds(
    chars: &mut Vec<char>,
    config: &SyntheticTextConfig,
    alphabet: &[char],
    rng: &mut SmallRng,
) {
    while chars.len() < config.min_length {
        insert_char(chars, alphabet, rng);
    }
    while chars.len() > config.max_length {
        delete_char(chars, rng);
    }
}

fn random_alphabet_char(alphabet: &[char], rng: &mut SmallRng) -> Option<char> {
    if alphabet.is_empty() {
        return None;
    }
    let index = rng.gen_range(0..alphabet.len());
    alphabet.get(index).copied()
}
