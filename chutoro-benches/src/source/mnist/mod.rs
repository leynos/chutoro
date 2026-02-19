//! MNIST download-and-cache helper for benchmark baselines.

use crate::source::{SyntheticError, numeric::SyntheticSource};
use flate2::read::GzDecoder;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const TRAIN_IMAGES_FILE: &str = "train-images-idx3-ubyte.gz";
const TEST_IMAGES_FILE: &str = "t10k-images-idx3-ubyte.gz";
const IDX_IMAGE_MAGIC: u32 = 2_051;
/// Number of MNIST images expected from train + test files.
pub const MNIST_POINT_COUNT: usize = 70_000;
/// Number of features (28x28) expected per MNIST image.
pub const MNIST_DIMENSIONS: usize = 784;

/// Configuration for MNIST download and cache behaviour.
#[derive(Clone, Debug)]
pub struct MnistConfig {
    /// Local directory where compressed MNIST files are cached.
    pub cache_dir: PathBuf,
    /// Base URL that hosts the MNIST gzip IDX files.
    pub base_url: String,
}

impl Default for MnistConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            base_url: "https://storage.googleapis.com/cvdf-datasets/mnist".to_owned(),
        }
    }
}

/// Download client abstraction for MNIST helpers.
pub trait MnistDownloadClient {
    /// Downloads URL contents as bytes.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] if the request fails.
    fn download_bytes(&self, url: &str) -> Result<Vec<u8>, SyntheticError>;
}

struct UreqMnistDownloadClient;

impl MnistDownloadClient for UreqMnistDownloadClient {
    fn download_bytes(&self, url: &str) -> Result<Vec<u8>, SyntheticError> {
        let mut response = ureq::get(url)
            .call()
            .map_err(|error| SyntheticError::Download {
                url: url.to_owned(),
                message: error.to_string(),
            })?;

        response
            .body_mut()
            .read_to_vec()
            .map_err(|error| SyntheticError::Download {
                url: url.to_owned(),
                message: error.to_string(),
            })
    }
}

impl SyntheticSource {
    /// Loads MNIST vectors (70,000 x 784) using a download-and-cache helper.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when downloading, parsing, or validating
    /// cached MNIST files fails.
    pub fn load_mnist(config: &MnistConfig) -> Result<Self, SyntheticError> {
        load_mnist_with_client(config, &UreqMnistDownloadClient)
    }
}

fn load_mnist_with_client(
    config: &MnistConfig,
    client: &dyn MnistDownloadClient,
) -> Result<SyntheticSource, SyntheticError> {
    fs::create_dir_all(&config.cache_dir)?;

    let train_path = config.cache_dir.join(TRAIN_IMAGES_FILE);
    let test_path = config.cache_dir.join(TEST_IMAGES_FILE);

    let train_bytes =
        ensure_cached_bytes(&train_path, &file_url(config, TRAIN_IMAGES_FILE), client)?;
    let test_bytes = ensure_cached_bytes(&test_path, &file_url(config, TEST_IMAGES_FILE), client)?;

    let train = parse_idx_images(&train_path, &train_bytes)?;
    let test = parse_idx_images(&test_path, &test_bytes)?;

    if train.dimensions != test.dimensions {
        return Err(SyntheticError::MnistDimensionMismatch {
            train: train.dimensions,
            test: test.dimensions,
        });
    }

    let point_count = train
        .count
        .checked_add(test.count)
        .ok_or(SyntheticError::Overflow)?;
    let dimensions = train.dimensions;

    if point_count != MNIST_POINT_COUNT || dimensions != MNIST_DIMENSIONS {
        return Err(SyntheticError::InvalidMnistFile {
            path: config.cache_dir.clone(),
            message: format!(
                "expected {MNIST_POINT_COUNT}x{MNIST_DIMENSIONS}, got {point_count}x{dimensions}"
            ),
        });
    }

    let mut data = train.data;
    data.extend(test.data);

    SyntheticSource::from_parts("mnist", data, point_count, dimensions)
}

fn ensure_cached_bytes(
    path: &Path,
    url: &str,
    client: &dyn MnistDownloadClient,
) -> Result<Vec<u8>, SyntheticError> {
    if path.exists() {
        return fs::read(path).map_err(SyntheticError::from);
    }

    let payload = client.download_bytes(url)?;
    write_atomic(path, &payload)?;
    Ok(payload)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), SyntheticError> {
    let mut part_path = path.to_path_buf();
    part_path.set_extension("part");
    if part_path.exists() {
        fs::remove_file(&part_path)?;
    }
    fs::write(&part_path, bytes)?;
    fs::rename(&part_path, path)?;
    Ok(())
}

fn file_url(config: &MnistConfig, file_name: &str) -> String {
    format!("{}/{}", config.base_url.trim_end_matches('/'), file_name)
}

fn default_cache_dir() -> PathBuf {
    if let Some(explicit) = env::var_os("CHUTORO_MNIST_CACHE_DIR") {
        return PathBuf::from(explicit);
    }

    if let Some(xdg_cache) = env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg_cache).join("chutoro").join("mnist");
    }

    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("chutoro")
            .join("mnist");
    }

    env::temp_dir().join("chutoro").join("mnist")
}

#[derive(Debug)]
struct ParsedImages {
    data: Vec<f32>,
    count: usize,
    dimensions: usize,
}

fn parse_idx_images(path: &Path, gzipped_bytes: &[u8]) -> Result<ParsedImages, SyntheticError> {
    let decoded = gunzip_bytes(path, gzipped_bytes)?;
    if decoded.len() < 16 {
        return Err(invalid_mnist(path, "header is shorter than 16 bytes"));
    }

    let magic = read_u32_be(slice_at(&decoded, 0, 4, path)?, path, "magic")?;
    if magic != IDX_IMAGE_MAGIC {
        return Err(invalid_mnist(
            path,
            &format!("unexpected IDX magic {magic}, expected {IDX_IMAGE_MAGIC}"),
        ));
    }

    let count = usize::try_from(read_u32_be(slice_at(&decoded, 4, 8, path)?, path, "count")?)
        .map_err(|_| invalid_mnist(path, "count does not fit usize"))?;
    let rows = usize::try_from(read_u32_be(slice_at(&decoded, 8, 12, path)?, path, "rows")?)
        .map_err(|_| invalid_mnist(path, "row count does not fit usize"))?;
    let cols = usize::try_from(read_u32_be(
        slice_at(&decoded, 12, 16, path)?,
        path,
        "cols",
    )?)
    .map_err(|_| invalid_mnist(path, "column count does not fit usize"))?;
    let dimensions = rows.checked_mul(cols).ok_or(SyntheticError::Overflow)?;
    let payload_len = count
        .checked_mul(dimensions)
        .ok_or(SyntheticError::Overflow)?;

    let payload = decoded
        .get(16..)
        .ok_or_else(|| invalid_mnist(path, "missing payload bytes"))?;
    if payload.len() != payload_len {
        return Err(invalid_mnist(
            path,
            &format!(
                "payload length mismatch: expected {payload_len}, got {}",
                payload.len()
            ),
        ));
    }

    let data = payload.iter().map(|value| f32::from(*value)).collect();
    Ok(ParsedImages {
        data,
        count,
        dimensions,
    })
}

fn gunzip_bytes(path: &Path, bytes: &[u8]) -> Result<Vec<u8>, SyntheticError> {
    let mut gzip_decoder = GzDecoder::new(bytes);
    let mut decompressed = Vec::new();
    gzip_decoder
        .read_to_end(&mut decompressed)
        .map_err(|error| SyntheticError::InvalidMnistFile {
            path: path.to_path_buf(),
            message: format!("gzip decode failure: {error}"),
        })?;
    Ok(decompressed)
}

fn read_u32_be(slice: &[u8], path: &Path, field: &str) -> Result<u32, SyntheticError> {
    if slice.len() != 4 {
        return Err(invalid_mnist(
            path,
            &format!("{field} field has invalid length {}", slice.len()),
        ));
    }

    let value = slice
        .iter()
        .fold(0_u32, |acc, byte| (acc << 8) | u32::from(*byte));
    Ok(value)
}

fn slice_at<'a>(
    data: &'a [u8],
    start: usize,
    end: usize,
    path: &Path,
) -> Result<&'a [u8], SyntheticError> {
    data.get(start..end)
        .ok_or_else(|| invalid_mnist(path, &format!("missing bytes for range {start}..{end}")))
}

fn invalid_mnist(path: &Path, message: &str) -> SyntheticError {
    SyntheticError::InvalidMnistFile {
        path: path.to_path_buf(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests;
