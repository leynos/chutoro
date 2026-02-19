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
        let response = ureq::get(url)
            .call()
            .map_err(|error| SyntheticError::Download {
                url: url.to_owned(),
                message: error.to_string(),
            })?;

        let mut reader = response.into_reader();
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .map_err(|error| SyntheticError::Download {
                url: url.to_owned(),
                message: error.to_string(),
            })?;
        Ok(buffer)
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
mod tests {
    use super::*;
    use chutoro_core::DataSource;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use rstest::rstest;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FakeClient {
        payloads: HashMap<String, Vec<u8>>,
        call_count: RefCell<usize>,
    }

    impl FakeClient {
        fn new(payloads: HashMap<String, Vec<u8>>) -> Self {
            Self {
                payloads,
                call_count: RefCell::new(0),
            }
        }

        fn calls(&self) -> usize {
            *self.call_count.borrow()
        }
    }

    impl MnistDownloadClient for FakeClient {
        fn download_bytes(&self, url: &str) -> Result<Vec<u8>, SyntheticError> {
            *self.call_count.borrow_mut() += 1;
            self.payloads
                .get(url)
                .cloned()
                .ok_or_else(|| SyntheticError::Download {
                    url: url.to_owned(),
                    message: "missing fake payload".to_owned(),
                })
        }
    }

    type MutationFn = fn(Vec<u8>) -> Vec<u8>;

    fn mutate_invalid_magic(mut decoded: Vec<u8>) -> Vec<u8> {
        if let Some(byte) = decoded.get_mut(3) {
            *byte = 0;
        }
        decoded
    }

    fn mutate_truncated_payload(mut decoded: Vec<u8>) -> Vec<u8> {
        decoded.truncate(decoded.len() - 1);
        decoded
    }

    #[rstest]
    #[case::invalid_magic(mutate_invalid_magic, "unexpected IDX magic")]
    #[case::truncated_payload(mutate_truncated_payload, "payload length mismatch")]
    fn parse_idx_images_rejects_invalid_data(
        #[case] mutate: MutationFn,
        #[case] expected_message: &str,
    ) {
        let payload = gzip_idx_images(2, 28, 28, 0_u8);
        let decoded_bytes = gunzip_bytes(Path::new("bad"), &payload).expect("decode must succeed");
        let mutated_bytes = mutate(decoded_bytes);
        let remade = gzip_bytes(&mutated_bytes);

        let error = parse_idx_images(Path::new("bad"), &remade)
            .expect_err("invalid IDX image payload should fail");
        let SyntheticError::InvalidMnistFile { message, .. } = error else {
            panic!("expected InvalidMnistFile error");
        };
        assert!(message.contains(expected_message));
    }

    #[rstest]
    fn load_mnist_uses_cache_after_first_download() {
        let cache_dir = test_cache_dir();
        let config = MnistConfig {
            cache_dir: cache_dir.clone(),
            base_url: "https://example.test/mnist".to_owned(),
        };

        let train_url = file_url(&config, TRAIN_IMAGES_FILE);
        let test_url = file_url(&config, TEST_IMAGES_FILE);
        let train_payload = gzip_idx_images(60_000, 28, 28, 3_u8);
        let test_payload = gzip_idx_images(10_000, 28, 28, 9_u8);

        let client = FakeClient::new(HashMap::from([
            (train_url, train_payload),
            (test_url, test_payload),
        ]));

        let first =
            load_mnist_with_client(&config, &client).expect("first load should download and cache");
        assert_eq!(first.len(), MNIST_POINT_COUNT);
        assert_eq!(first.dimensions(), MNIST_DIMENSIONS);
        assert_eq!(client.calls(), 2);

        let second =
            load_mnist_with_client(&config, &client).expect("second load should reuse cache");
        assert_eq!(second.len(), MNIST_POINT_COUNT);
        assert_eq!(client.calls(), 2);

        fs::remove_dir_all(cache_dir).expect("test cache dir cleanup must succeed");
    }

    fn test_cache_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic in tests")
            .as_nanos();
        env::temp_dir().join(format!("chutoro-mnist-test-{nanos}"))
    }

    fn gzip_idx_images(count: usize, rows: usize, cols: usize, fill: u8) -> Vec<u8> {
        let count_u32 = u32::try_from(count).expect("count should fit u32 in tests");
        let rows_u32 = u32::try_from(rows).expect("rows should fit u32 in tests");
        let cols_u32 = u32::try_from(cols).expect("cols should fit u32 in tests");

        let mut raw = Vec::new();
        append_u32_be(&mut raw, IDX_IMAGE_MAGIC);
        append_u32_be(&mut raw, count_u32);
        append_u32_be(&mut raw, rows_u32);
        append_u32_be(&mut raw, cols_u32);
        raw.extend(vec![fill; count * rows * cols]);
        gzip_bytes(&raw)
    }

    fn append_u32_be(buffer: &mut Vec<u8>, value: u32) {
        let first = u8::try_from((value >> 24) & 0xFF).expect("byte must fit u8");
        let second = u8::try_from((value >> 16) & 0xFF).expect("byte must fit u8");
        let third = u8::try_from((value >> 8) & 0xFF).expect("byte must fit u8");
        let fourth = u8::try_from(value & 0xFF).expect("byte must fit u8");
        buffer.extend([first, second, third, fourth]);
    }

    fn gzip_bytes(raw: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(raw)
            .expect("gzip payload writing must succeed in tests");
        encoder
            .finish()
            .expect("gzip payload finalization must succeed in tests")
    }
}
