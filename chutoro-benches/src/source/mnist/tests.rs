//! Unit tests for MNIST parsing and cache helpers.

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

    let second = load_mnist_with_client(&config, &client).expect("second load should reuse cache");
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
