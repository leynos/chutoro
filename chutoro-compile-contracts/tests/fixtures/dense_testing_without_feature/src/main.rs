//! Downstream consumer fixture that must not see the gated dense test module.

use chutoro_providers_dense::testing::build_array;

fn main() {
    let _array = build_array(&[[1.0, 2.0]]);
}
