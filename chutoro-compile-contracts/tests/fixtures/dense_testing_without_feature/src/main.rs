//! Downstream consumer fixture that must not see the gated dense test module.

use chutoro_providers_dense::testing as _;

fn main() {}
