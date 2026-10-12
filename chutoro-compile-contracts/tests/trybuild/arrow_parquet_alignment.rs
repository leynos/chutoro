//! Compile-pass contract for Arrow/Parquet type alignment.

use arrow_array::RecordBatchReader;
use arrow_schema::{DataType, SchemaRef};
use bytes::Bytes;
use chutoro_providers_dense::DenseMatrixProviderError;
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::reader::ChunkReader,
};

fn schema_from_reader<R>(reader: &R) -> SchemaRef
where
    R: RecordBatchReader,
{
    reader.schema()
}

fn accept_data_type(_: DataType) {}

fn verify_arrow_parquet_alignment<R>(source: R) -> Result<(), DenseMatrixProviderError>
where
    R: ChunkReader + Send + 'static,
{
    let builder = ParquetRecordBatchReaderBuilder::try_new(source)?;
    let reader = builder.with_projection(ProjectionMask::all()).build()?;
    let schema = schema_from_reader(&reader);
    for field in schema.fields() {
        accept_data_type(field.data_type().clone());
    }
    for batch in reader {
        let batch = batch?;
        for column in batch.columns() {
            accept_data_type(column.data_type().clone());
        }
    }
    Ok(())
}

fn main() {
    let _ = verify_arrow_parquet_alignment::<Bytes>;
}
