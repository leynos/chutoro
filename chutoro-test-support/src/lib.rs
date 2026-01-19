//! Shared test utilities used across chutoro crates.

pub mod tracing {
    //! Recording layer utilities for capturing spans and events in tests.
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::{Event, Level, Subscriber};
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::registry::LookupSpan;

    /// Recording layer installed during tests to capture spans and events for
    /// later assertions. The layer records structured metadata so integration
    /// and behavioural tests can verify instrumentation deterministically.
    #[derive(Clone, Default)]
    pub struct RecordingLayer {
        spans: Arc<Mutex<Vec<SpanRecord>>>,
        events: Arc<Mutex<Vec<EventRecord>>>,
    }

    impl RecordingLayer {
        /// Returns a snapshot of the closed spans recorded by the layer in
        /// completion order so instrumentation can be asserted without holding
        /// the internal lock.
        ///
        /// # Examples
        /// ```
        /// use chutoro_test_support::tracing::RecordingLayer;
        ///
        /// let layer = RecordingLayer::default();
        /// assert!(layer.spans().is_empty());
        /// ```
        #[must_use]
        pub fn spans(&self) -> Vec<SpanRecord> {
            self.spans.lock().expect("lock poisoned").clone()
        }

        /// Returns a snapshot of the emitted events recorded by the layer in
        /// emission order for verifying structured diagnostics in tests.
        ///
        /// # Examples
        /// ```
        /// use chutoro_test_support::tracing::RecordingLayer;
        ///
        /// let layer = RecordingLayer::default();
        /// assert!(layer.events().is_empty());
        /// ```
        #[must_use]
        pub fn events(&self) -> Vec<EventRecord> {
            self.events.lock().expect("lock poisoned").clone()
        }
    }

    /// Snapshot of a closed span, including its name and recorded fields, used
    /// by tests to assert span metadata and field values.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SpanRecord {
        /// Span name captured from the tracing metadata.
        pub name: String,
        /// Structured fields recorded against the span.
        pub fields: HashMap<String, String>,
    }

    /// Snapshot of an emitted tracing event, capturing its level, target, and
    /// structured fields so tests can assert diagnostic payloads precisely.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EventRecord {
        /// Log level associated with the recorded event.
        pub level: Level,
        /// Event target string extracted from the metadata.
        pub target: String,
        /// Structured fields attached to the event.
        pub fields: HashMap<String, String>,
    }

    #[derive(Default)]
    struct SpanData {
        name: String,
        fields: HashMap<String, String>,
    }

    impl<S> Layer<S> for RecordingLayer
    where
        S: Subscriber + for<'span> LookupSpan<'span>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::span::Id,
            ctx: Context<'_, S>,
        ) {
            if let Some(span) = ctx.span(id) {
                let mut data = SpanData {
                    name: attrs.metadata().name().to_owned(),
                    fields: HashMap::new(),
                };
                attrs.record(&mut FieldRecorder {
                    fields: &mut data.fields,
                });
                span.extensions_mut().insert(data);
            }
        }

        fn on_record(
            &self,
            id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            ctx: Context<'_, S>,
        ) {
            let Some(span) = ctx.span(id) else {
                return;
            };
            let mut extensions = span.extensions_mut();
            let Some(data) = extensions.get_mut::<SpanData>() else {
                return;
            };
            values.record(&mut FieldRecorder {
                fields: &mut data.fields,
            });
        }

        fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
            let Some(span) = ctx.span(&id) else {
                return;
            };
            let Some(data) = span.extensions_mut().remove::<SpanData>() else {
                return;
            };
            self.spans.lock().expect("lock poisoned").push(SpanRecord {
                name: data.name,
                fields: data.fields,
            });
        }

        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut fields = HashMap::new();
            event.record(&mut FieldRecorder {
                fields: &mut fields,
            });
            self.events
                .lock()
                .expect("lock poisoned")
                .push(EventRecord {
                    level: *event.metadata().level(),
                    target: event.metadata().target().to_owned(),
                    fields,
                });
        }
    }

    struct FieldRecorder<'a> {
        fields: &'a mut HashMap<String, String>,
    }

    impl Visit for FieldRecorder<'_> {
        fn record_bytes(&mut self, field: &Field, value: &[u8]) {
            let mut encoded = String::with_capacity(value.len() * 2);
            for byte in value {
                use std::fmt::Write as _;
                let _ = write!(&mut encoded, "{byte:02x}");
            }
            self.fields.insert(field.name().to_owned(), encoded);
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.fields
                .insert(field.name().to_owned(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.fields
                .insert(field.name().to_owned(), value.to_owned());
        }

        fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_bool(&mut self, field: &Field, value: bool) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_i64(&mut self, field: &Field, value: i64) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_i128(&mut self, field: &Field, value: i128) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_u128(&mut self, field: &Field, value: u128) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_f64(&mut self, field: &Field, value: f64) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }
    }
}

pub mod ci;
