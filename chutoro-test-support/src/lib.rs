//! Shared test utilities used across chutoro crates.

pub mod tracing {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::{Event, Level, Subscriber};
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::registry::LookupSpan;

    #[derive(Clone, Default)]
    pub struct RecordingLayer {
        spans: Arc<Mutex<Vec<SpanRecord>>>,
        events: Arc<Mutex<Vec<EventRecord>>>,
    }

    impl RecordingLayer {
        #[must_use]
        pub fn spans(&self) -> Vec<SpanRecord> {
            self.spans.lock().expect("lock poisoned").clone()
        }

        #[must_use]
        pub fn events(&self) -> Vec<EventRecord> {
            self.events.lock().expect("lock poisoned").clone()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SpanRecord {
        pub name: String,
        pub fields: HashMap<String, String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EventRecord {
        pub level: Level,
        pub target: String,
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
            if let Some(span) = ctx.span(id)
                && let Some(data) = span.extensions_mut().get_mut::<SpanData>()
            {
                values.record(&mut FieldRecorder {
                    fields: &mut data.fields,
                });
            }
        }

        fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(&id)
                && let Some(data) = span.extensions_mut().remove::<SpanData>()
            {
                self.spans.lock().expect("lock poisoned").push(SpanRecord {
                    name: data.name,
                    fields: data.fields,
                });
            }
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

    impl<'a> Visit for FieldRecorder<'a> {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.fields.insert(
                field.name().to_owned(),
                format!("{value:?}").trim_matches('"').to_owned(),
            );
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
