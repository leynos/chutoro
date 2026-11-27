//! Shared test utilities used across chutoro crates.

pub mod tracing {
    //! Recording layer utilities for capturing spans and events in tests.
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::subscriber::NoSubscriber;
    use tracing::{Dispatch, Event, Level, Subscriber};
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::registry::LookupSpan;

    /// Keeps a callsite's cached interest from being fixed at `never` by a
    /// first touch from a thread with no recording subscriber installed.
    ///
    /// `tracing-core` caches the [`Interest`] of every callsite the first time
    /// that callsite is recorded. While only one [`Dispatch`] is registered,
    /// `tracing-core` 0.1.36 rebuilds that interest through
    /// `Rebuilder::JustOne`, which asks only the current thread's default
    /// dispatcher. A test that records nothing can therefore cache
    /// [`Interest::never`] for a callsite before a capture test's
    /// `with_default` scope reaches it, because its thread has no default
    /// dispatcher to ask.
    ///
    /// This guard registers a second, inert [`Dispatch`] built from
    /// [`NoSubscriber`]. With two live dispatches, `tracing-core` rebuilds
    /// callsite interest from every registered dispatcher, and the inert
    /// dispatcher's `never` combines with the recording subscriber's `always`
    /// as [`Interest::sometimes`] rather than `never`. The recording
    /// subscriber is then asked on the thread that actually records.
    ///
    /// The pin records nothing and installs no subscriber. Create it before the
    /// recording subscriber, and keep it alive until `with_default` returns and
    /// every snapshot has been taken.
    ///
    /// ## Residual upstream race
    ///
    /// `DefaultCallsite::register` reads the registry's `has_just_one` flag
    /// before it computes interest, and stores the computed interest
    /// unconditionally afterwards. A registration that straddles a
    /// `Dispatch::new` can therefore store `never` after the rebuild that would
    /// have corrected it, even with several dispatches live. No sequence of
    /// downstream dispatches closes that window; it needs a fix in
    /// `tracing-core` itself. Create the pin immediately before the recording
    /// subscriber so the unprotected interval is limited to dispatch setup,
    /// rather than the whole time a capture test is running.
    ///
    /// [`Dispatch`]: tracing::Dispatch
    /// [`Interest`]: tracing::subscriber::Interest
    /// [`Interest::never`]: tracing::subscriber::Interest::never
    /// [`Interest::sometimes`]: tracing::subscriber::Interest::sometimes
    /// [`NoSubscriber`]: tracing::subscriber::NoSubscriber
    #[must_use = "callsite interest is only pinned while the guard is alive"]
    pub struct CallsiteInterestPin {
        /// Inert dispatcher whose sole purpose is to keep the callsite registry
        /// holding more than one live dispatch.
        _inert: Dispatch,
    }

    /// Registers an inert dispatcher that keeps callsite interest from being
    /// cached as [`Interest::never`] by an unscoped first touch.
    ///
    /// See [`CallsiteInterestPin`] for the mechanism and the lifetime rule.
    ///
    /// [`Interest::never`]: tracing::subscriber::Interest::never
    ///
    /// # Examples
    /// ```
    /// use chutoro_test_support::tracing::{RecordingLayer, pin_callsite_interest};
    /// use tracing_subscriber::layer::SubscriberExt;
    ///
    /// let _pin = pin_callsite_interest();
    /// let layer = RecordingLayer::default();
    /// let subscriber = tracing_subscriber::registry().with(layer.clone());
    ///
    /// tracing::subscriber::with_default(subscriber, || {
    ///     tracing::info_span!("example_span").in_scope(|| {});
    /// });
    ///
    /// assert_eq!(
    ///     layer.spans().first().map(|span| span.name.as_str()),
    ///     Some("example_span"),
    /// );
    /// ```
    #[must_use = "callsite interest is only pinned while the guard is alive"]
    pub fn pin_callsite_interest() -> CallsiteInterestPin {
        CallsiteInterestPin {
            _inert: Dispatch::new(NoSubscriber::new()),
        }
    }

    /// Recording layer installed during tests to capture spans and events for
    /// later assertions. The layer records structured metadata so integration
    /// and behavioural tests can verify instrumentation deterministically.
    #[derive(Clone, Default)]
    pub struct RecordingLayer {
        /// Closed spans captured in completion order.
        spans: Arc<Mutex<Vec<SpanRecord>>>,
        /// Events captured in emission order.
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
            self.spans
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
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
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
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

    /// Mutable span data accumulated before the span closes.
    #[derive(Default)]
    struct SpanData {
        /// Name read from the tracing metadata.
        name: String,
        /// Structured values recorded against the span.
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
            self.spans
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(SpanRecord {
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
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(EventRecord {
                    level: *event.metadata().level(),
                    target: event.metadata().target().to_owned(),
                    fields,
                });
        }
    }

    /// Visitor that writes tracing fields into a string-valued map.
    struct FieldRecorder<'a> {
        /// Map receiving the serialized field values.
        fields: &'a mut HashMap<String, String>,
    }

    impl Visit for FieldRecorder<'_> {
        fn record_bytes(&mut self, field: &Field, value: &[u8]) {
            let encoded = value.iter().fold(
                String::with_capacity(value.len() * 2),
                |mut encoded, byte| {
                    use std::fmt::Write as _;
                    let _write_result = write!(&mut encoded, "{byte:02x}");
                    encoded
                },
            );
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
