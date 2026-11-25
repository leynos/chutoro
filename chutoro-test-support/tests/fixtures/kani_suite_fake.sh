#!/bin/sh
if [ -n "$FAKE_KANI_MARKER" ]; then
  printf '%s\n' "$FAKE_KANI_SELECTOR" >> "$FAKE_KANI_MARKER"
fi
printf 'Kani Rust Verifier %s (cargo plugin)\n' "$FAKE_KANI_VERSION"
case "${FAKE_KANI_MODE:-ok}" in
  ok)
    printf 'Checking harness proofs::verify_one...\n'
    printf 'SUMMARY:\n ** 0 of 3 failed\n\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n'
    ;;
  two)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Checking harness proofs::verify_two...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 2 successfully verified harnesses, 0 failures, 2 total.\n'
    ;;
  missing)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 1 successfully verified harnesses, 0 failures, 1 total.\n'
    ;;
  unexpected)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Checking harness proofs::verify_unlisted...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 2 successfully verified harnesses, 0 failures, 2 total.\n'
    ;;
  duplicate)
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Checking harness proofs::verify_one...\nVERIFICATION:- SUCCESSFUL\n'
    printf 'Manual Harness Summary:\n'
    printf 'Complete - 2 successfully verified harnesses, 0 failures, 2 total.\n'
    ;;
  failed)
    printf 'Checking harness proofs::verify_one...\n'
    printf 'SUMMARY:\n ** 1 of 3 failed\n\nVERIFICATION:- FAILED\n'
    printf 'Manual Harness Summary:\n'
    printf 'Verification failed for - proofs::verify_one\n'
    printf 'Complete - 0 successfully verified harnesses, 1 failures, 1 total.\n'
    ;;
  incomplete)
    printf 'Checking harness proofs::verify_one...\n'
    ;;
  no-summary)
    printf 'Checking harness proofs::verify_one...\n'
    printf 'SUMMARY:\n ** 0 of 3 failed\n\nVERIFICATION:- SUCCESSFUL\n'
    ;;
  silent)
    ;;
esac
exit "${FAKE_KANI_STATUS:-0}"
