"""Unit tests for event-hour signal threshold logic."""

from backend.api.routes.event_hour import choose_event_hour_signal


def test_choose_signal_prefers_continuation_on_tie():
    signal = choose_event_hour_signal(
        continuation_prob=0.70,
        continuation_threshold=0.60,
        reversal_prob=0.70,
        reversal_threshold=0.60,
    )
    assert signal == "continuation"


def test_choose_signal_reversal_when_only_reversal_hits():
    signal = choose_event_hour_signal(
        continuation_prob=0.40,
        continuation_threshold=0.60,
        reversal_prob=0.62,
        reversal_threshold=0.60,
    )
    assert signal == "reversal"


def test_choose_signal_neutral_when_none_hit():
    signal = choose_event_hour_signal(
        continuation_prob=0.40,
        continuation_threshold=0.60,
        reversal_prob=0.42,
        reversal_threshold=0.60,
    )
    assert signal == "neutral"

