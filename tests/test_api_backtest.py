"""Integration-style tests for event-hour backtest API flow."""

import torch

from backend.services.ml.event_hour import EventHourLSTM, EventHourModelConfig

PREFIX = "/api/v1"


def test_event_hour_backtest_run_and_summary(client, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.settings.MODELS_DIR", tmp_path)

    # Build tiny valid checkpoints and dataset cache.
    input_size = 4
    model = EventHourLSTM(EventHourModelConfig(input_size=input_size))
    torch.save(model.state_dict(), tmp_path / "event_hour_continuation.pt")
    torch.save(model.state_dict(), tmp_path / "event_hour_reversal.pt")
    (tmp_path / "event_hour_continuation_metrics.json").write_text('{"best_threshold": 0.5}')
    (tmp_path / "event_hour_reversal_metrics.json").write_text('{"best_threshold": 0.5}')
    (tmp_path / "event_hour_continuation_calibration.json").write_text('{"temperature": 1.0}')
    (tmp_path / "event_hour_reversal_calibration.json").write_text('{"temperature": 1.0}')

    sequences = torch.randn(8, 5, input_size)
    dataset = {
        "sequences": sequences,
        "event_type": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
        "targets_cont": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32),
        "targets_rev": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32),
        "forward_return_60m": torch.tensor([0.01, -0.02, 0.01, -0.01, 0.03, -0.02, 0.01, -0.01], dtype=torch.float32),
    }
    torch.save(dataset, tmp_path / "event_hour_dataset.pt")

    run_resp = client.post(f"{PREFIX}/backtest/event-hour/run", json={"prefix": "ci_event_hour"})
    assert run_resp.status_code == 200, run_resp.text
    body = run_resp.json()
    assert "summary" in body
    assert body["summary"]["samples"] == 8

    summary_resp = client.get(f"{PREFIX}/backtest/event-hour/summary?prefix=ci_event_hour")
    assert summary_resp.status_code == 200
    assert "trades" in summary_resp.json()

