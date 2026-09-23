import csv
import json

from frontier.experiments.results import flatten, write_summary


def test_flatten_preserves_profile_workloads_and_repeat_vectors(tmp_path):
    summary = {
        "profiling": {
            "workloads": [
                {
                    "prompt_tokens": 8,
                    "decode_latency": {
                        "repetitions_seconds": [0.01, 0.02],
                        "median_seconds": 0.015,
                    },
                },
                {"prompt_tokens": 16, "decode_latency": {"repetitions_seconds": [0.03]}},
            ]
        }
    }

    flattened = flatten(summary)
    write_summary(tmp_path, summary)
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}

    assert flattened["profiling.workloads[0].prompt_tokens"] == 8
    assert flattened["profiling.workloads[0].decode_latency.repetitions_seconds"] == json.dumps(
        [0.01, 0.02]
    )
    assert flattened["profiling.workloads[1].prompt_tokens"] == 16
    assert csv_rows["profiling.workloads[1].decode_latency.repetitions_seconds"] == "[0.03]"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
