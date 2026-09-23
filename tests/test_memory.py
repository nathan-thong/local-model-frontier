from frontier.profiling.memory import maximum_accelerator_peaks


def test_maximum_accelerator_peaks_selects_highest_phase_peak():
    rows = [
        {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200},
        {"peak_allocated_bytes": 150, "peak_reserved_bytes": 180},
    ]
    assert maximum_accelerator_peaks(rows) == {
        "peak_allocated_bytes": 150,
        "peak_reserved_bytes": 200,
    }


def test_maximum_accelerator_peaks_keeps_unsupported_sensor_null():
    assert maximum_accelerator_peaks(
        [{"peak_allocated_bytes": None, "peak_reserved_bytes": None}]
    ) == {"peak_allocated_bytes": None, "peak_reserved_bytes": None}


def test_maximum_accelerator_peaks_handles_empty_invocation():
    assert maximum_accelerator_peaks([]) == {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
