import torch
from torch import nn

from frontier.profiling.benchmark import _state_memory_accounting
from frontier.profiling.memory import isolated_process_peak, maximum_accelerator_peaks


def _touch_pages(size_bytes: int) -> None:
    memory = bytearray(size_bytes)
    for index in range(0, size_bytes, 4096):
        memory[index] = 1


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


def test_isolated_process_peaks_do_not_retain_prior_workload_high_water():
    large_bytes = 32 * 1024 * 1024
    small_bytes = 1024 * 1024

    large_first = isolated_process_peak(_touch_pages, (large_bytes,))
    small_second = isolated_process_peak(_touch_pages, (small_bytes,))
    small_first = isolated_process_peak(_touch_pages, (small_bytes,))
    large_second = isolated_process_peak(_touch_pages, (large_bytes,))

    assert all(
        measurement["status"] == "measured"
        for measurement in (large_first, small_second, small_first, large_second)
    )
    assert (
        large_first["peak_rss_increase_bytes"]
        > small_second["peak_rss_increase_bytes"] + 16 * 1024 * 1024
    )
    assert (
        large_second["peak_rss_increase_bytes"]
        > small_first["peak_rss_increase_bytes"] + 16 * 1024 * 1024
    )
    assert large_first["isolation"] == "fresh spawned process per CPU phase"


class _ActiveViews(nn.Module):
    def active_state_tensors(self, state):
        return state["active"]


def test_nested_state_accounting_unions_aliases_and_active_regions_once():
    backing = torch.zeros(16, dtype=torch.float32)
    state = {
        "nested": [backing, {"alias": backing[4:12]}],
        "active": (backing[:8], backing[4:12]),
    }

    measured = _state_memory_accounting([_ActiveViews()], [state])

    assert measured["allocated_bytes"] == 16 * 4
    assert measured["active_bytes"] == 12 * 4
    assert measured["status"].startswith("exact union")


def test_state_active_accounting_is_unavailable_without_module_contract():
    class UnknownState(nn.Module):
        pass

    state = {"nested": [torch.zeros(4)]}
    measured = _state_memory_accounting([UnknownState()], [state])

    assert measured["allocated_bytes"] == 4 * 4
    assert measured["active_bytes"] is None
    assert "does not expose active state" in measured["status"]


def test_opaque_state_without_tensor_contract_does_not_report_zero_bytes():
    class OpaqueState:
        def __init__(self):
            self.tensor = torch.zeros(4)

    class OpaqueSequence(nn.Module):
        def active_state_tensors(self, state):
            return (state.tensor,)

    measured = _state_memory_accounting([OpaqueSequence()], [OpaqueState()])

    assert measured["allocated_bytes"] is None
    assert measured["active_bytes"] is None
    assert "opaque state without state_tensors" in measured["status"]
