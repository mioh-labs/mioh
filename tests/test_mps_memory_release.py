from unittest import mock

from lada.utils import mps_utils


def _memory_stats(*, pressure: float, available_gb: float = 16.0):
    gib = 1024 ** 3
    return {
        "pressure_ratio": pressure,
        "system_available_bytes": int(available_gb * gib),
        "system_total_bytes": 48 * gib,
    }


def test_mps_cache_is_kept_without_memory_pressure():
    with (
        mock.patch("torch.backends.mps.is_available", return_value=True),
        mock.patch.object(
            mps_utils,
            "get_mps_memory_stats",
            return_value=_memory_stats(pressure=0.50),
        ),
        mock.patch("torch.mps.empty_cache") as empty_cache,
    ):
        released = mps_utils.release_mps_memory_if_needed()

    assert released is False
    empty_cache.assert_not_called()


def test_mps_cache_release_is_synchronized_under_pressure():
    mps_utils._MPS_LAST_CACHE_RELEASE_AT = 0.0
    with (
        mock.patch("torch.backends.mps.is_available", return_value=True),
        mock.patch.object(
            mps_utils,
            "get_mps_memory_stats",
            return_value=_memory_stats(pressure=0.90),
        ),
        mock.patch("gc.collect") as collect,
        mock.patch("torch.mps.synchronize") as synchronize,
        mock.patch("torch.mps.empty_cache") as empty_cache,
    ):
        released = mps_utils.release_mps_memory_if_needed(cooldown_seconds=0)

    assert released is True
    collect.assert_called_once_with()
    synchronize.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_mps_cache_release_is_rate_limited():
    mps_utils._MPS_LAST_CACHE_RELEASE_AT = 0.0
    with (
        mock.patch("torch.backends.mps.is_available", return_value=True),
        mock.patch.object(
            mps_utils,
            "get_mps_memory_stats",
            return_value=_memory_stats(pressure=0.90),
        ),
        mock.patch("torch.mps.synchronize"),
        mock.patch("torch.mps.empty_cache") as empty_cache,
    ):
        first = mps_utils.release_mps_memory_if_needed(cooldown_seconds=60)
        second = mps_utils.release_mps_memory_if_needed(cooldown_seconds=60)

    assert first is True
    assert second is False
    empty_cache.assert_called_once_with()


def test_mps_cache_release_failure_does_not_crash_pipeline_worker():
    mps_utils._MPS_LAST_CACHE_RELEASE_AT = 0.0
    with (
        mock.patch("torch.backends.mps.is_available", return_value=True),
        mock.patch.object(
            mps_utils,
            "get_mps_memory_stats",
            return_value=_memory_stats(pressure=0.90),
        ),
        mock.patch("torch.mps.synchronize", side_effect=RuntimeError("test failure")),
        mock.patch("torch.mps.empty_cache") as empty_cache,
    ):
        released = mps_utils.release_mps_memory_if_needed(cooldown_seconds=0)

    assert released is False
    empty_cache.assert_not_called()
