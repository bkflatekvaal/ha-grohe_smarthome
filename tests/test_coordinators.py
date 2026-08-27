"""Regression tests for coordinator error handling and cumulative values."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from grohe import GroheUnauthorizedError
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.grohe_smarthome.entities.coordinator.blue_home_coordinator import (
    BlueHomeCoordinator,
)
from custom_components.grohe_smarthome.entities.coordinator.blue_prof_coordinator import (
    BlueProfCoordinator,
)
from custom_components.grohe_smarthome.entities.coordinator.guard_coordinator import (
    GuardCoordinator,
)
from custom_components.grohe_smarthome.entities.coordinator.profile_coordinator import (
    ProfileCoordinator,
)
from custom_components.grohe_smarthome.entities.coordinator.sense_coordinator import (
    SenseCoordinator,
)

DOMAIN = "grohe_smarthome"
INSTALLATION_DATE = "2020-01-01T00:00:00+00:00"


def _make_device() -> SimpleNamespace:
    return SimpleNamespace(
        appliance_id="appliance-1",
        name="Test Device",
        type="grohe_sense",
        location_id="loc-1",
        room_id="room-1",
        stripped_sw_version=(1, 0),
    )


def _guard_details() -> dict:
    return {
        "installation_date": INSTALLATION_DATE,
        "data_latest": {"daily_consumption": 10},
        "status": [],
    }


def _withdrawals(*values: float | None) -> dict:
    return {"data": {"withdrawals": [{"waterconsumption": value} for value in values]}}


def _fixed_datetime(value: datetime):
    return patch(
        "custom_components.grohe_smarthome.entities.coordinator.guard_coordinator.datetime",
        wraps=datetime,
        **{"now.return_value": value},
    )


async def test_sense_coordinator_raises_update_failed_on_api_error(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(side_effect=RuntimeError("API is down"))

    coordinator = SenseCoordinator(hass, DOMAIN, _make_device(), api)

    with pytest.raises(UpdateFailed):
        await coordinator._async_update_data()


async def test_guard_coordinator_raises_update_failed_on_api_error(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(side_effect=RuntimeError("API is down"))

    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)

    with pytest.raises(UpdateFailed):
        await coordinator._async_update_data()


async def test_guard_initial_total_reconstruction(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(
        side_effect=[_withdrawals(10), _withdrawals(40, 60, 10)]
    )
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)

    with _fixed_datetime(datetime.fromisoformat("2026-08-27T12:00:00+02:00")):
        data = await coordinator._get_data()

    assert data["total_water_consumption"] == 110
    assert coordinator._total_value == 100
    assert coordinator._total_value_update_day.date().isoformat() == "2026-08-27"
    assert api.get_appliance_data.await_count == 2


async def test_guard_same_day_update_does_not_rebuild_historical_total(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(side_effect=[_withdrawals(10), _withdrawals(12)])
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)
    coordinator._total_value = 100
    coordinator._total_value_update_day = datetime.fromisoformat(
        "2026-08-27T08:00:00+02:00"
    )

    with _fixed_datetime(datetime.fromisoformat("2026-08-27T12:00:00+02:00")):
        first = await coordinator._get_data()
    with _fixed_datetime(datetime.fromisoformat("2026-08-27T13:00:00+02:00")):
        second = await coordinator._get_data()

    assert first["total_water_consumption"] == 110
    assert second["total_water_consumption"] == 112
    assert coordinator._total_value == 100
    assert api.get_appliance_data.await_count == 2


async def test_guard_today_timeout_does_not_publish_lower_total(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(side_effect=TimeoutError("timed out"))
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)
    coordinator._total_value = 100
    coordinator._total_value_update_day = datetime.fromisoformat(
        "2026-08-27T08:00:00+02:00"
    )
    previous_data = {"total_water_consumption": 110}
    coordinator.data = previous_data

    with (
        _fixed_datetime(datetime.fromisoformat("2026-08-27T12:00:00+02:00")),
        pytest.raises(UpdateFailed),
    ):
        await coordinator._async_update_data()

    assert coordinator.data is previous_data
    assert coordinator._total_value == 100


async def test_guard_historical_timeout_preserves_baseline_and_refresh_day(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(
        side_effect=[_withdrawals(2), TimeoutError("timed out")]
    )
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)
    coordinator._total_value = 100
    previous_day = datetime.fromisoformat("2026-08-26T08:00:00+02:00")
    coordinator._total_value_update_day = previous_day
    previous_data = {"total_water_consumption": 110}
    coordinator.data = previous_data

    with (
        _fixed_datetime(datetime.fromisoformat("2026-08-27T12:00:00+02:00")),
        pytest.raises(UpdateFailed),
    ):
        await coordinator._async_update_data()

    assert coordinator.data is previous_data
    assert coordinator._total_value == 100
    assert coordinator._total_value_update_day is previous_day


@pytest.mark.parametrize("payload", [{}, {"data": {}}, {"data": {"withdrawals": {}}}])
async def test_guard_invalid_withdrawals_raise(hass, payload):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(return_value=payload)
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)

    with pytest.raises(UpdateFailed, match="data.withdrawals"):
        await coordinator._async_update_data()


async def test_guard_valid_empty_withdrawals_return_zero(hass):
    api = MagicMock()
    api.get_appliance_data = AsyncMock(return_value=_withdrawals())
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)
    now = datetime.fromisoformat("2026-08-27T12:00:00+02:00")

    assert await coordinator._get_total_value(now, now, MagicMock()) == 0.0


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ("2026-08-31T23:55:00+02:00", "2026-09-01T00:05:00+02:00"),
        ("2026-12-31T23:55:00+01:00", "2027-01-01T00:05:00+01:00"),
    ],
)
async def test_guard_calendar_rollover_rebuilds_historical_total(
    hass, previous, current
):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(
        side_effect=[_withdrawals(2), _withdrawals(100, 2)]
    )
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)
    coordinator._total_value = 90
    coordinator._total_value_update_day = datetime.fromisoformat(previous)

    with _fixed_datetime(datetime.fromisoformat(current)):
        data = await coordinator._get_data()

    assert data["total_water_consumption"] == 102
    assert coordinator._total_value == 100
    assert (
        coordinator._total_value_update_day.date()
        == datetime.fromisoformat(current).date()
    )
    assert api.get_appliance_data.await_count == 2


async def test_guard_aggregate_authentication_error_retains_behavior(hass):
    api = MagicMock()
    api.get_appliance_details = AsyncMock(return_value=_guard_details())
    api.get_appliance_data = AsyncMock(
        side_effect=GroheUnauthorizedError("unauthorized")
    )
    coordinator = GuardCoordinator(hass, DOMAIN, _make_device(), api)

    with pytest.raises(ConfigEntryAuthFailed):
        await coordinator._async_update_data()


async def test_blue_prof_coordinator_raises_update_failed_on_api_error(hass):
    api = MagicMock()
    api.set_appliance_command = AsyncMock(side_effect=RuntimeError("API is down"))

    coordinator = BlueProfCoordinator(hass, DOMAIN, _make_device(), api)

    with pytest.raises(UpdateFailed):
        await coordinator._async_update_data()


async def test_profile_coordinator_raises_update_failed_on_api_error(hass):
    api = MagicMock()
    api.get_profile_notifications = AsyncMock(side_effect=RuntimeError("API is down"))

    coordinator = ProfileCoordinator(hass, DOMAIN, api)

    with pytest.raises(UpdateFailed):
        await coordinator._async_update_data()


async def test_blue_home_coordinator_propagates_error_on_first_fetch(hass):
    """BlueHomeCoordinator doesn't wrap in try/except; DataUpdateCoordinator itself
    turns the raw exception into UpdateFailed once wired up via async_refresh, but
    here we assert the underlying fetch simply raises rather than being swallowed."""
    api = MagicMock()
    api.get_appliance_details = AsyncMock(side_effect=RuntimeError("API is down"))

    coordinator = BlueHomeCoordinator(hass, DOMAIN, _make_device(), api)

    with pytest.raises(RuntimeError):
        await coordinator._async_update_data()
