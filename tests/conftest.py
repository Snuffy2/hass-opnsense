from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def coordinator():
    return AsyncMock()


@pytest.fixture
def hass():
    """Canonical Home Assistant mock fixture with common attributes."""
    hass_instance = MagicMock()
    hass_instance.config_entries = MagicMock()
    hass_instance.config_entries.async_forward_entry_setups = AsyncMock(return_value=True)
    hass_instance.services = MagicMock()
    return hass_instance
