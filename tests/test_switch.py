"""Unit tests for the OPNsense switch platform."""

from collections.abc import MutableMapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.opnsense import switch as switch_mod
from custom_components.opnsense.const import (
    ATTR_UNBOUND_BLOCKLIST,
    CONF_DEVICE_UNIQUE_ID,
    CONF_SYNC_FIREWALL_AND_NAT,
    CONF_SYNC_SERVICES,
    CONF_SYNC_UNBOUND,
    CONF_SYNC_VPN,
    COORDINATOR,
)
from custom_components.opnsense.coordinator import OPNsenseDataUpdateCoordinator
from custom_components.opnsense.switch import (
    OPNsenseFirewallRuleSwitch,
    OPNsenseNATRuleSwitch,
    OPNsenseServiceSwitch,
    OPNsenseUnboundBlocklistSwitch,
    OPNsenseVPNSwitch,
    _compile_firewall_rules_switches,
    _compile_nat_destination_rules_switches,
    _compile_nat_npt_rules_switches,
    _compile_nat_one_to_one_rules_switches,
    _compile_nat_source_rules_switches,
    _compile_service_switches,
    _compile_unbound_switches,
    _compile_vpn_switches,
)
from homeassistant.components.switch import SwitchEntityDescription


def make_coord(data: MutableMapping[str, Any]) -> MagicMock:
    """Create a coordinator-like mock with state data."""
    mock = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    mock.data = data
    return mock


def attach_entity(entity, hass, state: MutableMapping[str, Any]) -> None:
    """Attach Home Assistant state to a switch entity for direct exercise."""
    entity.hass = hass
    entity.coordinator = make_coord(state)
    entity.entity_id = f"switch.{getattr(entity, '_attr_unique_id', entity.entity_description.key)}"
    entity.async_write_ha_state = lambda: None


@pytest.mark.asyncio
async def test_async_setup_entry_creates_rest_only_switches(
    coordinator, ph_hass, make_config_entry
):
    """The switch platform should build entities only from the REST-native state model."""
    created_entities = []

    def _add_entities(entities):
        created_entities.extend(entities)

    coordinator.data = {
        "firewall": {
            "rules": {
                "rule1": {
                    "uuid": "rule1",
                    "description": "Allow WAN",
                    "%interface": "wan",
                    "enabled": "1",
                }
            },
            "nat": {
                "source_nat": {
                    "src1": {
                        "uuid": "src1",
                        "description": "Source NAT",
                        "%interface": "wan",
                        "enabled": "1",
                    }
                },
                "d_nat": {
                    "dst1": {
                        "uuid": "dst1",
                        "description": "Destination NAT",
                        "%interface": "wan",
                        "enabled": "1",
                    }
                },
                "one_to_one": {
                    "oto1": {
                        "uuid": "oto1",
                        "description": "One to One",
                        "%interface": "lan",
                        "enabled": "0",
                    }
                },
                "npt": {
                    "npt1": {
                        "uuid": "npt1",
                        "description": "NPTv6",
                        "%interface": "lan",
                        "enabled": "1",
                    }
                },
            },
        },
        "services": [
            {"id": "svc1", "name": "svc1", "description": "DNS", "locked": 0, "status": True},
            {"id": "svc2", "name": "svc2", "description": "Locked", "locked": 1, "status": True},
        ],
        ATTR_UNBOUND_BLOCKLIST: {
            "dnsbl1": {"enabled": "1", "description": "Primary"},
        },
        "openvpn": {
            "clients": {"client1": {"enabled": True, "name": "Road Warrior"}},
            "servers": {"server1": {"enabled": False, "name": "Site Tunnel"}},
        },
        "wireguard": {"clients": {}, "servers": {}},
    }

    config_entry = make_config_entry(
        data={
            CONF_DEVICE_UNIQUE_ID: "dev1",
            CONF_SYNC_FIREWALL_AND_NAT: True,
            CONF_SYNC_SERVICES: True,
            CONF_SYNC_UNBOUND: True,
            CONF_SYNC_VPN: True,
        }
    )
    setattr(config_entry.runtime_data, COORDINATOR, coordinator)

    await switch_mod.async_setup_entry(ph_hass, config_entry, _add_entities)

    assert len(created_entities) == 9
    assert any(isinstance(entity, OPNsenseFirewallRuleSwitch) for entity in created_entities)
    assert any(isinstance(entity, OPNsenseNATRuleSwitch) for entity in created_entities)
    assert any(isinstance(entity, OPNsenseServiceSwitch) for entity in created_entities)
    assert any(isinstance(entity, OPNsenseUnboundBlocklistSwitch) for entity in created_entities)
    assert any(isinstance(entity, OPNsenseVPNSwitch) for entity in created_entities)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("compile_fn", "state", "expected_key"),
    [
        (
            _compile_firewall_rules_switches,
            {
                "firewall": {
                    "rules": {
                        "rule1": {
                            "uuid": "rule1",
                            "description": "Allow WAN",
                            "%interface": "wan",
                            "enabled": "1",
                        }
                    }
                }
            },
            "firewall.rule.rule1",
        ),
        (
            _compile_nat_source_rules_switches,
            {
                "firewall": {
                    "nat": {
                        "source_nat": {
                            "src1": {
                                "uuid": "src1",
                                "description": "Source NAT",
                                "%interface": "wan",
                                "enabled": "1",
                            }
                        }
                    }
                }
            },
            "firewall.nat.source_nat.src1",
        ),
        (
            _compile_nat_destination_rules_switches,
            {
                "firewall": {
                    "nat": {
                        "d_nat": {
                            "dst1": {
                                "uuid": "dst1",
                                "description": "Destination NAT",
                                "%interface": "wan",
                                "enabled": "1",
                            }
                        }
                    }
                }
            },
            "firewall.nat.d_nat.dst1",
        ),
        (
            _compile_nat_one_to_one_rules_switches,
            {
                "firewall": {
                    "nat": {
                        "one_to_one": {
                            "oto1": {
                                "uuid": "oto1",
                                "description": "One to One",
                                "%interface": "lan",
                                "enabled": "1",
                            }
                        }
                    }
                }
            },
            "firewall.nat.one_to_one.oto1",
        ),
        (
            _compile_nat_npt_rules_switches,
            {
                "firewall": {
                    "nat": {
                        "npt": {
                            "npt1": {
                                "uuid": "npt1",
                                "description": "NPTv6",
                                "%interface": "lan",
                                "enabled": "1",
                            }
                        }
                    }
                }
            },
            "firewall.nat.npt.npt1",
        ),
    ],
)
async def test_compile_rest_rule_switches(make_config_entry, compile_fn, state, expected_key):
    """REST-native firewall and NAT helpers should expose the expected entity key."""
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    coordinator = make_coord(state)

    entities = await compile_fn(config_entry, coordinator, state)

    assert len(entities) == 1
    assert entities[0].entity_description.key == expected_key


@pytest.mark.asyncio
async def test_compile_service_unbound_and_vpn_switches(make_config_entry):
    """Service, unbound, and VPN helpers should skip invalid rows and create valid entities."""
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})

    service_state = {
        "services": [
            {"id": "svc1", "name": "svc1", "description": "DNS", "locked": 0, "status": True},
            {"id": "svc2", "name": "svc2", "description": "Locked", "locked": 1, "status": True},
        ]
    }
    unbound_state = {
        ATTR_UNBOUND_BLOCKLIST: {
            "dnsbl1": {"enabled": "1", "description": "Primary"},
            "dnsbl2": "invalid",
        }
    }
    vpn_state = {
        "openvpn": {
            "clients": {"client1": {"enabled": True, "name": "Road Warrior"}},
            "servers": {"server1": {"enabled": False, "name": "Site Tunnel"}},
        },
        "wireguard": {
            "clients": {"wg1": {"enabled": True, "name": "Mobile"}},
            "servers": {"wg_bad": {"name": "Missing enabled"}},
        },
    }

    services = await _compile_service_switches(
        config_entry, make_coord(service_state), service_state
    )
    unbound = await _compile_unbound_switches(
        config_entry, make_coord(unbound_state), unbound_state
    )
    vpn = await _compile_vpn_switches(config_entry, make_coord(vpn_state), vpn_state)

    assert len(services) == 1
    assert services[0].entity_description.key == "service.svc1.status"
    assert len(unbound) == 1
    assert unbound[0].entity_description.key == "unbound_blocklist.switch.dnsbl1"
    assert {entity.entity_description.key for entity in vpn} == {
        "openvpn.clients.client1",
        "openvpn.servers.server1",
        "wireguard.clients.wg1",
    }


@pytest.mark.asyncio
async def test_firewall_switch_updates_and_toggles(monkeypatch, ph_hass, make_config_entry):
    """Firewall rule switches should expose state and call the REST toggle API."""
    monkeypatch.setattr(
        "custom_components.opnsense.switch.async_call_later",
        lambda hass, delay, action: lambda: None,
    )
    state = {
        "firewall": {
            "rules": {
                "rule1": {
                    "uuid": "rule1",
                    "description": "Allow WAN",
                    "%interface": "wan",
                    "enabled": "1",
                    "source_net": "any",
                    "destination_net": "lan net",
                }
            }
        }
    }
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    entity = OPNsenseFirewallRuleSwitch(
        config_entry=config_entry,
        coordinator=make_coord(state),
        entity_description=SwitchEntityDescription(
            key="firewall.rule.rule1",
            name="Firewall: wan: Allow WAN",
        ),
    )
    attach_entity(entity, ph_hass, state)
    entity._client = MagicMock()
    entity._client.toggle_firewall_rule = AsyncMock(return_value=True)

    entity._handle_coordinator_update()
    assert entity.available is True
    assert entity.is_on is True
    assert entity.extra_state_attributes["source"] == "any"
    assert entity.icon == "mdi:play-network"

    await entity.async_turn_off()
    entity._client.toggle_firewall_rule.assert_awaited_once_with("rule1", "off")
    assert entity.is_on is False
    assert entity.delay_update is True


@pytest.mark.asyncio
async def test_nat_switch_updates_and_toggles(monkeypatch, ph_hass, make_config_entry):
    """NAT rule switches should expose rule attributes and toggle through the REST API."""
    monkeypatch.setattr(
        "custom_components.opnsense.switch.async_call_later",
        lambda hass, delay, action: lambda: None,
    )
    state = {
        "firewall": {
            "nat": {
                "d_nat": {
                    "dst1": {
                        "uuid": "dst1",
                        "description": "Destination NAT",
                        "%interface": "wan",
                        "enabled": "1",
                        "destination.%network": "wan address",
                        "target": "192.0.2.10",
                    }
                }
            }
        }
    }
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    entity = OPNsenseNATRuleSwitch(
        config_entry=config_entry,
        coordinator=make_coord(state),
        entity_description=SwitchEntityDescription(
            key="firewall.nat.d_nat.dst1",
            name="NAT Destination: wan: Destination NAT",
        ),
    )
    attach_entity(entity, ph_hass, state)
    entity._client = MagicMock()
    entity._client.toggle_nat_rule = AsyncMock(return_value=True)

    entity._handle_coordinator_update()
    assert entity.available is True
    assert entity.is_on is True
    assert entity.extra_state_attributes["destination"] == "wan address"
    assert entity.extra_state_attributes["redirect_target"] == "192.0.2.10"

    await entity.async_turn_off()
    entity._client.toggle_nat_rule.assert_awaited_once_with("d_nat", "dst1", "off")
    assert entity.is_on is False
    assert entity.delay_update is True


@pytest.mark.asyncio
async def test_service_switch_updates_and_toggles(monkeypatch, ph_hass, make_config_entry):
    """Service switches should map service status and call start/stop methods."""
    monkeypatch.setattr(
        "custom_components.opnsense.switch.async_call_later",
        lambda hass, delay, action: lambda: None,
    )
    state = {
        "services": [
            {"id": "svc1", "name": "svc1", "description": "DNS", "locked": 0, "status": False}
        ]
    }
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    entity = OPNsenseServiceSwitch(
        config_entry=config_entry,
        coordinator=make_coord(state),
        entity_description=SwitchEntityDescription(
            key="service.svc1.status",
            name="Service DNS status",
        ),
    )
    attach_entity(entity, ph_hass, state)
    entity._client = MagicMock()
    entity._client.start_service = AsyncMock(return_value=True)
    entity._client.stop_service = AsyncMock(return_value=True)

    entity._handle_coordinator_update()
    assert entity.available is True
    assert entity.is_on is False
    assert entity.extra_state_attributes["service_id"] == "svc1"

    await entity.async_turn_on()
    entity._client.start_service.assert_awaited_once_with("svc1")
    assert entity.is_on is True

    await entity.async_turn_off()
    entity._client.stop_service.assert_awaited_once_with("svc1")
    assert entity.is_on is False


@pytest.mark.asyncio
async def test_unbound_switch_handles_missing_data_and_toggles(
    monkeypatch, ph_hass, make_config_entry
):
    """Unbound blocklist switches should become unavailable on missing data and toggle by UUID."""
    monkeypatch.setattr(
        "custom_components.opnsense.switch.async_call_later",
        lambda hass, delay, action: lambda: None,
    )
    state = {
        ATTR_UNBOUND_BLOCKLIST: {
            "dnsbl1": {"enabled": "1", "description": "Primary", "nxdomain": "1"}
        }
    }
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    entity = OPNsenseUnboundBlocklistSwitch(
        config_entry=config_entry,
        coordinator=make_coord(state),
        entity_description=SwitchEntityDescription(
            key="unbound_blocklist.switch.dnsbl1",
            name="Unbound Blocklist Primary",
        ),
    )
    attach_entity(entity, ph_hass, state)
    entity._client = MagicMock()
    entity._client.enable_unbound_blocklist = AsyncMock(return_value=True)
    entity._client.disable_unbound_blocklist = AsyncMock(return_value=True)

    entity._handle_coordinator_update()
    assert entity.available is True
    assert entity.is_on is True
    assert entity.extra_state_attributes["Return NXDOMAIN"] is True

    await entity.async_turn_off()
    entity._client.disable_unbound_blocklist.assert_awaited_once_with("dnsbl1")
    assert entity.is_on is False

    entity.delay_update = False
    entity.coordinator = make_coord({ATTR_UNBOUND_BLOCKLIST: {}})
    entity._handle_coordinator_update()
    assert entity.available is False


@pytest.mark.asyncio
async def test_vpn_switch_toggles_and_honors_noop_preconditions(
    monkeypatch, ph_hass, make_config_entry
):
    """VPN switches should toggle through the client and skip duplicate actions."""
    monkeypatch.setattr(
        "custom_components.opnsense.switch.async_call_later",
        lambda hass, delay, action: lambda: None,
    )
    state = {"openvpn": {"clients": {"client1": {"enabled": False, "name": "Road Warrior"}}}}
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    entity = OPNsenseVPNSwitch(
        config_entry=config_entry,
        coordinator=make_coord(state),
        entity_description=SwitchEntityDescription(
            key="openvpn.clients.client1",
            name="OpenVPN Client Road Warrior",
        ),
    )
    attach_entity(entity, ph_hass, state)
    entity._client = MagicMock()
    entity._client.toggle_vpn_instance = AsyncMock(return_value=True)

    entity._handle_coordinator_update()
    assert entity.available is True
    assert entity.is_on is False

    await entity.async_turn_on()
    entity._client.toggle_vpn_instance.assert_awaited_once_with(
        "openvpn",
        "clients",
        "client1",
    )
    assert entity.is_on is True
    assert entity.delay_update is True

    before = entity._client.toggle_vpn_instance.await_count
    await entity.async_turn_on()
    assert entity._client.toggle_vpn_instance.await_count == before


def test_delay_update_setter_captures_and_clears_remover(monkeypatch, make_config_entry):
    """The base switch delay flag should capture and clear the scheduled remover."""
    config_entry = make_config_entry(data={CONF_DEVICE_UNIQUE_ID: "dev1"})
    entity = OPNsenseFirewallRuleSwitch(
        config_entry=config_entry,
        coordinator=make_coord({"firewall": {"rules": {}}}),
        entity_description=SwitchEntityDescription(
            key="firewall.rule.rule1",
            name="Firewall: wan: Allow WAN",
        ),
    )
    entity.hass = MagicMock()
    called = {"removed": False}

    def _fake_async_call_later(*args, **kwargs):
        def _remover():
            called["removed"] = True

        return _remover

    monkeypatch.setattr(
        "custom_components.opnsense.switch.async_call_later", _fake_async_call_later
    )

    entity.delay_update = True
    assert entity.delay_update is True
    assert callable(entity._delay_update_remove)

    entity.delay_update = False
    assert called["removed"] is True
    assert entity.delay_update is False
