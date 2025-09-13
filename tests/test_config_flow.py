"""Unit tests for the config flow and options flow of the hass-opnsense integration.

Tests include URL parsing/validation, exception mapping for user input,
and options flow behaviors such as device tracker handling.
"""

import importlib
import socket
from unittest.mock import AsyncMock, MagicMock
import xmlrpc.client

import aiohttp
import pytest
from yarl import URL

cf_mod = importlib.import_module("custom_components.opnsense.config_flow")


def test_mac_and_ip_and_cleanse():
    """Validate MAC/IP helpers and cleanse sensitive data."""
    assert cf_mod.is_valid_mac_address("aa:bb:cc:dd:ee:ff")
    assert not cf_mod.is_valid_mac_address("not-a-mac")

    # IP validation
    assert cf_mod.is_ip_address("192.168.1.1")
    assert not cf_mod.is_ip_address("not-an-ip")

    # cleanse sensitive data
    msg = "user=admin&pass=secret"
    out = cf_mod.cleanse_sensitive_data(msg, ["secret"])
    assert "[redacted]" in out
    assert "secret" not in out


@pytest.mark.asyncio
async def test_clean_and_parse_url_success_and_failure():
    """Clean and parse URL, fix missing scheme and handle invalid URL."""
    ui = {cf_mod.CONF_URL: "router.example"}
    await cf_mod._clean_and_parse_url(ui)
    assert ui[cf_mod.CONF_URL] == "https://router.example"

    # invalid netloc -> raise InvalidURL
    with pytest.raises(cf_mod.InvalidURL):
        await cf_mod._clean_and_parse_url({cf_mod.CONF_URL: ""})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_key, expected",
    [
        ("below_min", "below_min_firmware"),
        ("unknown_fw", "unknown_firmware"),
        ("missing_id", "missing_device_unique_id"),
        ("plugin_missing", "plugin_missing"),
        ("invalid_url", "invalid_url_format"),
        ("xmlrpc_invalid_auth", "invalid_auth"),
        ("xmlrpc_privilege", "privilege_missing"),
        ("xmlrpc_plugin", "plugin_missing"),
        ("xmlrpc_other", "cannot_connect"),
        ("client_connector_ssl", "cannot_connect_ssl"),
        ("resp_401", "invalid_auth"),
        ("resp_403", "privilege_missing"),
        ("resp_500", "cannot_connect"),
        ("protocol_307", "url_redirect"),
        ("too_many_redirects", "url_redirect"),
        ("timeout", "connect_timeout"),
        ("server_timeout", "connect_timeout"),
        ("os_ssl", "privilege_missing"),
        ("os_timed_out", "connect_timeout"),
        ("os_ssl_handshake", "cannot_connect_ssl"),
        ("os_unknown", "unknown"),
    ],
)
async def test_validate_input_exception_mapping(monkeypatch, exc_key, expected):
    """Ensure validate_input maps various exceptions to the expected error code."""

    # Build exception object lazily to avoid constructor issues at collection time
    if exc_key == "below_min":
        exc = cf_mod.BelowMinFirmware()
    elif exc_key == "unknown_fw":
        exc = cf_mod.UnknownFirmware()
    elif exc_key == "missing_id":
        exc = cf_mod.MissingDeviceUniqueID("x")
    elif exc_key == "plugin_missing":
        exc = cf_mod.PluginMissing()
    elif exc_key == "invalid_url":
        exc = aiohttp.InvalidURL("u")
    elif exc_key == "xmlrpc_invalid_auth":
        exc = xmlrpc.client.Fault(1, "Invalid username or password")
    elif exc_key == "xmlrpc_privilege":
        exc = xmlrpc.client.Fault(1, "Authentication failed: not enough privileges")
    elif exc_key == "xmlrpc_plugin":
        exc = xmlrpc.client.Fault(1, "opnsense.exec_php does not exist")
    elif exc_key == "xmlrpc_other":
        exc = xmlrpc.client.Fault(1, "other fault")
    elif exc_key == "client_connector_ssl":
        # Simulate an SSL-related client error that maps to "cannot_connect_ssl".
        # ClientSSLError (and its base ClientConnectorError) require a connection
        # key and an underlying os_error; provide a minimal connector-like
        # object and an OSError to construct the exception instance.
        class Conn:
            host = "host.example"
            port = 443
            ssl = None

        exc = aiohttp.ClientSSLError(Conn(), OSError("ssl error"))
    elif exc_key in ("resp_401", "resp_403", "resp_500"):
        status = 401 if exc_key == "resp_401" else 403 if exc_key == "resp_403" else 500

        # Provide minimal request_info with a real_url to satisfy logging/str()
        class RI:
            real_url = URL("http://localhost")

        exc = aiohttp.ClientResponseError(request_info=RI(), history=(), status=status, message="m")
    elif exc_key == "protocol_307":
        exc = xmlrpc.client.ProtocolError("u", 307, "307 Temporary Redirect", {})
    elif exc_key == "too_many_redirects":

        class RI:
            real_url = URL("http://localhost")

        exc = aiohttp.TooManyRedirects(request_info=RI(), history=())
    elif exc_key == "timeout":
        exc = TimeoutError("t")
    elif exc_key == "server_timeout":
        exc = aiohttp.ServerTimeoutError("t")
    elif exc_key == "os_ssl":
        exc = OSError("unsupported XML-RPC protocol")
    elif exc_key == "os_timed_out":
        exc = OSError("timed out")
    elif exc_key == "os_ssl_handshake":
        exc = OSError("SSL: handshake")
    else:
        exc = OSError("unknown")

    async def _raiser(*args, **kwargs):
        raise exc

    monkeypatch.setattr(cf_mod, "_handle_user_input", _raiser)
    errors = {}
    res = await cf_mod.validate_input(
        hass=MagicMock(), user_input={}, errors=errors, config_step="user"
    )
    assert res.get("base") == expected


def test_validate_firmware_version_raises():
    """_validate_firmware_version should raise BelowMinFirmware for old versions."""
    # pick an obviously old version
    with pytest.raises(cf_mod.BelowMinFirmware):
        cf_mod._validate_firmware_version("1.0")


def test_log_and_set_error_sets_base(caplog):
    """_log_and_set_error should log the message and set errors['base']."""
    errors = {}
    cf_mod._log_and_set_error(errors=errors, key="test_key", message="an msg")
    assert errors.get("base") == "test_key"
    assert "an msg" in caplog.text


@pytest.mark.asyncio
async def test_get_dt_entries_sorts_and_includes_selected(
    monkeypatch,
    fake_client,
    patch_async_create_clientsession,
    patch_cf_opnsense_client,
    hass_with_running_loop,
):
    """Ensure _get_dt_entries returns selected devices first and ARP entries sorted by IP."""

    # Create a client class via fixture and attach a get_arp_table implementation
    client_cls = fake_client()

    async def _get_arp_table(self, resolve_hostnames=True):
        return [
            {"mac": "aa:bb:cc:00:00:01", "hostname": "hostb", "ip": "192.168.1.20"},
            {"mac": "11:22:33:44:55:66", "hostname": "", "ip": "10.0.0.5"},
            {"mac": "bb:cc:dd:00:00:02", "hostname": "hosta", "ip": "192.168.1.10"},
        ]

    setattr(client_cls, "get_arp_table", _get_arp_table)
    patch_cf_opnsense_client(client_cls)

    # Patch async_create_clientsession on the module under test to avoid real network I/O
    patch_async_create_clientsession(lambda *args, **kwargs: MagicMock())

    hass = hass_with_running_loop
    config = {cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"}
    selected = ["aa:bb:cc:00:00:01"]
    res = await cf_mod._get_dt_entries(hass=hass, config=config, selected_devices=selected)

    # ensure selected device is present and IP-based entries are present
    keys = list(res.keys())
    assert "aa:bb:cc:00:00:01" in keys
    assert "11:22:33:44:55:66" in keys
    # Smallest IP-labeled entry appears first after sort
    assert keys[0] == "11:22:33:44:55:66"
    # IP-labeled entries are sorted numerically (10.0.0.5 before 192.168.1.10 < 192.168.1.20)
    vals = list(res.values())
    assert vals.index("10.0.0.5 [11:22:33:44:55:66]") < vals.index(
        "192.168.1.10 (hosta) [bb:cc:dd:00:00:02]"
    )
    assert vals.index("192.168.1.10 (hosta) [bb:cc:dd:00:00:02]") < vals.index(
        "192.168.1.20 (hostb) [aa:bb:cc:00:00:01]"
    )


def test_build_user_input_and_granular_and_options_schemas_defaults():
    """Verify the schema builders accept empty input and return defaults where applicable."""
    uis = None
    # user input schema should provide keys and defaults
    schema = cf_mod._build_user_input_schema(user_input=uis)
    validated = schema({})
    assert cf_mod.CONF_URL in validated

    # granular sync schema
    gschema = cf_mod._build_granular_sync_schema(user_input=None)
    gvalidated = gschema({})
    # every granular item should be present (defaults applied)
    for item in cf_mod.GRANULAR_SYNC_ITEMS:
        assert item in gvalidated

    # options init schema: test clamping/coercion for scan interval
    oschema = cf_mod._build_options_init_schema(user_input=None)
    out = oschema({})
    assert cf_mod.CONF_SCAN_INTERVAL in out


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (5, 10),  # below minimum -> clamped to 10
        (150, 150),  # within range -> unchanged
        (1000, 300),  # above maximum -> clamped to 300
    ],
)
def test_options_scan_interval_clamp(input_value, expected):
    """_build_options_init_schema should clamp CONF_SCAN_INTERVAL to min/max values."""
    oschema = cf_mod._build_options_init_schema(user_input=None)
    # pass a dict with the scan interval set to the test value
    validated = oschema({cf_mod.CONF_SCAN_INTERVAL: input_value})
    assert validated.get(cf_mod.CONF_SCAN_INTERVAL) == expected


def test_async_get_options_flow_returns_options_flow():
    """async_get_options_flow should return an OPNsenseOptionsFlow instance."""
    cfg = MagicMock()
    res = cf_mod.OPNsenseConfigFlow.async_get_options_flow(cfg)
    assert isinstance(res, cf_mod.OPNsenseOptionsFlow)


@pytest.mark.asyncio
async def test_options_flow_init_with_user_triggers_update(ph_hass):
    """Submitting user input to async_step_init should update entry and create entry."""
    cfg = MagicMock()
    cfg.data = {cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"}
    cfg.options = {cf_mod.CONF_DEVICE_TRACKER_ENABLED: False}

    flow = cf_mod.OPNsenseOptionsFlow(cfg)
    flow.hass = ph_hass
    # ensure options flow has a handler and config_entries helpers available
    flow.handler = "opnsense"
    flow.hass.config_entries.async_get_known_entry = MagicMock(return_value=cfg)
    flow.hass.config_entries.async_update_entry = MagicMock()

    # populate internals to avoid Home Assistant property lookups in this unit test
    flow._config = dict(cfg.data)
    flow._options = dict(cfg.options)

    user_input = {cf_mod.CONF_SCAN_INTERVAL: 30}
    res = await flow.async_step_init(user_input=user_input)

    # should have called update_entry and returned create_entry
    flow.hass.config_entries.async_update_entry.assert_called()
    assert res["type"] == "create_entry"
    assert flow._options.get(cf_mod.CONF_SCAN_INTERVAL) == 30


@pytest.mark.asyncio
async def test_handle_user_input_raises_unknown_firmware_via_compare(monkeypatch, ph_hass):
    """If awesomeversion comparison fails inside _handle_user_input it should raise UnknownFirmware."""

    # create a fake client that returns a firmware string and minimal methods used
    class FakeClient:
        async def get_host_firmware_version(self):
            return "x"

        async def set_use_snake_case(self, initial=True):
            return None

        async def is_plugin_installed(self):
            return True

        async def get_system_info(self):
            return {"name": "r"}

        async def get_device_unique_id(self):
            return "devid"

    async def _get_client(user_input, hass):
        return FakeClient()

    monkeypatch.setattr(cf_mod, "_get_client", _get_client)

    # make _validate_firmware_version raise the AwesomeVersionCompareException used in the module
    def _raise_compare(v):
        raise cf_mod.awesomeversion.exceptions.AwesomeVersionCompareException

    monkeypatch.setattr(cf_mod, "_validate_firmware_version", _raise_compare)

    with pytest.raises(cf_mod.UnknownFirmware):
        await cf_mod._handle_user_input(
            hass=ph_hass,
            user_input={
                cf_mod.CONF_URL: "https://x",
                cf_mod.CONF_USERNAME: "u",
                cf_mod.CONF_PASSWORD: "p",
            },
            config_step="user",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc, expected_base",
    [
        (socket.gaierror("gai"), "cannot_connect"),
        (xmlrpc.client.ProtocolError("u", 500, "500 Server Error", {}), "cannot_connect"),
    ],
)
async def test_validate_input_maps_socket_and_protocol(monkeypatch, exc, expected_base):
    """validate_input should map socket.gaierror and non-redirect ProtocolError to cannot_connect."""

    async def _raiser(*args, **kwargs):
        raise exc

    monkeypatch.setattr(cf_mod, "_handle_user_input", _raiser)
    errors = {}
    res = await cf_mod.validate_input(
        hass=MagicMock(), user_input={}, config_step="user", errors=errors
    )
    assert res.get("base") == expected_base


@pytest.mark.asyncio
async def test_get_dt_entries_handles_empty_and_skips_empty_mac(
    monkeypatch, patch_async_create_clientsession, patch_cf_opnsense_client
):
    """_get_dt_entries should return selected devices when arp_table empty, and skip entries with empty mac."""

    class C1:
        async def get_arp_table(self, resolve_hostnames=True):
            return []

    class C2:
        async def get_arp_table(self, resolve_hostnames=True):
            return [
                {"mac": "", "hostname": "h", "ip": "1.2.3.4"},
                {"mac": "aa:bb:cc:dd:ee:ff", "hostname": "", "ip": "10.0.0.1"},
            ]

    # patch async_create_clientsession to avoid aiohttp use
    patch_async_create_clientsession(lambda *a, **k: MagicMock())

    patch_cf_opnsense_client(lambda *a, **k: C1())
    res1 = await cf_mod._get_dt_entries(
        hass=MagicMock(),
        config={cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"},
        selected_devices=["sel1"],
    )
    assert "sel1" in res1

    patch_cf_opnsense_client(lambda *a, **k: C2())
    res2 = await cf_mod._get_dt_entries(
        hass=MagicMock(),
        config={cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"},
        selected_devices=[],
    )
    # empty-mac entry is skipped; the good mac appears
    assert "aa:bb:cc:dd:ee:ff" in res2


@pytest.mark.asyncio
async def test_options_flow_init_routes_to_granular_and_device_tracker(monkeypatch, ph_hass):
    """async_step_init should route to granular sync when flag set and to device_tracker when enabled."""
    cfg = MagicMock()
    cfg.data = {cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"}
    cfg.options = {}
    # emulate a real config_entry handler presence
    cfg.handler = "opnsense"

    # Avoid patching OptionsFlow.config_entry (brittle across HA versions).
    # Instead set the flow handler and make hass.config_entries return our cfg
    # so OptionsFlow can resolve the config entry during tests.
    flow = cf_mod.OPNsenseOptionsFlow(cfg)
    flow.hass = ph_hass
    flow.handler = "opnsense"
    # Ensure the hass config_entries API returns our cfg when requested by the flow.
    ph_hass.config_entries.async_get_known_entry = MagicMock(return_value=cfg)

    # granular path
    flow.async_step_granular_sync = AsyncMock(return_value={"type": "create_entry"})
    _ = await flow.async_step_init(user_input={cf_mod.CONF_GRANULAR_SYNC_OPTIONS: True})
    flow.async_step_granular_sync.assert_awaited()

    # device tracker path
    flow.async_step_device_tracker = AsyncMock(return_value={"type": "create_entry"})
    _ = await flow.async_step_init(user_input={cf_mod.CONF_DEVICE_TRACKER_ENABLED: True})
    flow.async_step_device_tracker.assert_awaited()


@pytest.mark.asyncio
async def test_options_flow_granular_sync_calls_validate_and_updates(monkeypatch, ph_hass):
    """async_step_granular_sync should call validate_input and update entry when no errors."""
    cfg = MagicMock()
    cfg.data = {cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"}
    cfg.options = {cf_mod.CONF_DEVICE_TRACKER_ENABLED: False}

    flow = cf_mod.OPNsenseOptionsFlow(cfg)
    flow.hass = ph_hass
    flow.hass.config_entries.async_update_entry = MagicMock()

    # monkeypatch validate_input to return no errors
    async def fake_validate(hass, user_input, errors, **kwargs):
        return {}

    monkeypatch.setattr(cf_mod, "validate_input", fake_validate)

    # use an actual granular sync key present in the module
    gkey = next(iter(cf_mod.GRANULAR_SYNC_ITEMS))
    # populate internals so the flow method doesn't access Home Assistant internals
    flow._config = dict(cfg.data)
    flow._options = dict(cfg.options)
    user_input = {gkey: True}
    # set a handler and make async_get_known_entry return our cfg so the flow can access
    # config_entry and options during unit tests without Home Assistant internals.
    flow.handler = "opnsense"
    flow.hass.config_entries.async_get_known_entry = MagicMock(return_value=cfg)
    res = await flow.async_step_granular_sync(user_input=user_input)
    flow.hass.config_entries.async_update_entry.assert_called()
    assert res["type"] == "create_entry"


@pytest.mark.asyncio
async def test_device_tracker_shows_form_when_no_user_input(
    monkeypatch, make_config_entry, ph_hass
):
    """async_step_device_tracker should show form containing data_schema when called without user_input."""
    cfg = make_config_entry(
        data={cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"},
        options={cf_mod.CONF_DEVICES: ["11:22:33:44:55:66"]},
    )

    flow = cf_mod.OPNsenseOptionsFlow(cfg)
    flow.hass = ph_hass

    # monkeypatch _get_dt_entries to return an ordered dict-like mapping
    async def fake_get_dt_entries(hass, config, selected_devices):
        return {"11:22:33:44:55:66": "label1", "aa:bb:cc:dd:ee:ff": "label2"}

    monkeypatch.setattr(cf_mod, "_get_dt_entries", fake_get_dt_entries)

    # ensure internals are present so we don't trigger config_entry property lookup
    flow._config = dict(cfg.data)
    flow._options = dict(cfg.options)
    # set a handler and make async_get_known_entry return our cfg so the flow can access
    # config_entry and options during unit tests without Home Assistant internals.
    flow.handler = "opnsense"
    flow.hass.config_entries.async_get_known_entry = MagicMock(return_value=cfg)

    res = await flow.async_step_device_tracker(user_input=None)
    assert res["type"] == "form"
    assert "data_schema" in res


@pytest.mark.asyncio
async def test_options_flow_device_tracker_user_input(monkeypatch, make_config_entry, ph_hass):
    """When user submits manual devices, they should be parsed and saved to options."""
    # Build a fake config_entry using shared factory
    config_entry = make_config_entry(
        data={
            cf_mod.CONF_URL: "https://x",
            cf_mod.CONF_USERNAME: "u",
            cf_mod.CONF_PASSWORD: "p",
        },
        options={cf_mod.CONF_DEVICE_TRACKER_ENABLED: True, cf_mod.CONF_DEVICES: []},
    )

    flow = cf_mod.OPNsenseOptionsFlow(config_entry)
    flow.hass = ph_hass
    flow.hass.config_entries.async_update_entry = MagicMock()
    # make the flow aware of its handler so config_entry property works during tests
    flow.handler = "opnsense"
    flow.hass.config_entries.async_get_known_entry = MagicMock(return_value=config_entry)

    # emulate what async_step_init would do: populate _config and _options from entry
    flow._config = dict(config_entry.data)
    flow._options = dict(config_entry.options)

    user_input = {
        cf_mod.CONF_MANUAL_DEVICES: "aa:bb:cc:dd:ee:ff, bad, 11:22:33:44:55:66",
        cf_mod.CONF_DEVICES: ["11:22:33:44:55:66"],
    }

    result = await flow.async_step_device_tracker(user_input=user_input)

    # flow should have returned a create_entry
    assert result["type"] == "create_entry"

    # The flow should have parsed manual devices into _options
    assert cf_mod.CONF_DEVICES in flow._options
    assert "aa:bb:cc:dd:ee:ff" in flow._options[cf_mod.CONF_DEVICES]
    assert "11:22:33:44:55:66" in flow._options[cf_mod.CONF_DEVICES]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "granular_flag, config_step, expected_called",
    [
        # user step: no plugin check no matter the granular sync flag
        (True, "user", False),
        (False, "user", False),
        # granular_sync or reconfigure step: if granular sync is enabled, plugin check should happen
        (True, "granular_sync", True),
        (False, "granular_sync", False),
        (True, "reconfigure", True),
        (False, "reconfigure", False),
    ],
)
async def test_validate_input_user_respects_granular_flag_for_plugin_check(
    monkeypatch,
    granular_flag,
    config_step,
    expected_called,
    fake_flow_client,
    ph_hass,
    patch_async_create_clientsession,
    patch_cf_opnsense_client,
):
    """Plugin check not required for config step of user.

    Otherwise, plugin check is required if granular sync options is enabled
    """
    client_cls = fake_flow_client()
    patch_cf_opnsense_client(client_cls)

    # avoid real network sessions
    patch_async_create_clientsession(lambda *a, **k: MagicMock())

    user_input = {
        cf_mod.CONF_URL: "https://host.example",
        cf_mod.CONF_USERNAME: "user",
        cf_mod.CONF_PASSWORD: "pass",
        cf_mod.CONF_GRANULAR_SYNC_OPTIONS: granular_flag,
    }
    # Do not set granular sync items here; leave them absent so defaults apply

    # Create a real config flow and stub methods that interact with Home Assistant internals
    flow = cf_mod.OPNsenseConfigFlow()
    flow.hass = ph_hass

    async def _noop(*args, **kwargs):
        return None

    # Prevent base ConfigFlow methods from touching HA internals during unit test
    flow.async_set_unique_id = _noop
    flow._abort_if_unique_id_configured = lambda: None

    # Call the requested config step which will call validate_input internally
    if config_step == "user":
        await flow.async_step_user(user_input=user_input)
    elif config_step == "granular_sync":
        # populate internal config as if the user completed the first step
        flow._config = {
            cf_mod.CONF_URL: "https://host.example",
            cf_mod.CONF_USERNAME: "user",
            cf_mod.CONF_PASSWORD: "pass",
            cf_mod.CONF_GRANULAR_SYNC_OPTIONS: granular_flag,
        }
        await flow.async_step_granular_sync(user_input={})
    elif config_step == "reconfigure":
        # reconfigure should behave like granular_sync for plugin-check testing;
        # populate internal config and invoke the reconfigure step
        reconfigure_entry = MagicMock()
        reconfigure_entry.data = {
            cf_mod.CONF_URL: "https://host.example",
            cf_mod.CONF_USERNAME: "user",
            cf_mod.CONF_PASSWORD: "pass",
            cf_mod.CONF_GRANULAR_SYNC_OPTIONS: granular_flag,
        }
        # Monkeypatch the helper the config flow uses to get the reconfigure entry
        flow._get_reconfigure_entry = lambda: reconfigure_entry
        # Prevent HA internals from being accessed if the flow reaches update/abort paths
        flow.hass.config_entries = MagicMock()
        flow.hass.config_entries.async_update_entry = MagicMock()
        await flow.async_step_reconfigure(user_input={})
    else:
        raise ValueError(f"unknown config_step: {config_step}")

    # ensure client was instantiated
    assert client_cls.last_instance is not None
    # Check whether the plugin check was called according to expected behavior
    called_count = getattr(client_cls.last_instance, "_is_plugin_called", 0)
    if expected_called:
        assert called_count > 0
    else:
        assert called_count == 0


@pytest.mark.asyncio
async def test_async_step_user_granular_and_create_entry(monkeypatch, ph_hass):
    """Test async_step_user routes to granular when flag set and creates entry otherwise."""
    flow = cf_mod.OPNsenseConfigFlow()
    flow.hass = ph_hass

    # monkeypatch validate_input to return no errors
    async def _no_errors(*args, **kwargs):
        return {}

    monkeypatch.setattr(cf_mod, "validate_input", _no_errors)

    # If granular flag True, should call async_step_granular_sync; monkeypatch that
    flow.async_step_granular_sync = AsyncMock(return_value={"type": "create_entry"})

    # Prevent base class from mutating mappingproxy context during unit test
    flow.async_set_unique_id = AsyncMock()
    flow._abort_if_unique_id_configured = lambda: None

    user_input = {
        cf_mod.CONF_URL: "https://x",
        cf_mod.CONF_USERNAME: "u",
        cf_mod.CONF_PASSWORD: "p",
        cf_mod.CONF_DEVICE_UNIQUE_ID: "id",
        cf_mod.CONF_NAME: "name",
        cf_mod.CONF_GRANULAR_SYNC_OPTIONS: True,
    }

    res = await flow.async_step_user(user_input=user_input)
    assert res["type"] == "create_entry"

    # Now granular False -> should create entry directly
    user_input[cf_mod.CONF_GRANULAR_SYNC_OPTIONS] = False
    res2 = await flow.async_step_user(user_input=user_input)
    assert res2["type"] == "create_entry"


@pytest.mark.asyncio
async def test_async_step_reconfigure_calls_update_and_abort(monkeypatch, ph_hass):
    """Ensure reconfigure path calls update/reload/abort when validate_input ok."""
    flow = cf_mod.OPNsenseConfigFlow()
    flow.hass = ph_hass

    # create a fake reconfigure entry
    re = MagicMock()
    re.data = {cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"}

    flow._get_reconfigure_entry = lambda: re

    async def _no_errors(*args, **kwargs):
        return {}

    monkeypatch.setattr(cf_mod, "validate_input", _no_errors)

    # stub async_set_unique_id and _abort_if_unique_id_mismatch to avoid base class behavior
    flow.async_set_unique_id = AsyncMock()
    flow._abort_if_unique_id_mismatch = lambda: None

    # stub the method on the instance to return sentinel
    def _sentinel_update_reload_and_abort(*args, **kwargs):
        return {"type": "abort"}

    flow.async_update_reload_and_abort = _sentinel_update_reload_and_abort

    res = await flow.async_step_reconfigure(user_input={cf_mod.CONF_URL: "https://x"})
    assert res["type"] == "abort"


@pytest.mark.asyncio
async def test_handle_user_input_plugin_missing_and_missing_device(
    monkeypatch,
    ph_hass,
    fake_flow_client,
    patch_async_create_clientsession,
    patch_cf_opnsense_client,
):
    """Test _handle_user_input raising PluginMissing and MissingDeviceUniqueID using fixture."""
    # Prepare base user_input
    ui = {cf_mod.CONF_URL: "https://x", cf_mod.CONF_USERNAME: "u", cf_mod.CONF_PASSWORD: "p"}

    client_cls = fake_flow_client(plugin_installed=False)
    patch_cf_opnsense_client(client_cls)
    # Avoid real network sessions
    patch_async_create_clientsession(lambda *a, **k: MagicMock())

    # First: granular sync set and a plugin-required item -> PluginMissing
    ui_with_plugin = dict(ui)
    ui_with_plugin[cf_mod.CONF_GRANULAR_SYNC_OPTIONS] = True
    key = next(iter(cf_mod.SYNC_ITEMS_REQUIRING_PLUGIN))
    ui_with_plugin[key] = True

    with pytest.raises(cf_mod.PluginMissing):
        await cf_mod._handle_user_input(
            hass=ph_hass, user_input=ui_with_plugin, config_step="granular_sync"
        )

    # Now produce a client that returns empty device id to trigger MissingDeviceUniqueID
    client_cls2 = fake_flow_client(device_id="")
    patch_cf_opnsense_client(client_cls2)

    with pytest.raises(cf_mod.MissingDeviceUniqueID):
        await cf_mod._handle_user_input(hass=ph_hass, user_input=ui, config_step="user")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flow_type, require_plugin, expected_called",
    [
        ("config", True, True),
        ("config", False, False),
        ("options", True, True),
        ("options", False, False),
    ],
)
async def test_granular_sync_flow_plugin_check(
    monkeypatch,
    flow_type,
    require_plugin,
    expected_called,
    fake_flow_client,
    ph_hass,
    patch_async_create_clientsession,
    patch_cf_opnsense_client,
):
    """Test plugin check behavior when granular sync is enabled and granular items are set.

        For granular_sync step from both ConfigFlow and OptionsFlow:
    - If any SYNC_ITEMS_REQUIRING_PLUGIN is True -> is_plugin_installed should be called.
    - If none are True -> is_plugin_installed should NOT be called.
    """
    client_cls = fake_flow_client()
    patch_cf_opnsense_client(client_cls)
    patch_async_create_clientsession(lambda *a, **k: MagicMock())

    # Build a user_input payload for granular sync where items in SYNC_ITEMS_REQUIRING_PLUGIN are toggled
    # Start with all granular items as False, then set one plugin-required item True if require_plugin
    granular_input = dict.fromkeys(cf_mod.GRANULAR_SYNC_ITEMS, False)
    if require_plugin:
        # pick the first item that requires plugin
        plugin_item = list(cf_mod.SYNC_ITEMS_REQUIRING_PLUGIN)[0]
        granular_input[plugin_item] = True

    if flow_type == "config":
        # Prepare a config flow and populate internal config
        flow = cf_mod.OPNsenseConfigFlow()
        flow.hass = ph_hass
        flow._config = {
            cf_mod.CONF_URL: "https://host.example",
            cf_mod.CONF_USERNAME: "user",
            cf_mod.CONF_PASSWORD: "pass",
            cf_mod.CONF_GRANULAR_SYNC_OPTIONS: True,
        }
        # Call the granular sync step which will invoke validate_input -> _handle_user_input
        await flow.async_step_granular_sync(user_input=granular_input)
    else:
        # Options flow branch
        cfg = MagicMock()
        cfg.data = {
            cf_mod.CONF_URL: "https://host.example",
            cf_mod.CONF_USERNAME: "user",
            cf_mod.CONF_PASSWORD: "pass",
            cf_mod.CONF_GRANULAR_SYNC_OPTIONS: True,
        }
        cfg.options = {cf_mod.CONF_DEVICE_TRACKER_ENABLED: False}
        flow = cf_mod.OPNsenseOptionsFlow(cfg)
        flow.hass = ph_hass
        # emulate HA internals required by the options flow methods in tests
        flow.handler = "opnsense"
        flow.hass.config_entries = MagicMock()
        flow.hass.config_entries.async_get_known_entry = MagicMock(return_value=cfg)
        flow._config = dict(cfg.data)
        flow._options = dict(cfg.options)
        await flow.async_step_granular_sync(user_input=granular_input)

    # Check whether is_plugin_installed was invoked according to expectations
    assert client_cls.last_instance is not None
    called_count = getattr(client_cls.last_instance, "_is_plugin_called", 0)
    if expected_called:
        assert called_count > 0
    else:
        assert called_count == 0
