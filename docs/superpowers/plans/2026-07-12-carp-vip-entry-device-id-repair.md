# CARP VIP Entry and Device-ID Repair Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deliberately narrow, read-only CARP VIP config-entry type and replace the dead-end device-ID mismatch failure with a confirmed registry rebuild repair for normal OPNsense device entries.

**Architecture:** Existing entries and new normal-device entries remain bound to one physical OPNsense node and keep their current device-ID validation and full entity inventory. A new CARP VIP entry uses the config entry ID as its Home Assistant entity/device identity, calls `OPNsenseClient.validate(require_device_id=False)`, never stores or compares a physical `device_id`, polls only `get_system_info()` and `get_carp()`, and forwards only the sensor platform. Device-ID mismatch issues become fixable repairs that re-read the replacement ID, remove every registry entity and config-entry device association (including disabled entities), update the config entry to the replacement ID, and reload it.

**Tech Stack:** Python 3.14+, Home Assistant config entries/coordinators/entity and device registries/repairs, the aiopnsense release containing `validate(*, require_device_id: bool = True)`, pytest, pytest-homeassistant-custom-component, prek, ruff, mypy.

## Global Constraints

- Base all implementation work on `upstream/Remove-pyopnsense` at or after `e32a703448fd3db9a71f24e19640548777764c2a`; do not restore bundled `pyopnsense` code or compatibility branches.
- Implement and release the coordinated aiopnsense `optional-device-id-validation` branch first, following `docs/superpowers/plans/2026-07-12-optional-device-id-validation.md` in that repository. Do not guess a future aiopnsense version number in this plan.
- Existing config entries without `entry_type` are normal device entries. Do not bump `OPNsenseConfigFlow.VERSION` and do not add an entity migration.
- Normal device entries retain their existing config-entry unique ID, entity unique-ID prefix, full platform list, CARP sensors, and node-targeted CARP maintenance switch.
- CARP VIP entries are new entries, have no `device_unique_id`, use `ConfigEntry.entry_id` as the entity/device identifier prefix, forward only `Platform.SENSOR`, and expose no actions, switches, binary sensors, update entities, or device trackers.
- CARP VIP entities are limited to active responder name, aggregate CARP status, and VIP status keyed by synchronized `vhid + subnet`; do not expose CPU, memory, firmware, interfaces unrelated to CARP, gateways, services, VPN, DHCP, firewall/NAT, SMART, disks, filesystems, temperatures, or device trackers.
- CARP config-flow and reconfigure validation must create the client with `throw_errors=True`, call `OPNsenseClient.validate(require_device_id=False)`, then explicitly fetch and validate the system-information and CARP payloads.
- CARP runtime setup must match existing device-entry behavior: create the config-entry client with the default `throw_errors=False`, call `validate(require_device_id=False)` (which throws during validation and restores the default afterward), and let coordinator polling use the existing suppressed-error/unavailable-state behavior. Normal device entries continue calling `validate()` with its strict device-ID default.
- Do not claim that the VIP proves standby-node health or complete cluster health. User-facing copy must describe the responding node and CARP VIP state only.
- Device-ID repair is intentionally a registry rebuild, not an entity migration. It preserves connection settings and options but removes entity/device registry customizations and may change entity IDs; the confirmation and README must say so.
- Registry cleanup must remove disabled entities directly through the entity registry; it must never require enabling entities or editing Home Assistant `.storage`.
- Use repository tooling only through `./.venv/bin/python -m pytest` and `./.venv/bin/python -m prek run -a`; finish with the full suite and `git diff --check`.
- Do not create a pull request or push implementation commits unless separately requested.

---

## File Structure

- `custom_components/opnsense/const.py`: Define entry-type constants and the CARP-only platform list.
- `custom_components/opnsense/helpers.py`: Provide shared `is_carp_entry()` and stable `config_entry_identity()` helpers.
- `custom_components/opnsense/config_flow.py`: Add device-vs-CARP setup routing, explicit CARP validation, CARP reconfigure behavior, duplicate URL protection, and scan-interval-only CARP options.
- `custom_components/opnsense/manifest.json`: Pin the first published aiopnsense release containing `validate(require_device_id=False)` before implementing CARP entries.
- `custom_components/opnsense/__init__.py`: Route CARP entries through a narrow setup path and make device-ID mismatch issues fixable.
- `custom_components/opnsense/coordinator.py`: Allow an identity-less CARP coordinator mode with only `system_info` and `carp` categories and no mismatch check.
- `custom_components/opnsense/entity.py`: Use the shared config-entry identity so CARP entities use `entry_id` while device entries remain unchanged.
- `custom_components/opnsense/sensor.py`: Compile the CARP-only sensor set, including VIP keys based on VHID and subnet rather than node interface names.
- `custom_components/opnsense/services.py`: Exclude CARP entries from every domain action, including untargeted calls that normally fan out to all configured clients.
- `custom_components/opnsense/repairs.py`: Own mismatch issue creation and the confirmed hardware-replacement registry rebuild.
- `custom_components/opnsense/translations/en.json`: Add config menu, CARP validation, CARP options, entity, issue, and repair copy.
- `README.md`: Document normal-node, multi-node CARP, CARP VIP, hardware repair, entity scope, and limitations extensively.
- `tests/test_config_flow.py`: Cover device/CARP setup routing, validation, duplicates, reconfigure, and CARP options.
- `tests/test_init.py`: Cover unchanged device setup, narrow CARP setup, no device-ID calls, and fixable issue creation.
- `tests/test_coordinator.py`: Cover CARP categories and mismatch bypass without regressing device entries.
- `tests/test_entity.py`: Cover entry-ID fallback for CARP device/entity identity.
- `tests/test_sensor.py`: Cover exact CARP-only inventory and stable failover keys.
- `tests/test_services.py`: Prove CARP clients cannot be selected explicitly or through untargeted action fan-out.
- `tests/test_repairs.py`: Cover confirmation, stale/disabled registry cleanup, config/device-ID update, duplicate rejection, and reload scheduling.

---

### Task 1: Add explicit entry identity semantics

**Files:**
- Modify: `custom_components/opnsense/const.py:14-25`
- Modify: `custom_components/opnsense/helpers.py:17-20`
- Modify: `custom_components/opnsense/entity.py:71-96`
- Test: `tests/test_entity.py`
- Test: `tests/test_helpers.py`

**Interfaces:**
- Produces: `CONF_ENTRY_TYPE`, `ENTRY_TYPE_DEVICE`, `ENTRY_TYPE_CARP`, `CARP_PLATFORMS`, `is_carp_entry(config_entry: ConfigEntry) -> bool`, and `config_entry_identity(config_entry: ConfigEntry) -> str`.
- Consumes: Existing `CONF_DEVICE_UNIQUE_ID`, `ConfigEntry.entry_id`, and `Platform.SENSOR`.

- [ ] **Step 1: Write failing helper and entity identity tests**

Add tests proving old entries default to device mode, explicit CARP entries are detected, device identity remains its stored device ID, and CARP identity falls back to `entry_id`:

```python
def test_entry_type_and_identity_helpers(
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    device_entry = make_config_entry(
        entry_id="device-entry",
        data={CONF_DEVICE_UNIQUE_ID: "aa_bb_cc_dd_ee_ff"},
    )
    carp_entry = make_config_entry(
        entry_id="carp-entry",
        data={CONF_ENTRY_TYPE: ENTRY_TYPE_CARP},
    )

    assert is_carp_entry(device_entry) is False
    assert config_entry_identity(device_entry) == "aa_bb_cc_dd_ee_ff"
    assert is_carp_entry(carp_entry) is True
    assert config_entry_identity(carp_entry) == "carp-entry"
```

Extend the existing entity construction test with a CARP entry and assert the generated unique ID and device identifier both use `carp-entry`.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_helpers.py tests/test_entity.py -q
```

Expected: FAIL because the entry-type constants and helper functions do not exist and `OPNsenseBaseEntity` still indexes `CONF_DEVICE_UNIQUE_ID` directly.

- [ ] **Step 3: Add constants and identity helpers**

Add to `const.py`:

```python
CONF_ENTRY_TYPE = "entry_type"
ENTRY_TYPE_DEVICE = "device"
ENTRY_TYPE_CARP = "carp"

CARP_PLATFORMS: list[Platform] = [Platform.SENSOR]
```

Add to `helpers.py`:

```python
from .const import (
    CONF_DEVICE_UNIQUE_ID,
    CONF_ENTRY_TYPE,
    ENTRY_TYPE_CARP,
    ENTRY_TYPE_DEVICE,
)


def is_carp_entry(config_entry: ConfigEntry) -> bool:
    """Return whether a config entry represents a CARP virtual endpoint."""
    return config_entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_DEVICE) == ENTRY_TYPE_CARP


def config_entry_identity(config_entry: ConfigEntry) -> str:
    """Return the stable Home Assistant identity prefix for a config entry."""
    device_id = config_entry.data.get(CONF_DEVICE_UNIQUE_ID)
    return device_id if isinstance(device_id, str) and device_id else config_entry.entry_id
```

Update `OPNsenseBaseEntity.__init__()` to use `config_entry_identity(config_entry)` instead of direct `CONF_DEVICE_UNIQUE_ID` indexing. Keep the `_device_unique_id` attribute name for compatibility with existing entity code, but document that it is the config-entry identity prefix for CARP entries.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_helpers.py tests/test_entity.py -q
```

Expected: PASS, including all pre-existing device-entry identity assertions.

- [ ] **Step 5: Commit the identity boundary**

```bash
git add custom_components/opnsense/const.py custom_components/opnsense/helpers.py custom_components/opnsense/entity.py tests/test_helpers.py tests/test_entity.py
git commit -m "Add config entry identity modes"
```

---

### Task 2: Add the CARP-specific config and options flows

**Files:**
- Modify: `custom_components/opnsense/manifest.json:13-17`
- Modify: `custom_components/opnsense/config_flow.py:333-566`
- Modify: `custom_components/opnsense/config_flow.py:775-940`
- Modify: `custom_components/opnsense/config_flow.py:943-1011`
- Modify: `custom_components/opnsense/translations/en.json:3-113`
- Test: `tests/test_config_flow.py`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: `CONF_ENTRY_TYPE`, `ENTRY_TYPE_DEVICE`, `ENTRY_TYPE_CARP`, `is_carp_entry()`, the published aiopnsense `validate(*, require_device_id: bool = True)` contract, `create_opnsense_client(..., throw_errors=True)`, `get_host_firmware_version()`, `get_system_info()`, and `get_carp()`.
- Produces: `OPNsenseCarpNotConfiguredError`, `_validate_carp_client_details()`, `_build_carp_input_schema()`, `_build_carp_options_schema()`, `async_step_device()`, and `async_step_carp()`.

- [ ] **Step 1: Write failing config-flow tests for both entry types**

Add tests with a mocked aiopnsense client that assert:

```python
result = await flow.async_step_user()
assert result["type"] == "menu"
assert result["menu_options"] == ["device", "carp"]
```

For the existing device path, call `async_step_device(user_input)` and assert the created data includes `entry_type == "device"`, retains the physical `device_unique_id`, and still routes through granular sync when selected.

For the CARP path, provide a client returning:

```python
client.validate = AsyncMock(return_value=None)
client.get_host_firmware_version = AsyncMock(return_value="26.1.11")
client.get_system_info = AsyncMock(return_value={"name": "fw-a.example"})
client.get_carp = AsyncMock(
    return_value={
        "interfaces": [
            {
                "interface": "lan",
                "subnet": "192.0.2.1",
                "vhid": "1",
                "status": "MASTER",
            }
        ],
        "status_summary": {"state": "healthy", "vip_count": 1},
    }
)
```

Assert the CARP result creates an entry titled `fw-a.example CARP VIP`, stores `entry_type == "carp"`, stores normalized connection and firmware fields, and omits `device_unique_id` and `granular_sync_options`. Assert `client.validate.assert_awaited_once_with(require_device_id=False)` and `client.get_device_unique_id.assert_not_awaited()`.

Add negative tests for an empty `interfaces` list producing `errors["base"] == "carp_not_configured"`, `client.validate` raising `OPNsenseBelowMinFirmware` preserving `below_min_firmware`, and duplicate normalized URL aborting with `already_configured`.

Add reconfigure tests proving CARP reconfigure calls CARP validation, preserves `entry_type`, and never invokes `_abort_if_unique_id_mismatch()`. Add options tests proving a CARP entry sees only `scan_interval` and cannot route to granular-sync or device-tracker steps.

- [ ] **Step 2: Run the targeted config-flow tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_config_flow.py tests/test_integration.py -q
```

Expected: FAIL because `async_step_user()` is still the device form, there is no CARP validation path, and CARP options are not restricted.

- [ ] **Step 3: Pin the released aiopnsense validation contract**

After the aiopnsense branch is implemented and released, replace the current exact aiopnsense requirement in `custom_components/opnsense/manifest.json` with the exact published release containing `validate(require_device_id=False)`. Do not use a Git branch URL or guess a future version number.

Refresh the repository-local environment using the project dependency workflow, then verify the installed signature:

```bash
./.venv/bin/python -c "import inspect; from aiopnsense import OPNsenseClient; print(inspect.signature(OPNsenseClient.validate))"
```

Expected: `(self, *, require_device_id: bool = True) -> None`.

- [ ] **Step 4: Add CARP validation using the library opt-out**

Import the new entry constants and `is_carp_entry`. Define:

```python
class OPNsenseCarpNotConfiguredError(OPNsenseError):
    """Raised when an endpoint has no usable CARP VIP rows."""
```

Add this complete validation helper:

```python
async def _validate_carp_client_details(
    hass: HomeAssistant,
    user_input: MutableMapping[str, Any],
) -> None:
    """Validate and enrich a CARP VIP flow submission without a device-ID probe."""
    await _clean_and_parse_url(user_input)
    client = create_opnsense_client(
        hass=hass,
        url=user_input[CONF_URL],
        username=user_input[CONF_USERNAME],
        password=user_input[CONF_PASSWORD],
        verify_ssl=user_input.get(CONF_VERIFY_SSL),
        throw_errors=True,
    )
    try:
        await client.validate(require_device_id=False)
        firmware = await client.get_host_firmware_version()
        user_input[CONF_FIRMWARE_VERSION] = firmware
        system_info = await client.get_system_info()
        carp = await client.get_carp()
        carp_interfaces = carp.get("interfaces") if isinstance(carp, Mapping) else None
        if not isinstance(carp_interfaces, list) or not carp_interfaces:
            raise OPNsenseCarpNotConfiguredError("No CARP VIPs were returned")

        responder_name = system_info.get("name") if isinstance(system_info, Mapping) else None
        if not user_input.get(CONF_NAME):
            base_name = responder_name if isinstance(responder_name, str) else "OPNsense"
            user_input[CONF_NAME] = f"{base_name} CARP VIP"
        user_input[CONF_ENTRY_TYPE] = ENTRY_TYPE_CARP
        user_input.pop(CONF_DEVICE_UNIQUE_ID, None)
        user_input.pop(CONF_GRANULAR_SYNC_OPTIONS, None)
    finally:
        await client.async_close()
```

Extend `validate_input()` with a keyword-only `carp: bool = False`. Dispatch to `_validate_carp_client_details()` when true and map `OPNsenseCarpNotConfiguredError` to `carp_not_configured` before the generic `OPNsenseError` mapping.

- [ ] **Step 5: Add separate setup forms and CARP-only options**

Keep `_build_user_input_schema()` for normal devices and add a CARP connection schema containing URL, verify SSL, API key, API secret, and optional name but no granular-sync field. Add:

```python
def _build_carp_options_schema(
    user_input: Mapping[str, Any] | None,
    stored_options: Mapping[str, Any] | None,
) -> vol.Schema:
    """Build the scan-interval-only options schema for a CARP VIP entry."""
    defaults = {
        CONF_SCAN_INTERVAL: DEFAULT_SCAN_INTERVAL,
        **(stored_options or {}),
        **(user_input or {}),
    }
    scan_interval = _normalize_int_option(defaults[CONF_SCAN_INTERVAL], 10, 300)
    return vol.Schema(
        {
            vol.Optional(CONF_SCAN_INTERVAL, default=scan_interval): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=10,
                    max=300,
                    step=1,
                    unit_of_measurement="seconds",
                )
            )
        }
    )
```

Change `async_step_user()` to return:

```python
return self.async_show_menu(step_id="user", menu_options=["device", "carp"])
```

Move the existing device form behavior to `async_step_device()`, set `user_input[CONF_ENTRY_TYPE] = ENTRY_TYPE_DEVICE` before validation, and retain the existing device unique-ID and granular-sync logic unchanged.

Add `async_step_carp()` that validates with `carp=True`, calls `_async_abort_entries_match({CONF_URL: user_input[CONF_URL]})`, and creates the entry without setting a config-entry unique ID. Change YAML import to call `async_step_device()`.

In `async_step_reconfigure()`, branch on `is_carp_entry(reconfigure_entry)`: CARP entries use CARP validation and `async_update_and_abort()` without calling `async_set_unique_id()` or `_abort_if_unique_id_mismatch()`; normal device entries retain current behavior.

At the start of `OPNsenseOptionsFlow.async_step_init()`, add a CARP branch that normalizes and saves only `CONF_SCAN_INTERVAL`, then returns the CARP options form. Do not reuse `_create_options_entry()` for CARP because that helper also persists device-flow config mutations.

- [ ] **Step 6: Add config-flow translations**

Add menu labels `OPNsense device` and `CARP virtual IP`, a CARP form explaining that it is read-only and that each physical node should also be configured through its management address, `carp_not_configured`, and the scan-interval-only CARP options step. Keep the normal-device copy and existing translation keys intact.

- [ ] **Step 7: Run config-flow and integration tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_config_flow.py tests/test_integration.py -q
```

Expected: PASS with device setup behavior preserved and all CARP paths covered.

- [ ] **Step 8: Commit the config-flow boundary**

```bash
git add custom_components/opnsense/manifest.json custom_components/opnsense/config_flow.py custom_components/opnsense/translations/en.json tests/test_config_flow.py tests/test_integration.py
git commit -m "Add CARP VIP configuration flow"
```

---

### Task 3: Add CARP-only runtime coordination

**Files:**
- Modify: `custom_components/opnsense/__init__.py:70-79`
- Modify: `custom_components/opnsense/__init__.py:198-370`
- Modify: `custom_components/opnsense/coordinator.py:50-96`
- Modify: `custom_components/opnsense/coordinator.py:169-294`
- Test: `tests/test_init.py`
- Test: `tests/test_coordinator.py`

**Interfaces:**
- Consumes: `is_carp_entry()`, `config_entry_identity()`, `CARP_PLATFORMS`, and config-flow-validated CARP data.
- Produces: An `OPNsenseDataUpdateCoordinator` mode with `device_unique_id=None`, categories exactly `system_info` and `carp`, and no mismatch enforcement.

- [ ] **Step 1: Write failing coordinator tests**

Construct a CARP config entry with no `device_unique_id`, instantiate the coordinator with `device_unique_id=None`, and assert:

```python
assert coordinator._build_categories() == [
    {"function": "get_system_info", "state_key": "system_info"},
    {"function": "get_carp", "state_key": "carp"},
]
coordinator._state = {
    "system_info": {"name": "fw-b.example"},
    "carp": {"interfaces": [], "status_summary": {"state": "healthy"}},
}
assert await coordinator._check_device_unique_id() is True
client.get_device_unique_id.assert_not_awaited()
```

Retain the existing device-entry category and three-mismatch shutdown tests unchanged.

- [ ] **Step 2: Write failing CARP setup tests**

Add a setup test whose config entry contains `entry_type="carp"` and no device ID. Assert setup:

- creates the client from the config entry without overriding the existing `throw_errors=False` default;
- calls `client.validate(require_device_id=False)` exactly once and does not call `client.get_device_unique_id()`;
- creates one coordinator with `device_unique_id=None`;
- forwards exactly `[Platform.SENSOR]`;
- does not create a device-tracker coordinator;
- stores `device_unique_id=None` in runtime data.

Add an unchanged-device test assertion that normal entries still call validation, check the saved ID, and forward the existing platform list.

- [ ] **Step 3: Run targeted setup/coordinator tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_init.py tests/test_coordinator.py -q
```

Expected: FAIL because setup requires `CONF_DEVICE_UNIQUE_ID` and the coordinator always builds and enforces the device-ID category.

- [ ] **Step 4: Make coordinator device identity optional only for CARP entries**

Change the constructor annotation to `device_unique_id: str | None`, store the optional value, and make `_build_categories()` return the two CARP categories before building the existing device list:

```python
if is_carp_entry(self.config_entry):
    return [
        {"function": "get_system_info", "state_key": "system_info"},
        {"function": "get_carp", "state_key": "carp"},
    ]
```

At the start of `_check_device_unique_id()`, add:

```python
if self._device_unique_id is None and is_carp_entry(self.config_entry):
    return True
```

Do not weaken the normal-device missing or mismatched ID branches.

- [ ] **Step 5: Add a dedicated CARP setup helper and early route**

Add `_async_setup_carp_entry(hass, entry)` in `__init__.py`. It must:

1. Create the client through `create_opnsense_client_from_config_entry(hass=hass, config_entry=entry)` without a `throw_errors` override, exactly like normal device setup.
2. Call `await client.validate(require_device_id=False)` to preserve connection, authentication, and firmware validation without probing physical-device identity.
3. Create `OPNsenseDataUpdateCoordinator(..., device_unique_id=None)`.
4. Run the first refresh.
5. Store runtime data with `loaded_platforms=CARP_PLATFORMS.copy()` and no tracker coordinator.
6. Register the existing update listener, store the client in `hass.data`, and forward only the sensor platform.
7. Shut down/close resources on failure with the same ownership pattern as device setup.

Use the same validation-exception handling, runtime error-suppression behavior, and resource cleanup ownership as normal setup. Do not duplicate aiopnsense's firmware comparison in the CARP helper. Config flow has already validated the system-information and CARP payloads; subsequent coordinator failures follow the existing unavailable-state behavior used by device entries.

At the start of `async_setup_entry()`, add:

```python
if is_carp_entry(entry):
    return await _async_setup_carp_entry(hass, entry)
```

Leave the existing normal-device setup body structurally unchanged.

- [ ] **Step 6: Run targeted setup/coordinator tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_init.py tests/test_coordinator.py -q
```

Expected: PASS, with explicit assertions that CARP setup uses the default `throw_errors=False`, calls `validate(require_device_id=False)`, never calls the device-ID API, and device setup retains strict `validate()` behavior.

- [ ] **Step 7: Commit CARP runtime routing**

```bash
git add custom_components/opnsense/__init__.py custom_components/opnsense/coordinator.py tests/test_init.py tests/test_coordinator.py
git commit -m "Add narrow CARP VIP runtime"
```

---

### Task 4: Add the stable read-only CARP sensor inventory

**Files:**
- Modify: `custom_components/opnsense/sensor.py:919-1048`
- Modify: `custom_components/opnsense/sensor.py:1259-1299`
- Modify: `custom_components/opnsense/sensor.py:1674-1799`
- Modify: `custom_components/opnsense/translations/en.json`
- Test: `tests/test_sensor.py`

**Interfaces:**
- Consumes: CARP coordinator data shaped as `{"system_info": ..., "carp": {"interfaces": ..., "status_summary": ...}}` and `is_carp_entry()`.
- Produces: `OPNsenseCarpActiveResponderSensor`, `OPNsenseCarpVipSensor`, `_compile_carp_vip_sensors()`, and stable VIP keys `carp.vip.<vhid_slug>.<subnet_slug>`.

- [ ] **Step 1: Write failing exact-inventory and failover tests**

Create a CARP entry and a coordinator snapshot containing two VIP rows. Assert `async_setup_entry()` adds exactly:

```python
{
    "carp-entry_carp_active_responder",
    "carp-entry_carp_status_summary",
    "carp-entry_carp_vip_1_192_0_2_1",
    "carp-entry_carp_vip_2_198_51_100_1",
}
```

Adjust expected slug spelling to the repository's `slugify()` output in the test itself rather than hard-coding punctuation behavior.

Simulate failover by replacing `system_info.name` and changing the CARP row's physical `interface` from `igc0` to `ix0` while retaining VHID and subnet. Assert the active-responder value changes, the VIP sensor remains available, and its unique ID does not change.

Assert no telemetry, firmware, interface, gateway, DHCP, VPN, certificate, SMART, or temperature entity class is created for the CARP entry. Retain existing normal-device CARP sensor tests unchanged.

- [ ] **Step 2: Run the CARP sensor tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sensor.py -q
```

Expected: FAIL because CARP entries currently follow all default sync categories and interface sensor identity includes the node's interface name.

- [ ] **Step 3: Add active-responder and stable VIP sensor classes**

Add a fixed active-responder description with key `carp.active_responder`. Its update callback reads `system_info.name`, becomes unavailable for a missing/blank name, sets that name as the native value, and has no node telemetry attributes.

Add `_build_carp_vip_sensor_key(vhid: str, subnet: str) -> str`:

```python
def _build_carp_vip_sensor_key(vhid: str, subnet: str) -> str:
    """Build a node-independent CARP VIP key from synchronized values."""
    return f"carp.vip.{slugify(vhid.strip())}.{slugify(subnet.strip())}"
```

`_compile_carp_vip_sensors()` must skip rows without non-empty `vhid` and `subnet`, create names such as `CARP VIP 192.0.2.1 (VHID 1)`, and keep entities disabled by default. `OPNsenseCarpVipSensor` must locate the current row by normalized VHID and subnet, publish `status` as its native value, and expose interface, VHID, advertisement, subnet, description, and mode as attributes. Interface name must never participate in the unique ID.

- [ ] **Step 4: Branch sensor setup before normal category compilation**

Immediately after validating coordinator data in `sensor.async_setup_entry()`, add a CARP branch:

```python
if is_carp_entry(config_entry):
    entities = [
        _create_sensor(
            OPNsenseCarpActiveResponderSensor,
            config_entry,
            coordinator,
            SensorEntityDescription(
                key="carp.active_responder",
                name="Active CARP Responder",
                icon="mdi:server-network",
                entity_registry_enabled_default=True,
            ),
        )
    ]
    entities.extend(await _compile_carp_status_sensor(config_entry, coordinator, state))
    entities.extend(await _compile_carp_vip_sensors(config_entry, coordinator, state))
    async_add_entities(entities)
    return
```

The existing normal-device setup path remains below this early return and continues compiling its current CARP interface sensors and every selected category.

- [ ] **Step 5: Add entity translations and run tests**

Add entity translations for active responder, CARP status, and CARP VIP. Use wording that says `responder` and `VIP`; do not use `cluster healthy` or imply standby visibility.

Run:

```bash
./.venv/bin/python -m pytest tests/test_sensor.py tests/test_entity.py -q
```

Expected: PASS, including failover with a changed physical interface name and unchanged VIP entity IDs.

- [ ] **Step 6: Commit the CARP sensor inventory**

```bash
git add custom_components/opnsense/sensor.py custom_components/opnsense/translations/en.json tests/test_sensor.py tests/test_entity.py
git commit -m "Add read-only CARP VIP sensors"
```

---

### Task 5: Enforce read-only CARP entries across domain actions

**Files:**
- Modify: `custom_components/opnsense/services.py:281-321`
- Test: `tests/test_services.py`

**Interfaces:**
- Consumes: `is_carp_entry(config_entry: ConfigEntry) -> bool`, `hass.data[DOMAIN]`, and `hass.config_entries.async_get_entry()`.
- Produces: Service target resolution that returns only normal-device clients.

- [ ] **Step 1: Write failing read-only service tests**

Add three tests around `_get_clients()` and one representative action handler:

1. With one normal-device entry and one CARP entry loaded, an untargeted call returns only the normal-device client.
2. Explicitly targeting the CARP device or one of its sensor entities raises the existing localized `no_target_clients` `ServiceValidationError`.
3. With only a CARP entry loaded, an untargeted action resolves no clients and performs no backend call.
4. `system_reboot`, used as the representative destructive action, is never awaited on the CARP client in any of the above cases.

Use real `MockConfigEntry` records added to `hass.config_entries` so the tests exercise entry-type lookup instead of relying on mock truthiness.

- [ ] **Step 2: Run service tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_services.py -q
```

Expected: FAIL because `_get_clients()` currently returns every client in `hass.data[DOMAIN]`, and its single-client fast path can return a CARP client directly.

- [ ] **Step 3: Filter client mappings before every target-resolution branch**

Import `is_carp_entry` and build the eligible mapping before the existing first-entry and target checks:

```python
service_clients: dict[str, OPNsenseServiceClient] = {}
for entry_id, client in hass.data[DOMAIN].items():
    config_entry = hass.config_entries.async_get_entry(entry_id)
    if config_entry is None or is_carp_entry(config_entry):
        continue
    service_clients[entry_id] = client
```

Use `service_clients`, not the raw domain mapping, for the single-client fast path and final iteration. Preserve current explicit-target error behavior:

```python
if not service_clients:
    if opndevice_id or opnentity_id:
        raise _service_validation_error(_TRANSLATION_KEY_NO_TARGET_CLIENTS)
    return []

first_entry_id = next(iter(service_clients))
if len(service_clients) == 1 and not opndevice_id and not opnentity_id:
    return [service_clients[first_entry_id]]
```

Do not weaken entity/device registry target resolution. A CARP target may resolve to a config-entry ID, but because that ID is absent from `service_clients`, the existing explicit-target no-client check must raise.

- [ ] **Step 4: Run service and setup tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_services.py tests/test_init.py -q
```

Expected: PASS, proving that the CARP client may remain in runtime storage for polling while all actions exclude it.

- [ ] **Step 5: Commit read-only action enforcement**

```bash
git add custom_components/opnsense/services.py tests/test_services.py
git commit -m "Exclude CARP entries from actions"
```

---

### Task 6: Replace device-ID dead ends with a confirmed registry rebuild repair

**Files:**
- Create: `custom_components/opnsense/repairs.py`
- Modify: `custom_components/opnsense/__init__.py:250-275`
- Modify: `custom_components/opnsense/coordinator.py:251-294`
- Modify: `custom_components/opnsense/translations/en.json:115-128`
- Create: `tests/test_repairs.py`
- Modify: `tests/test_init.py`
- Modify: `tests/test_coordinator.py`

**Interfaces:**
- Produces: `async_create_device_id_mismatch_issue(hass, config_entry, observed_device_id)`, `DeviceIDMismatchRepairFlow`, and Home Assistant `async_create_fix_flow()`.
- Consumes: `create_opnsense_client_from_config_entry(..., throw_errors=True)`, entity/device registries, config-entry unload/update/reload APIs, and current observed device ID.

- [ ] **Step 1: Write failing issue-creation tests**

Change startup and coordinator mismatch tests to assert the issue is:

```python
assert issue_kwargs["is_fixable"] is True
assert issue_kwargs["issue_id"] == f"{entry.entry_id}_device_id_mismatched"
assert issue_kwargs["data"] == {
    "entry_id": entry.entry_id,
    "old_device_id": "dev1",
    "new_device_id": "other",
}
assert issue_kwargs["translation_placeholders"] == {
    "entry_title": entry.title,
    "old_device_id": "dev1",
    "new_device_id": "other",
}
```

Assert normal startup still returns `False` before forwarding platforms, and the runtime coordinator still shuts down only after three consecutive mismatches.

- [ ] **Step 2: Write failing repair-flow tests**

Create `tests/test_repairs.py` with tests for:

1. Initial flow renders a confirmation containing old/new IDs and a destructive-rebuild warning.
2. Confirmation re-probes the current device ID with a strict client.
3. A duplicate config-entry unique ID aborts before unload or registry deletion.
4. A loaded entry is unloaded before cleanup.
5. Every registry entity for the config entry is removed, including an entry whose `disabled_by` is integration/user.
6. Every associated device uses `async_update_device(device.id, remove_config_entry_id=entry.entry_id)`, preserving devices shared with another config entry.
7. Entry data and config-entry unique ID change to the replacement ID while connection data and options remain unchanged.
8. Reload is scheduled and the repair issue is deleted.
9. A removed config entry aborts cleanly without mutations.
10. A CARP entry is rejected by the repair flow because CARP entries never create device-ID issues.

- [ ] **Step 3: Run repair and mismatch tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_repairs.py tests/test_init.py tests/test_coordinator.py -q
```

Expected: FAIL because the issue is not fixable and `repairs.py` does not exist.

- [ ] **Step 4: Implement centralized fixable issue creation**

Create this public helper in `repairs.py` and call it from both startup and coordinator mismatch paths:

```python
def async_create_device_id_mismatch_issue(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    observed_device_id: str,
) -> None:
    """Create a fixable hardware-replacement issue for a normal device entry."""
    old_device_id = config_entry.data[CONF_DEVICE_UNIQUE_ID]
    ir.async_create_issue(
        hass=hass,
        domain=DOMAIN,
        issue_id=f"{config_entry.entry_id}_device_id_mismatched",
        is_fixable=True,
        is_persistent=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="device_id_mismatched",
        translation_placeholders={
            "entry_title": config_entry.title,
            "old_device_id": old_device_id,
            "new_device_id": observed_device_id,
        },
        data={
            "entry_id": config_entry.entry_id,
            "old_device_id": old_device_id,
            "new_device_id": observed_device_id,
        },
    )
```

Remove duplicated issue construction but keep the existing stop-before-platforms and three-consecutive-runtime-mismatch behavior.

- [ ] **Step 5: Implement the complete repair flow**

Implement `DeviceIDMismatchRepairFlow` with these ordered mutations after confirmation:

```python
entry = self.hass.config_entries.async_get_entry(self._entry_id)
if entry is None or is_carp_entry(entry):
    return self.async_abort(reason="entry_not_found")

client = create_opnsense_client_from_config_entry(
    hass=self.hass,
    config_entry=entry,
    throw_errors=True,
)
try:
    observed_device_id = await client.get_device_unique_id()
finally:
    await client.async_close()

if not isinstance(observed_device_id, str) or not observed_device_id:
    return self.async_abort(reason="cannot_connect")

duplicate = next(
    (
        candidate
        for candidate in self.hass.config_entries.async_entries(DOMAIN)
        if candidate.entry_id != entry.entry_id
        and candidate.unique_id == observed_device_id
    ),
    None,
)
if duplicate is not None:
    return self.async_abort(reason="already_configured")

if entry.state is ConfigEntryState.LOADED:
    if not await self.hass.config_entries.async_unload(entry.entry_id):
        return self.async_abort(reason="cannot_unload")

entity_registry = er.async_get(self.hass)
for entity in er.async_entries_for_config_entry(entity_registry, entry.entry_id):
    entity_registry.async_remove(entity.entity_id)

device_registry = dr.async_get(self.hass)
for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id):
    device_registry.async_update_device(
        device.id,
        remove_config_entry_id=entry.entry_id,
    )

new_data = {**entry.data, CONF_DEVICE_UNIQUE_ID: observed_device_id}
self.hass.config_entries.async_update_entry(
    entry,
    data=new_data,
    unique_id=observed_device_id,
)
ir.async_delete_issue(
    self.hass,
    DOMAIN,
    f"{entry.entry_id}_device_id_mismatched",
)
self.hass.config_entries.async_schedule_reload(entry.entry_id)
return self.async_create_entry(data={})
```

`async_step_init()` must load the issue placeholders and delegate to a zero-field `confirm` form. `async_create_fix_flow()` must validate the issue suffix and data types before constructing the flow; unknown issues return `ConfirmRepairFlow()` or raise `ValueError` consistently with current Home Assistant examples.

- [ ] **Step 6: Add repair translations with explicit data-loss scope**

Update the issue description to stop saying uninstall/reinstall is required. Add `fix_flow.step.confirm` text that says the repair will:

- remove all current entities and devices for this config entry, including disabled entities;
- preserve URL, credentials, and options;
- rebuild entities from the replacement hardware;
- discard entity registry names, enabled/disabled selections, areas, and other customizations;
- potentially require dashboard and automation updates if recreated entity IDs differ.

Add abort copy for missing entry, connection failure, duplicate device, and unload failure.

- [ ] **Step 7: Run repair and mismatch tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_repairs.py tests/test_init.py tests/test_coordinator.py -q
```

Expected: PASS, including direct deletion of disabled entity-registry entries without enabling or restarting Home Assistant first.

- [ ] **Step 8: Commit the hardware repair**

```bash
git add custom_components/opnsense/repairs.py custom_components/opnsense/__init__.py custom_components/opnsense/coordinator.py custom_components/opnsense/translations/en.json tests/test_repairs.py tests/test_init.py tests/test_coordinator.py
git commit -m "Add device ID replacement repair"
```

---

### Task 7: Document the two entry models and operational consequences

**Files:**
- Modify: `README.md`
- Modify: `custom_components/opnsense/translations/en.json`

**Interfaces:**
- Consumes: Final config-flow labels, exact CARP sensor inventory, action exclusion, and repair behavior from Tasks 2-6.
- Produces: User-facing installation, multi-device/CARP, hardware-replacement, entity, and limitation documentation.

- [ ] **Step 1: Expand the README table of contents and configuration model**

Add sections for:

- `OPNsense Device Entry`
- `CARP VIP Entry`
- `Recommended CARP Topology`
- `CARP VIP Entities and Limitations`
- `Replacing OPNsense Hardware`

Include this concrete topology:

```text
Node A management URL -> full OPNsense device entry
Node B management URL -> full OPNsense device entry
CARP virtual URL      -> optional read-only CARP VIP entry
```

State that existing entries remain device entries automatically and require no migration.

- [ ] **Step 2: Document exact CARP scope and non-goals**

Document that the CARP VIP entry shows only:

- active responding node name;
- aggregate CARP status from that responder;
- CARP VIP state keyed by VHID and subnet.

Explicitly state that it does not expose node hardware telemetry, firmware/update entities, general interfaces, services, gateways, VPN, DHCP, firewall/NAT, SMART, disks, temperatures, device trackers, actions, or switches. Explain that the VIP cannot prove standby-node health and that node management entries are required for complete monitoring.

Explain why the persistent CARP maintenance switch remains on physical-node entries: enabling maintenance through the VIP can move the VIP, causing a later disable call to reach the other node.

- [ ] **Step 3: Replace the hardware-change known issue with the repair workflow**

Explain why hardware replacement is treated as an entity-inventory boundary, why stale disabled entities are dangerous, and how the fixable repair rebuilds the registry. List exactly what is preserved and what is lost, matching the confirmation translation.

- [ ] **Step 4: Review translation wording against README terminology**

Use only these terms consistently:

- `OPNsense device entry`
- `CARP VIP entry`
- `active responder`
- `CARP VIP status`
- `replacement hardware`
- `rebuild entities`

Remove `cluster healthy` claims and do not call a physical device ID a cluster identity.

- [ ] **Step 5: Run documentation and translation lint**

Run:

```bash
./.venv/bin/python -m prek run -a
git diff --check
```

Expected: PASS with valid JSON, Markdown formatting, Ruff, and mypy.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md custom_components/opnsense/translations/en.json
git commit -m "Document CARP VIP entries and hardware repair"
```

---

### Task 8: Full regression verification

**Files:**
- Verify: `custom_components/opnsense`
- Verify: `custom_components/opnsense/manifest.json`
- Verify: `tests`
- Verify: `README.md`

**Interfaces:**
- Consumes: All prior tasks.
- Produces: Evidence that normal device behavior, new CARP behavior, repair cleanup, typing, and formatting all pass together.

- [ ] **Step 1: Run focused feature tests as one gate**

Run:

```bash
./.venv/bin/python -m pytest tests/test_config_flow.py tests/test_init.py tests/test_coordinator.py tests/test_entity.py tests/test_sensor.py tests/test_services.py tests/test_repairs.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete pytest suite**

Run:

```bash
./.venv/bin/python -m pytest
```

Expected: PASS with no skipped/failing tests introduced by this feature.

- [ ] **Step 3: Run all repository checks**

Run:

```bash
./.venv/bin/python -m prek run -a
git diff --check
```

Expected: PASS.

- [ ] **Step 4: Verify branch scope and forbidden changes**

Run:

```bash
git diff --stat upstream/Remove-pyopnsense...HEAD
git diff --name-only upstream/Remove-pyopnsense...HEAD
rg -n "pyopnsense" custom_components tests README.md
```

Expected: the diff contains only the hass-opnsense integration, tests, documentation, and aiopnsense dependency pin described above; no bundled pyopnsense files are restored; `rg` finds only intentional historical documentation if any.

- [ ] **Step 5: Confirm no entity migration was added**

Run:

```bash
git diff upstream/Remove-pyopnsense...HEAD -- custom_components/opnsense/__init__.py custom_components/opnsense/config_flow.py | rg "async_migrate_entry|VERSION ="
```

Expected: no new migration stage and `OPNsenseConfigFlow.VERSION` remains `5`.

- [ ] **Step 6: Record final status without pushing**

Run:

```bash
git status --short --branch
git branch -vv
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

Expected: the feature branch tracks `origin/carp-vip-entry-device-id-repair-remove-pyopnsense`. Do not push implementation commits without a separate explicit request.

---

## Self-Review Checklist

- Spec coverage: Tasks 1-4 cover entry identity, configuration, runtime, and the intentionally narrow CARP inventory; Task 5 enforces read-only behavior across domain actions; Task 6 covers the replacement-hardware repair and disabled entity cleanup; Task 7 covers extensive README and UI copy; Task 8 covers all repository gates.
- No migration ambiguity: missing `entry_type` means device, config-flow version stays 5, device entity IDs remain unchanged, CARP entries are new, and hardware repair is an explicit destructive rebuild.
- aiopnsense boundary: the coordinated companion branch adds only the backward-compatible `require_device_id` validation option; hass-opnsense pins its published release, uses the opt-out only for CARP entries, and does not duplicate firmware comparisons.
- Error propagation: CARP config-flow validation uses `throw_errors=True` for the post-validation payload requests, while runtime clients retain the same `throw_errors=False` coordinator semantics as existing device entries.
- CARP safety: no switches/actions are forwarded, persistent maintenance remains node-targeted, and VIP keys exclude physical interface names.
- Repair safety: current replacement ID is re-read, duplicates and unload failures abort before cleanup, disabled entries are removed directly, and shared device records lose only this config-entry association.
- Type consistency: `device_unique_id: str | None` is accepted only by the coordinator; normal entries still pass `str`, CARP entries pass `None`, and entity identity always resolves to `str` through `config_entry_identity()`.
- Placeholder scan: the plan contains no deferred requirements; every implementation task identifies exact functions, tests, commands, and expected outcomes.
