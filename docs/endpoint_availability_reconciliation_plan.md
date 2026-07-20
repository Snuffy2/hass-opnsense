# Endpoint Availability and Optional-Entity Reconciliation Contract

## Implementation Status

The code contract is implemented on coordinated aiopnsense and hass-opnsense
feature branches, but it is not yet released or pinned in hass-opnsense.
aiopnsense classifies optional category results as
`available`, `pending`, `missing`, `transient`, or `malformed` and reports an
explicit `authoritative` bit. hass-opnsense unwraps result data for the existing
entity state contract while retaining the result as coordinator sidecar state.
Legacy clients remain readable but are always `pending` and non-authoritative.
The consumer contract includes `get_nut_ups_status_result() ->
CategoryResult[dict]` and per-device `get_smart_info_result(device,
info_type="a") -> CategoryResult[dict]`; the existing flattened getters remain
available only as compatibility fallbacks.

Branch-local contract tests exercise these public method names and result
shapes. Cross-repository validation against an installed, published aiopnsense
artifact remains pending until a real release exists; no version is guessed and
no dependency pin is changed before that gate passes.

Device-ID repair reconciliation records platform completion separately from
category authority. Stale registry entities are removed only inside a category
whose latest normalized inventory is both structurally complete and
authoritative. Missing, transient, malformed, legacy, and unknown scopes are
preserved, including tracker-device relationships. Runtime additions use stable
fingerprints derived only from validated normalized inventory identities.

## Purpose

Make optional OPNsense API capabilities respond predictably when plugins or
endpoints appear, disappear, or temporarily fail while Home Assistant is
running. The design must avoid probing every optional endpoint on every
coordinator update, must not turn router-wide outages into long-lived
per-endpoint absences, and must keep Home Assistant entity availability and
creation aligned with the capabilities aiopnsense actually observes.

This requires coordinated changes in both repositories:

- **aiopnsense** owns HTTP response classification, endpoint availability
  caching, and optional-endpoint results.
- **hass-opnsense** owns coordinator behavior, entity availability, and entity
  creation or reconciliation when an optional capability changes.

## Current Behavior and Gaps

- Endpoint availability results use a shared six-hour TTL.
- A cached positive result can outlive an endpoint. The subsequent real request
  may repeatedly return 404 while the availability check continues returning
  the stale cached value.
- A cached negative result delays recognition of a newly installed plugin for
  up to six hours.
- Ordinary transport requests do not feed successful or 404 observations back
  into the availability cache.
- The transport logs a failed optional request before callers can classify an
  expected 404 as a capability change.
- hass-opnsense compiles entities during platform setup. Coordinator data that
  gains a new optional capability later does not itself add entities.
- hass-opnsense currently polls NUT UPS status through its separately
  configurable granular sync option, which follows the integration's existing
  default-enabled sync behavior. It exposes three fixed, disabled-by-default sensors. Those sensors
  are pre-created even when the initial payload is missing or empty so endpoint
  recovery does not require an integration reload.

## Desired Semantics

### Cache observations

1. Use method-aware, exact-path cache keys for every observation.
2. Classify core health and correlated 404s before mutating any per-endpoint
   cache entry. Router-wide or correlated failures preserve existing positive,
   negative, and pending state and their timestamps.
3. A successful real request refreshes an existing positive availability entry.
4. Only an isolated 404 from an explicitly optional endpoint invalidates its
   positive entry immediately.
5. The next access performs an actual probe. A confirmed isolated optional 404
   creates a negative entry with a shorter TTL than a positive entry.
6. Timeouts, connection errors, 401/403 responses, 429 responses, and 5xx
   responses do not change endpoint availability. They describe transport,
   authentication, authorization, throttling, or service health—not route
   existence.
7. Never infer a base endpoint from a resource-specific or parameterized path.
   Derived paths may affect a registered probe key only through an explicit
   mapping.
8. Do not populate the availability cache for every successful HTTP request.
   Refresh only keys already registered by endpoint probing or by an explicit
   optional-endpoint declaration.

### Positive and negative TTLs

- Keep a relatively long positive TTL as a fallback for endpoints that are
  probed but not subsequently fetched.
- Introduce a separately configurable negative TTL. Start with five minutes,
  subject to test and live-observation results.
- A successful real request continuously refreshes the positive observation, so
  active endpoints should rarely need a dedicated periodic probe.
- A persistently missing plugin should cost at most one probe per negative TTL,
  not one probe per coordinator update.

### Router-wide failures

- Treat correlated failures of known core endpoints as router or transport
  health failures, not as evidence that many optional endpoints disappeared.
- A global timeout, connection failure, or 5xx outage naturally leaves endpoint
  availability unchanged.
- If a proxy or router-wide failure returns 404 for many unrelated paths, use a
  core-health signal or failure correlation before committing negative
  per-endpoint state.
- Keep this guard bounded: it must not conceal an isolated optional-plugin 404.

### Logging and public behavior

- The first 404 from an optional endpoint should be a debug or concise warning
  describing a capability transition, not a generic transport error.
- Unexpected 404s from required or resource-specific endpoints remain errors.
- Do not expose internal cache objects through the public aiopnsense API.
- Preserve existing normalized return shapes. For example, missing Speedtest
  remains `{"available": false}`.

## Implementation Work

### Phase 1: aiopnsense cache contract

Create a companion aiopnsense branch from its latest `origin/main`.

1. Centralize construction of method-aware endpoint cache keys.
2. Add private helpers to:
   - refresh a registered positive observation;
   - invalidate an exact registered observation;
   - store a confirmed negative probe with the negative TTL;
   - inspect whether a cached entry is positive, negative, expired, or absent.
3. Separate positive and negative TTL configuration in `const.py` and client
   initialization.
4. Keep `force_refresh` behavior for validation, diagnostics, and targeted
   recovery.
5. Ensure concurrent requests cannot leave a newer observation overwritten by
   an older result. The cache contract must use per-key synchronization or an
   atomic version/order check so stale writes are rejected unconditionally.

### Phase 2: aiopnsense status-aware optional transport

1. Add a private status-aware request path for optional endpoints rather than
   teaching all generic GET and POST calls that every 404 means absence.
2. Carry enough response information to distinguish:
   - success;
   - confirmed optional-endpoint 404;
   - non-404 HTTP failure;
   - transport failure;
   - malformed successful payload.
3. Classify core health and correlated 404s before invoking any per-endpoint
   cache helper. Preserve all existing state and timestamps for router-wide or
   correlated failures; feed success and only isolated confirmed optional 404
   observations into the cache helpers.
4. Migrate optional endpoint callers incrementally, beginning with Speedtest and
   NUT. Do not alter mutating endpoint behavior as part of this work.
5. Retain exact-path behavior for Speedtest `showlog` and `showstat`. Do not let a
   parameterized request invalidate an unrelated probe key.

### Phase 3: hass-opnsense coordinator consumption (branch implemented; release pending)

1. After the companion release is available, update the aiopnsense dependency
   pin in both `manifest.json` and `pyproject.toml`, plus the aiopnsense pin used
   by prek configuration. This remains a post-release step; no unreleased or
   guessed stable version is pinned by this implementation.
   The pin gate is explicit: do not change any of these three files until a real
   aiopnsense release containing both public result methods exists and its
   published artifact passes the cross-repository contract tests.
2. Keep transient transport failures distinct from an optional capability being
   absent:
   - transient failure: preserve the category contract and mark affected
     entities unavailable for that update according to existing coordinator
     policy;
   - confirmed absence: return the category's normalized unavailable shape;
   - recovery: resume on the first successful poll after the cache allows a
     retry.
3. Verify that one optional category failing does not fail or erase unrelated
   coordinator categories.
4. NUT remains a separately configurable, default-enabled granular sync
   category. Its result sidecar, not a flattened empty mapping, supplies repair
   authority. SMART sensor metrics require an authoritative list plus every
   applicable per-device detail result to be authoritative and schema-complete
   before stale metric sensors may be deleted. SMART status binary sensors
   depend only on the authoritative, schema-complete SMART list.
5. Fetch per-device SMART details with controlled concurrency and one fixed
   category deadline. Preserve completed healthy details, cancel unfinished
   calls as transient and non-authoritative, and continue later categories
   without accumulating one timeout per disk.

### Phase 4: hass-opnsense entity lifecycle (branch implemented; release pending)

Choose entity behavior based on whether the optional capability has a static or
dynamic entity schema.

1. For fixed schemas such as Speedtest and NUT metrics, create the configured
   entities regardless of whether the plugin is
   currently installed. Keep them unavailable until valid coordinator data
   arrives. This avoids needing runtime platform re-entry when the plugin is
   installed later.
2. For inventory-driven schemas, add bounded reconciliation that can add newly
   discovered entities without treating incomplete or transient inventory as
   authoritative for deletion.
3. Continue using `record_desired_entities(..., None)` when a dynamic inventory
   is incomplete so a temporary failure cannot remove registry entries.
4. Do not automatically delete entities when a plugin disappears. Existing
   entities become unavailable and retain their registry identity in case the
   plugin returns.
5. If runtime entity creation is added, make it idempotent and ensure repeated
   coordinator updates cannot add duplicate entities or listeners.

## Required Scenario Tests

### Speedtest removed while Home Assistant is running

- Begin with cached-positive `showlog` and `showstat` observations.
- Return 404 from the next real request.
- Establish that core health is intact and the 404 is isolated before asserting
  that the matching positive entry is invalidated and classified as optional
  disappearance.
- On the next access, confirm one real probe stores a short-lived negative
  result.
- Assert later polls skip payload requests while the negative entry is fresh.
- Assert existing Speedtest entities become unavailable but remain registered.

### Router-wide five-minute transient failure

- Cover timeouts, connection errors, and 5xx responses.
- Assert endpoint availability entries are unchanged.
- Assert recovery occurs on the first successful update after service returns.
- Cover correlated 404 responses from core and optional endpoints and assert
  they are classified as global health failure before cache mutation rather
  than independent plugin removals. Assert positive, negative, and pending
  state and all associated timestamps remain unchanged.

### Speedtest-only five-minute transient failure

- For timeout and 5xx failures, assert the positive cache remains intact and
  recovery occurs on the first later success.
- For 404 responses, first establish healthy core endpoints and no correlated
  failures. Then assert invalidation, negative probing, skipped requests during
  the negative TTL, and recovery no later than one negative TTL after the
  endpoint returns.
- Assert unrelated categories continue updating.

### Optional plugin installed while Home Assistant is running

- Begin with a confirmed negative cache entry.
- Install or simulate the endpoint before the negative TTL expires.
- Assert no request occurs until expiry unless explicitly force-refreshed.
- After expiry, assert the probe succeeds, the payload is fetched, and positive
  observations are refreshed by later real requests.
- For a fixed-schema hass-opnsense category, assert pre-created unavailable
  entities become available without an integration reload.
- For NUT specifically, keep backend detection tests separate from
  hass-opnsense category/entity recovery tests.

## Validation

### aiopnsense

- Add focused cache and transport tests covering GET and POST method-aware keys,
  success refresh, exact 404 invalidation, negative TTL expiry, force refresh,
  malformed payloads, concurrency, and non-404 failures.
- Add a deterministic controlled-interleaving regression in which an older
  observation completes after a newer one and prove it cannot overwrite the
  newer cache state or timestamp.
- Add Speedtest and NUT behavioral regressions at their real caller seams.
- Run the full pytest suite and `prek run --all-files`.

### hass-opnsense

- Mock aiopnsense behavior only at the integration boundary.
- Add coordinator tests for absent, transiently failing, recovered, and newly
  available optional categories.
- Add sensor setup/update tests for fixed-schema entities created unavailable
  and later becoming available.
- Add entity-registry reconciliation tests proving transient or incomplete data
  cannot delete existing entities.
- After aiopnsense is released, install the published artifact in the
  hass-opnsense test environment and run the public result-method contract tests
  before changing dependency pins. This installed-artifact validation is still
  pending.
- Run the full pytest suite and `prek run --all-files`.

## Rollout and Observability

1. Release aiopnsense first.
2. Pin that release in hass-opnsense and land the integration behavior second.
3. Add debug logging for cache transitions containing method, path, old state,
   new state, and reason, without logging credentials or response payloads.
4. Count probes, positive refreshes, invalidations, negative-cache skips, and
   recoveries in tests. Avoid adding permanent high-cardinality production
   metrics solely for this change.
5. Validate read-only behavior against a live OPNsense instance if available.
   Plugin installation or removal remains a mutating live operation and requires
   explicit permission.

## Acceptance Criteria

- Active optional endpoints do not require an extra availability request on
  every coordinator update.
- Removing an optional plugin produces at most one handled real-request 404,
  followed by bounded negative probes.
- A timeout or 5xx outage does not poison endpoint availability state.
- Core-health and correlated-404 classification occurs before any per-endpoint
  mutation; a router-wide failure preserves positive, negative, and pending
  state and timestamps and cannot mark plugins absent for the negative TTL.
- Per-key cache writes unconditionally reject stale observations, with a
  deterministic controlled-interleaving regression proving an older result
  cannot overwrite newer state or timestamps.
- An isolated optional-endpoint outage does not affect unrelated categories.
- Newly installed supported plugins are detected within the negative TTL.
- Fixed-schema entities can become available without reloading the integration.
- Dynamic entity reconciliation never deletes entities from incomplete or
  transient data.
- aiopnsense and hass-opnsense full test and lint gates pass on coordinated
  branches.

## Implemented Decisions

1. The initial negative TTL is five minutes.
2. Optional endpoint results carry explicit category state and authority; repair
   reconciliation consumes that authority rather than inferring deletion safety
   from normalized empty payloads.
3. Category scopes follow existing unique-ID families: binary sensors use
   interfaces, SMART, and notices; sensors use telemetry, vnStat, SMART,
   Speedtest, certificates, VPN, gateways, interfaces, CARP, and DHCP; switches use
   firewall/NAT, services, VPN, CARP, and Unbound; trackers use ARP.
4. NUT is a fixed three-sensor schema and always creates its configured entities;
   dynamic SMART sensors continue to follow validated device inventory.
5. NUT integration is part of this hass-opnsense delivery as a separately
   configurable granular sync category built on the result contract. It follows
   the existing default-enabled granular sync behavior rather than requiring
   explicit opt-in.
6. Decide whether the first optional-endpoint disappearance is debug-only or a
   rate-limited warning.
