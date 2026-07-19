# Endpoint Availability and Optional-Entity Reconciliation Plan

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
- aiopnsense exposes NUT UPS status, but `origin/main` of hass-opnsense does not
  currently poll or expose it.

## Desired Semantics

### Cache observations

1. Use method-aware, exact-path cache keys for every observation.
2. A successful real request refreshes an existing positive availability entry.
3. A real 404 from an explicitly optional endpoint invalidates its positive
   entry immediately.
4. The next access performs an actual probe. A confirmed 404 creates a negative
   entry with a shorter TTL than a positive entry.
5. Timeouts, connection errors, 401/403 responses, 429 responses, and 5xx
   responses do not change endpoint availability. They describe transport,
   authentication, authorization, throttling, or service health—not route
   existence.
6. Never infer a base endpoint from a resource-specific or parameterized path.
   Derived paths may affect a registered probe key only through an explicit
   mapping.
7. Do not populate the availability cache for every successful HTTP request.
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
   an older result. Add a small per-key lock or equivalent single-flight
   mechanism only if tests demonstrate an actual race.

### Phase 2: aiopnsense status-aware optional transport

1. Add a private status-aware request path for optional endpoints rather than
   teaching all generic GET and POST calls that every 404 means absence.
2. Carry enough response information to distinguish:
   - success;
   - confirmed optional-endpoint 404;
   - non-404 HTTP failure;
   - transport failure;
   - malformed successful payload.
3. Feed success and confirmed optional 404 observations into the cache helpers.
4. Migrate optional endpoint callers incrementally, beginning with Speedtest and
   NUT. Do not alter mutating endpoint behavior as part of this work.
5. Retain exact-path behavior for Speedtest `showlog` and `showstat`. Do not let a
   parameterized request invalidate an unrelated probe key.

### Phase 3: hass-opnsense coordinator consumption

1. Update the aiopnsense dependency pin in both `manifest.json` and
   `pyproject.toml` after the companion release is available.
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
4. Add NUT polling only as a separate, explicit integration feature. Do not make
   cache reconciliation implicitly enable a category hass-opnsense does not
   support.

### Phase 4: hass-opnsense entity lifecycle

Choose entity behavior based on whether the optional capability has a static or
dynamic entity schema.

1. For fixed schemas such as Speedtest metrics and the proposed fixed NUT
   metrics, create the configured entities regardless of whether the plugin is
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
- Assert that the matching positive entry is invalidated and the failure is
  classified as optional disappearance.
- On the next access, confirm one real probe stores a short-lived negative
  result.
- Assert later polls skip payload requests while the negative entry is fresh.
- Assert existing Speedtest entities become unavailable but remain registered.

### Router-wide five-minute transient failure

- Cover timeouts, connection errors, and 5xx responses.
- Assert endpoint availability entries are unchanged.
- Assert recovery occurs on the first successful update after service returns.
- Cover correlated 404 responses from core and optional endpoints and assert
  they are treated as global health failure rather than independent plugin
  removals.

### Speedtest-only five-minute transient failure

- For timeout and 5xx failures, assert the positive cache remains intact and
  recovery occurs on the first later success.
- For isolated 404 responses, assert invalidation, negative probing, skipped
  requests during the negative TTL, and recovery no later than one negative TTL
  after the endpoint returns.
- Assert unrelated categories continue updating.

### Optional plugin installed while Home Assistant is running

- Begin with a confirmed negative cache entry.
- Install or simulate the endpoint before the negative TTL expires.
- Assert no request occurs until expiry unless explicitly force-refreshed.
- After expiry, assert the probe succeeds, the payload is fetched, and positive
  observations are refreshed by later real requests.
- For a fixed-schema hass-opnsense category, assert pre-created unavailable
  entities become available without an integration reload.
- For NUT specifically, keep backend detection tests separate from the future
  hass-opnsense NUT category/entity tests.

## Validation

### aiopnsense

- Add focused cache and transport tests covering GET and POST method-aware keys,
  success refresh, exact 404 invalidation, negative TTL expiry, force refresh,
  malformed payloads, concurrency, and non-404 failures.
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
- A router-wide outage cannot mark many plugins absent for the negative TTL.
- An isolated optional-endpoint outage does not affect unrelated categories.
- Newly installed supported plugins are detected within the negative TTL.
- Fixed-schema entities can become available without reloading the integration.
- Dynamic entity reconciliation never deletes entities from incomplete or
  transient data.
- aiopnsense and hass-opnsense full test and lint gates pass on coordinated
  branches.

## Decisions to Confirm Before Implementation

1. Confirm the initial negative TTL (recommended starting value: five minutes).
2. Decide whether correlated-404 protection uses one known core health endpoint
   or a bounded threshold across unrelated endpoints.
3. Confirm which optional categories have fixed schemas and should always create
   configured entities.
4. Decide whether NUT integration is part of the first hass-opnsense delivery or
   remains a separate follow-up built on the cache contract.
5. Decide whether the first optional-endpoint disappearance is debug-only or a
   rate-limited warning.
