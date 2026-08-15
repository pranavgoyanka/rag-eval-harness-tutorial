# Fleet Operations — Maintenance, Updates, and Service Levels

## Remote monitoring

Every Nimbus robot is monitored 24/7 from the Austin NOC. Operators receive
live telemetry and can intervene remotely (re-route, pause, or recall) when a
robot flags an issue it cannot resolve on its own.

- NB-1: average 4.2 remote interventions per 1,000 km
- NB-2: average 3.1 remote interventions per 1,000 km

## Maintenance schedule

| Item | NB-1 | NB-2 |
|---|---|---|
| Scheduled service interval | every 500 operating hours or 2,000 km (whichever comes first) | every 800 operating hours |
| Battery replacement | after 800 charge cycles (capacity < 80%) | pack swap after 1,000 cycles |
| Warranty (parts and labor) | 2 years | 3 years |

## Over-the-air (OTA) updates

- **Window:** Tuesdays, 02:00–04:00 local time.
- **Rollout:** the fleet is split into **25% cohorts**; a cohort is promoted
  to the next stage only if no critical regressions are detected.
- Current firmware: NB-1 v4.7.2, NB-2 v2.3.1.

## Geofencing

- Operators define geofenced operating zones for each deployment site.
- Default speed limit inside a geofence: **24 km/h** (lower than the NB-2's
  30 km/h top speed).
- If a robot crosses a geofence boundary it halts with error **E317** and
  requires operator approval to resume. See the FAQ for E317 details.

## Daily service window

Robots return to dock and charge between **01:00 and 05:00 local time** when
not on an active run.

## Onsite service response

| Severity | Response time |
|---|---|
| Critical (robot down, blocking right-of-way) | 4 business hours |
| Non-critical (degraded, e.g., single sensor fault) | 24 hours |
