# Fleet API Reference (v2)

## Base URL

- Production: `https://api.nimbusrobotics.com/v2`
- Sandbox: `https://sandbox.nimbusrobotics.com/v2` (data resets every 7 days)

## Authentication

- Bearer token in the `Authorization` header.
- Access tokens expire after **1 hour**; refresh tokens are valid for
  **30 days**.
- Scopes: `fleet:read`, `fleet:command`, `geofence:write`.

## Rate limits

| Limit | Value |
|---|---|
| Sustained | 600 requests/minute per token |
| Burst | 10 requests/second |

Exceeding the limits returns HTTP 429.

## Core endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/robots` | List robots (paginated, max 100 per page) |
| GET | `/robots/{id}` | Robot detail |
| POST | `/robots/{id}/commands` | Issue a command: `start`, `stop`, `pause`, `return_to_dock` |
| GET | `/robots/{id}/telemetry` | Telemetry, last 24 h at 10-second resolution |
| GET | `/deliveries` | List deliveries; filters: `robot_id`, `from`, `to` |
| POST | `/geofences` | Create a geofence (polygon, max 200 vertices) |
| GET | `/fleet/health` | Fleet-wide health summary |

## Webhooks

Subscribe to: `delivery.completed`, `robot.fault`, `geofence.breach`.

## Error codes

| HTTP | Meaning |
|---|---|
| 403 | Insufficient scope |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 503 | Service unavailable (check the status page) |

## SDKs

- Python: `nimbus-fleet` (Python 3.9+)
- Node.js: `@nimbus/fleet` (Node 18+)
