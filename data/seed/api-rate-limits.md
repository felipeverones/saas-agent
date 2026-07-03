# API rate limits and error 429

The Nimbus REST API enforces per-workspace rate limits to protect shared
infrastructure. Limits depend on your plan.

## Limits by plan

- **Free**: 60 requests/minute, 10,000 requests/day.
- **Pro**: 600 requests/minute, 200,000 requests/day.
- **Business**: 3,000 requests/minute, 2,000,000 requests/day.
- **Enterprise**: custom limits, configured per contract.

Upload endpoints (`POST /v2/files`) have a separate limit of 100 concurrent
uploads per workspace on all paid plans.

## How limits are enforced

Every response includes headers:

- `X-RateLimit-Limit` — your per-minute ceiling.
- `X-RateLimit-Remaining` — requests left in the current window.
- `X-RateLimit-Reset` — unix timestamp when the window resets.

When you exceed the limit, the API returns **HTTP 429 Too Many Requests** with
a `Retry-After` header (seconds). Error body:

```json
{"error": "rate_limited", "code": "ND-API-429", "retry_after": 12}
```

## Recommended client behavior

Respect `Retry-After` and use exponential backoff with jitter. Do not retry
immediately in a tight loop — repeated bursts after a 429 can trigger a
temporary block (**ND-API-429-HARD**, 15 minutes) that support cannot lift
early. Batch endpoints (`/v2/batch`) count as one request per call, not per
item; prefer them for bulk operations.

## Common causes of unexpected 429s

- Polling for changes instead of using webhooks (see "Webhooks").
- Several CI jobs sharing one API token — issue one token per integration under
  Settings > Integrations > API tokens.
- Sync loops: two automations writing to the same folder and re-triggering each
  other.

If you believe the limit is misconfigured for your plan, support can verify the
workspace tier; genuine limit increases on Business require the Enterprise plan.
