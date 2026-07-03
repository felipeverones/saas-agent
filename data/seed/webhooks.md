# Webhooks: delivery, signatures and troubleshooting

Webhooks push events (file created, share link opened, member joined, etc.) to
your HTTPS endpoint, replacing polling.

## Setting up an endpoint

Create endpoints under Settings > Integrations > Webhooks. Requirements:
public HTTPS URL with a valid certificate (self-signed rejected), responds with
a 2xx status within **10 seconds**. Slow handlers should enqueue the event and
return 200 immediately.

## Verifying signatures

Every delivery includes header `X-Nimbus-Signature`:
`sha256=` + HMAC-SHA256 of the raw request body, keyed with your endpoint's
signing secret (shown once at creation; rotate under the endpoint's settings).
Always verify the signature before trusting a payload — unsigned requests to
your endpoint may be forgeries. Compare using a constant-time function.

## Retry policy

Failed deliveries (non-2xx, timeout, TLS error) are retried up to **5 times**
with exponential backoff: 1 min, 5 min, 30 min, 2 h, 12 h. After the final
failure the event is dropped and a `webhook.delivery_failed` notice appears in
the audit log. An endpoint failing continuously for **72 hours is automatically
disabled** (error code **ND-WH-DISABLED**) and must be re-enabled manually.

## Common failure modes

- **ND-WH-TLS**: certificate expired or incomplete chain. Test with
  `openssl s_client -connect host:443`.
- **ND-WH-TIMEOUT**: handler exceeded 10 s. Move processing to a queue.
- **ND-WH-SIG-MISMATCH** (in your logs, not ours): you are verifying against
  the wrong secret after a rotation, or a proxy is rewriting the body
  (compression, re-encoding) before verification. Verify the raw bytes.
- Deliveries arriving out of order: order is not guaranteed. Use the
  `event.sequence` field to reorder, or design handlers to be idempotent using
  `event.id` (deliveries may occasionally duplicate).

## Replaying missed events

Business and Enterprise plans can replay events from the last 7 days:
endpoint page > Deliveries > Redeliver. The API equivalent is
`POST /v2/webhooks/{id}/redeliver` with a list of event ids.
