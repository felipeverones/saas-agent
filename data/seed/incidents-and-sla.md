# Incidents, status page and SLA

## Checking service status

Live status is published at status.nimbusdesk.io, with components for Web app,
Desktop sync, API, Webhooks and SSO. Subscribe to updates via email, RSS, Slack
or `GET /v2/status` (unauthenticated, never rate-limited). Before deep
troubleshooting, always check whether an active incident already explains the
symptom.

## Incident severity levels

- **P1** — full outage or data-integrity risk. Status page within 10 minutes,
  updates every 30 minutes.
- **P2** — major feature degraded (e.g. sync delayed > 15 min for many
  workspaces). Updates hourly.
- **P3** — minor feature degraded, workaround exists.
- **P4** — cosmetic or isolated issue, tracked but not on the status page.

Post-incident reviews (public RCA) are published within 5 business days for
every P1 and P2.

## Uptime SLA (Enterprise)

Enterprise contracts include a **99.9% monthly uptime SLA** for the Web app and
API components, measured by our external probes. Scheduled maintenance
(announced ≥72 h ahead, max 4 h/month, off-peak) is excluded.

Service credits on breach, applied to the next invoice on request within 30
days of the incident:

- Below 99.9%: 10% of the monthly fee
- Below 99.0%: 25% of the monthly fee
- Below 95.0%: 50% of the monthly fee

Credits are the exclusive remedy and require the customer to open a claim
ticket referencing the incident id. Pro and Business plans have no contractual
SLA, though they benefit from the same infrastructure.

## When to escalate to NimbusDesk support

Escalate as **urgent** if: you suspect data loss or corruption, a security
incident affecting your workspace, or sync is fully stopped for more than 30
minutes with no active incident on the status page. Enterprise customers have
a 1-hour first-response target for urgent tickets, 24/7.
