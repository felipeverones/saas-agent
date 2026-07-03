# Security overview

## Encryption

All data is encrypted in transit (TLS 1.2+) and at rest (AES-256). File
contents are stored as encrypted, content-addressed blocks; encryption keys are
managed in a dedicated KMS with automatic rotation every 90 days. Enterprise
customers can enable customer-managed keys (CMK) via AWS KMS — note that
revoking a CMK makes all workspace data unreadable, including to Nimbus.

## Compliance and certifications

NimbusDesk maintains SOC 2 Type II and ISO 27001 certifications, renewed
annually. Reports are available under NDA at trust.nimbusdesk.io. GDPR: we act
as processor; a signable DPA is available on Business and Enterprise. HIPAA
BAAs are offered on Enterprise only.

## Access controls

- Two-factor authentication (TOTP or WebAuthn keys) can be enforced
  workspace-wide by admins.
- Session length is configurable (8 h to 30 days); revoking a session takes
  effect within 60 seconds.
- Share links support passwords, expiration dates and download-disable on paid
  plans. "Anyone with the link" can be disabled workspace-wide.
- Device approval (Enterprise): new desktop clients require admin approval
  before their first sync.

## Employee access to customer data

Production access requires hardware-key MFA, is limited to on-call engineers,
and every access is logged and reviewed weekly. Support agents can see file
and folder **names** for troubleshooting, but can only view file **contents**
after the customer grants explicit, time-boxed consent from the support ticket
UI (24-hour window, revocable).

## Vulnerability reporting

Report vulnerabilities to security@nimbusdesk.io — we run a public bounty
program with a 90-day disclosure policy. Do not run automated scanners against
production; use the sandbox at sandbox.nimbusdesk.io for testing.
