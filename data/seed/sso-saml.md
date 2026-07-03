# SSO / SAML setup and troubleshooting

Single sign-on via SAML 2.0 is available on Business and Enterprise plans.
SCIM provisioning is Enterprise-only.

## Configuring your identity provider

Under Settings > Security > SSO you'll find the values your IdP needs:

- ACS URL: `https://<workspace>.nimbusdesk.io/saml/acs`
- Entity ID: `https://<workspace>.nimbusdesk.io/saml/metadata`
- NameID format: `emailAddress` (required — persistent NameIDs are not supported)

Tested guides exist for Okta, Microsoft Entra ID (Azure AD), Google Workspace
and OneLogin. Upload your IdP metadata XML or paste the SSO URL + certificate.

## Enforcing SSO

Once verified, admins can set SSO to **Required**, which disables password
login for all members except the workspace owner (break-glass account). Guests
authenticate with email codes and are not subject to SSO enforcement.

## Common errors

- **ND-SSO-401 "Invalid signature"**: the IdP signing certificate in Nimbus is
  outdated — usually after an IdP certificate rotation. Re-upload the current
  metadata. This is the single most common SSO ticket.
- **ND-SSO-403 "User not assigned"**: the user exists in the IdP but the Nimbus
  app is not assigned to them (Okta: Assignments tab; Entra: Users and groups).
- **ND-SSO-404 "Unknown NameID"**: the IdP sends a NameID that doesn't match
  any member email. Check for plus-addressing or domain aliases; the assertion
  email must match the invited email exactly.
- **Clock skew**: assertions are valid ±5 minutes. IdP servers with wrong
  clocks produce intermittent "assertion expired" failures.

## SCIM provisioning (Enterprise)

SCIM tokens are issued under Settings > Security > SCIM and expire after 12
months. Deprovisioned users are moved to a deactivated state immediately and
their licenses are freed at the next billing sync (within 24 h). SCIM group
push maps IdP groups to Nimbus teams; nested groups are flattened.
