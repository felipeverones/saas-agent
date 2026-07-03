# Data export and retention

## Exporting your data

Workspace owners can request a full export under Settings > General > Export.
Exports include all files, folder structure, comments (as JSON) and the member
list (CSV). Version history and audit logs are included only on Business and
Enterprise. Exports are packaged as ZIP archives of at most 50 GB each and
download links are valid for **7 days**. A workspace export can be requested at
most once every 24 hours.

Individual users can export their personal files anytime via the desktop client
(right-click > Download) or `GET /v2/files/archive` on the API.

## Version history retention

- Free: 30 days
- Pro: 90 days
- Business: 1 year
- Enterprise: configurable, up to indefinite

Restoring a previous version never consumes extra storage; versions share
unchanged blocks.

## Deleted file retention (trash)

Deleted files go to the workspace trash: 30 days on Free/Pro, 90 days on
Business/Enterprise, then they are purged permanently. Admins can restore any
member's trashed files. Purged data is unrecoverable — support cannot restore
files after the trash window, no exceptions.

## Account and workspace deletion

Workspace deletion (owner only) starts a 14-day grace period during which the
owner can cancel the deletion by logging in. After the grace period, all data
is deleted from production systems within 24 hours and from backups within
**35 days**. Enterprise customers can request a certificate of deletion.

## Data residency

By default data is stored in the US (us-east). Enterprise workspaces can choose
EU (Frankfurt) residency at creation time. Residency cannot be changed after
workspace creation — migrating requires a new workspace and a support-assisted
transfer.
