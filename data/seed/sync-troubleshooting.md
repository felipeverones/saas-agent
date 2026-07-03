# Sync stuck or slow: troubleshooting guide

The desktop client syncs continuously in the background. Most "sync is broken"
tickets fall into one of the patterns below — work through them in order.

## 1. Check the basics

- Is there an active incident? Check status.nimbusdesk.io first.
- Is the client signed in to the right workspace? (Tray icon > account.)
- Is the client up to date? Versions older than 12 months are blocked from
  syncing entirely (error **ND-SYNC-UPGRADE-REQUIRED**).

## 2. Sync stuck at 99% or on specific files

Usually caused by files that cannot be read or synced:

- **Locked files**: Outlook PST files, open Access databases and similar
  always-locked files cannot sync while open. The client shows
  **ND-SYNC-LOCKED** with the file list under tray icon > Sync issues.
- **Path too long** (Windows): paths over 260 characters fail with
  **ND-SYNC-PATH**. Enable long paths in Windows or shorten the folder names.
- **Invalid characters**: names containing `< > : " | ? *` or trailing dots
  fail on Windows clients even when created from macOS or the web.
- **Zero-byte placeholder conflicts**: if antivirus quarantines a file mid-sync
  it can leave a placeholder; right-click > "Re-download" resolves it.

## 3. Slow sync

- Initial sync is capped at 40 MB/s per client; this is expected.
- Check Settings > Network in the client: LAN sync should be ON for office
  deployments (peers fetch blocks from each other instead of the internet).
- Corporate proxies that intercept TLS force the client into a slower HTTP/1.1
  fallback; whitelist `*.nimbusdesk.io` from inspection.
- Thousands of tiny files sync slower than few large files — this is expected
  (per-file overhead), consider zipping archival folders.

## 4. Conflicts

When two clients edit the same file offline, the older save is kept as
`filename (conflicted copy YYYY-MM-DD by user)`. Conflicted copies are never
deleted automatically. Frequent conflicts usually mean a shared folder is being
edited simultaneously — recommend the web editor (real-time co-editing) for
those files.

## 5. Collecting logs for support

Tray icon > Help > Export logs produces a diagnostic bundle (last 7 days,
no file contents). Attach it to the ticket together with the workspace
subdomain and the approximate time of the issue in UTC.
