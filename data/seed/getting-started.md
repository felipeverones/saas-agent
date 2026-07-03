# Getting started with Nimbus

Nimbus is NimbusDesk's cloud workspace for file sync, sharing and team
collaboration. This guide covers first-time setup for workspace admins.

## Creating your workspace

Sign up at app.nimbusdesk.io with a work email. The first user automatically
becomes the workspace **owner**. Choose a workspace subdomain
(e.g. `acme.nimbusdesk.io`) — it can be changed later under Settings > General,
but existing share links will break when you do.

## Inviting your team

Go to Settings > Members > Invite. You can invite by email (up to 50 at a time)
or enable a shareable invite link. Each invited member consumes one seat on your
plan. Pending invitations expire after 14 days.

Roles available:
- **Owner** — billing, plan changes, workspace deletion. One per workspace.
- **Admin** — member management, security settings, integrations.
- **Member** — full product access, no admin settings.
- **Guest** — access only to items explicitly shared with them. Guests are free
  on Business and Enterprise plans.

## Installing the desktop and mobile clients

Desktop clients (Windows 10+, macOS 12+, Ubuntu 22.04+) are available at
nimbusdesk.io/download. The desktop client is required for Smart Sync
(placeholder files that download on demand). Mobile apps are on the App Store
and Play Store; offline file access requires a Pro plan or higher.

## First sync

After installing, sign in and pick the folders to sync. Initial sync speed is
capped at 40 MB/s per client. Large workspaces (over 100 GB) can take several
hours — the client shows an estimate under the sync icon. If a first sync
appears stuck at 99%, see the troubleshooting article "Sync stuck or slow".
