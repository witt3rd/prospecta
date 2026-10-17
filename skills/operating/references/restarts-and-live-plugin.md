# Restarts and the live plugin

This is house practice from the go-live runbooks; it is not described elsewhere in this repo, so it is
recorded here and a change to it is a PR to this skill. What the code itself guarantees is marked.

## What needs a restart, and what does not

| change | when it takes effect | basis |
|---|---|---|
| library code (the Python package in the gateway's venv) | at a gateway restart | a process imports code once |
| plugin code (`__init__.py` and the rest of the live plugin directory) | at a gateway restart | same |
| the recall contract file | on the next recall, at pull time: no restart | `load_contract` caches by (path, file modification time), so a changed file is re-parsed on the next call |
| plugin switches and files (`prefetch` settings, `contract_path`) | at the next session's start | the plugin README, "Switches and files" |
| a bank's `recall_config` | on the next recall | `read_recall_config` is read on every recall |

Because a contract (or other runtime file) change is live the moment the live directory is updated, **moving
that directory is the go-live act**; treat it like a deploy.

## The quiet check

Before a restart, twice, 20 seconds apart: the log shows no turn for a few minutes; the gateway has no child
processes; the state file says running. Both passes must be quiet. Not at a mail-check minute. One gateway at a time;
confirm the first gateway is healthy before touching the next.

## A venv per library pin

Each prospecta pin gets its own venv. The gateway's systemd drop-in names the venv, so an upgrade is a new
venv plus a repointed drop-in, and a rollback is pointing the drop-in back and restarting. *Why:* the old venv still
matches the old pin exactly, so nothing has to be rebuilt to go back.

## The live plugin directory

- The directory the gateways load is **detached** at a recorded commit (note the sha in the go-live notes).
- It is moved only by the go-live runbook: fetch, check out the new commit detached, record it, then
  restart if code changed.
- It does **not** follow main and is not a clone you pull in. A working clone that syncs with main is a different directory.
- After any move, check one recall in the gateway log: a contract change is already live, code is not yet.
