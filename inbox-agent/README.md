# Inbox agent

Reads your unread email, works out which messages need a reply from you, drafts
those replies, and sends each one **only after you approve it**. It learns your
preferences from what you do with its drafts.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # from console.anthropic.com
export EMAIL_ADDRESS=you@gmail.com
export EMAIL_PASSWORD=...             # Gmail: an app password, not your normal password
```

Gmail app passwords: Google Account > Security > 2-Step Verification > App passwords.
For another provider, also set `IMAP_HOST` and `SMTP_HOST`.

## Run

```bash
python agent.py --eml-dir samples   # practise on two sample emails; never sends
python agent.py --dry-run           # your real inbox; never sends
python agent.py                     # your real inbox; sends what you approve
```

For each email that needs a reply you see the email, the draft, and:

| Key | What happens |
|---|---|
| `s` | Send the draft as shown |
| `e` | Edit it in your editor (`$EDITOR`, default nano), then decide again |
| `r` | Tell the agent what to change; it redrafts |
| `k` | Skip, optionally saying why |
| `q` | Stop; the remaining emails come back next run |

At the end it lists the emails it judged need no reply, and you can ask for a
draft for any of them.

It never marks mail as read, never deletes anything, and has no way to send
without your `s`.

## How it learns

After each session it compares its drafts with what you sent, reads your
revision requests and skip reasons, and rewrites `data/preferences.md`: which
emails you reply to, your tone, length, sign-off, and how you treat particular
senders. That file is included in every future draft request. It is plain
text, so you can read it, correct it, or add your own rules.

`data/history.jsonl` keeps the full log of decisions. `data/` stays on your
machine and is git-ignored.

## Limits

- Every draft needs your approval, however long you use it. There is no
  auto-send mode yet.
- Plain-text replies only; attachments are not read.
- Each email costs one model call (two or more if you ask for revisions).
  Change the model with `INBOX_AGENT_MODEL`.
