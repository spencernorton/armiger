# Contributor rules

Use independent worktrees and preserve existing research artifacts. Keep real labels,
private data and credentials out of changes. Synthetic control tests demonstrate
validation behavior; they do not establish trained-model accuracy or adoption.

Security regression checks: `python -m pytest test_armiger_dht_rsi.py
test_dht_security_and_9head_remediation.py test_bep44_udp_wire_e2e.py -q`.

Read the latest entry in `AGENTS/journal.md` before modifying the update protocol.
