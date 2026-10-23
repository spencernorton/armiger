## 2026-10-23 — Mutable update validation and canary drift controls

Reconciled the expanded 500-item canary, structural-evidence quorum and stratified
scope threshold consumers with the existing privacy-budget, retry and authenticated
quorum controls. Added bounded canonical bencode, signed sequence bounds, canonical
signature validation, strict zlib framing, and observed per-category nonconformity
quantiles. The drift comparison uses a 0.05 maximum shift and retains an in-process
reference across successive small updates; restart-durable reference management is
still required for a cross-restart cumulative guarantee.

The combined isolated regression and UDP/TCP adversarial suite passed 144 tests.
Four mutation controls fail when the corresponding defense is removed. Tests use
synthetic releases; these results do not qualify model accuracy or live public
publication. Changes are prepared for integration; no production rollout occurred.
