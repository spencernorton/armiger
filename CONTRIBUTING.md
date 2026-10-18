# Contributing to Armiger

Thank you for contributing to Armiger.

## Ground Rules

1. **Copyright & Legal Hygiene (`LEGAL.md`):**
   - Never include copyrighted media titles, scene/release-group names, or private-tracker identifiers in plaintext in code, regexes, vocabularies, or test fixtures.
   - Use Creative Commons Attribution (CC-BY) open movies (*Sintel*, *Tears of Steel*, *Big Buck Bunny*, *Cosmos Laundromat*, *Caminandes*, *Spring*, *Coffee Run*, *Charge*, *Wing It!*, *Elephants Dream*), CC-BY webcomics (*Pepper&Carrot*), CC0 recordings (*Open Goldberg Variations*), or public-domain literature (*LibriVox*, *Project Gutenberg*) in all tests and examples.
   - Store release-group or title-catalog lookup tokens exclusively as keyed 64-bit BLAKE2b digests (`digest_size=8`).
2. **Causal Separation Guarantee:**
   - Swarm liveliness (`seeders`, `leechers`, `completed`, `liveness_status`) must **never** alter the 11-class `category` probability distribution (`0.00%` category flip invariant).
3. **Zero PII & Private-Tracker Exclusion:**
   - All telemetry and source normalizers must strip passkeys, authkeys, local filesystem paths, usernames, and peer IP addresses.
4. **Developer Certificate of Origin (DCO):**
   - Sign off every commit with `git commit -s` (`Signed-off-by: Name <email>`), certifying the [Developer Certificate of Origin v1.1](https://developercertificate.org/).

## Running the Test Suite

```bash
# Stdlib unittest suites (81 tests across DHT DB Sync, CSAM Guard, and DHT RSI):
python3 -m unittest discover -s . -p "test_*.py" -v

# Full 91-test verification suite (including ONNX + PyTorch 11-head pipeline parity):
pytest -v
```
