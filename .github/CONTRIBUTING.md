# Contributing

Report bugs and request features through [GitHub Issues](https://github.com/spencernorton/armiger/issues/new/choose). For security issues, use [private vulnerability reporting](SECURITY.md). Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Use synthetic, public-domain or appropriately licensed fixtures; exclude private tracker credentials and personal data. Keep swarm viability independent of content-category probabilities. Preserve benchmark provenance and the default-off sharing behavior. See the [use policy](LEGAL.md).

```bash
python -m pip install -e '.[ultra,dev]'
python -m armiger download --variant all
python -m pytest -q
python -m build
```

Tests use `ARMIGER_MODEL_DIR` or the default verified model cache. The release check installs the wheel outside the checkout and runs inference. Sign commits with `git commit -s` to certify the [Developer Certificate of Origin](https://developercertificate.org/).

Changes to images, connector pins or setup commands also require the native container and stock-upstream checks in [release validation](../docs/release-validation.md).

Keep setup instructions short and executable. Bitmagnet and Rats Search Compose guides are supported entry points; verify them whenever dependencies or model loading change. Put detailed measurements in [evaluation](../docs/evaluation.md), not the README.
