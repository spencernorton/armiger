FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[ultra]" && useradd --create-home --uid 10001 armiger
# Immutable connector source. Model downloads contain data only.
ADD --chmod=0444 --checksum=sha256:86adfd358b09d4964ae51145aa3dfac121d9972f20f26ca5ad0bb588978caf2b https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/9b00089a06df24497e45d2f2144eb14a1bb03ac2/bitmagnet/bitmagnet_connector.py /opt/connectors/bitmagnet/bitmagnet_connector.py
ADD --chmod=0444 --checksum=sha256:ecb2c234da0c5654fe9edb11dc0426b0a2f0c6ea27d825e8e544cbc741b801ae https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/9b00089a06df24497e45d2f2144eb14a1bb03ac2/rats-search/rats_search_connector.py /opt/connectors/rats-search/rats_search_connector.py
# Bundle model data at build time; checksum verification never executes downloaded code.
ADD --chmod=0444 --checksum=sha256:caf3c30c7f6380b84af4f61ebc0afa80707b9137ec3be4e2d03e09c3e05a0ad4 https://huggingface.co/NorviTech/armiger-ultra/resolve/aa760eeea91fba887124e535dcd1df7496591cea/config.json /models/config.json
ADD --chmod=0444 --checksum=sha256:c10afea430188e0207752e93bf0b340fbb585858684037c5134a477cbc0dd7bb https://huggingface.co/NorviTech/armiger-ultra/resolve/aa760eeea91fba887124e535dcd1df7496591cea/vocab.json /models/vocab.json
ADD --chmod=0444 --checksum=sha256:ea7a4df70747160b41ae0cbca0292562f097bbbea7b5d549c5348f743e30b33c https://huggingface.co/NorviTech/armiger-ultra/resolve/aa760eeea91fba887124e535dcd1df7496591cea/calibration.json /models/calibration.json
ADD --chmod=0444 --checksum=sha256:ebed96dfef6f78842768ecdad646f6e5d838681feed9a4919d4b95aa0f912ea4 https://huggingface.co/NorviTech/armiger-ultra/resolve/aa760eeea91fba887124e535dcd1df7496591cea/armiger_ultra.fp32.onnx /models/armiger_ultra.fp32.onnx
ADD --chmod=0444 --checksum=sha256:4e4c7a2dd59fb91a234e6627e7ade35062f99268d3f05370432bb9f2a2f9b1c9 https://huggingface.co/NorviTech/armiger-ultra/resolve/aa760eeea91fba887124e535dcd1df7496591cea/swarm_adapter.pt /models/swarm_adapter.pt
ADD --chmod=0444 --checksum=sha256:8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26 https://huggingface.co/NorviTech/armiger-ultra/resolve/aa760eeea91fba887124e535dcd1df7496591cea/validated_zero_cue_catalog.json /models/validated_zero_cue_catalog.json
# Remote ADD applies --chmod to newly created parent directories too. Keep
# directories searchable while the downloaded scripts and model files stay 0444.
RUN chmod 0755 /opt/connectors /opt/connectors/bitmagnet /opt/connectors/rats-search /models \
    && python -m armiger download --directory /models --variant ultra --offline \
    && chown -R armiger:armiger /app
USER 10001:10001
ENV ARMIGER_MODEL_DIR=/models ARMIGER_VARIANT=ultra ARMIGER_AUTO_DOWNLOAD=true ARMIGER_DHT_AUTO_UPDATE=true PYTHONUNBUFFERED=1
# Fail the image build if its runtime user cannot read either connector or model.
RUN python -c "from pathlib import Path; from armiger.artifacts import download_model; \
    [compile(Path(p).read_bytes(), p, 'exec') for p in \
    ('/opt/connectors/bitmagnet/bitmagnet_connector.py', '/opt/connectors/rats-search/rats_search_connector.py')]; \
    download_model(variant='ultra', offline=True)"
EXPOSE 8096
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/healthz', timeout=3).read()"]
CMD ["python", "-m", "armiger", "serve", "--host", "0.0.0.0", "--port", "8096"]
