FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir ".[ultra]" \
    && useradd --create-home --uid 10001 armiger
# Immutable connector source. Model downloads contain data only.
ADD --chmod=0444 --checksum=sha256:4f5f5707f50997539ac5fcbcb8e88d7302993cdcd1bf275504a9db8a63224ef1 https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/283f4461c18f2b873460c4cde8f43d03ac4e3c03/bitmagnet/bitmagnet_connector.py /opt/connectors/bitmagnet/bitmagnet_connector.py
ADD --chmod=0444 --checksum=sha256:1193569b699fa6b211189bd0797e4831c19c269347dc2e944f0c21455a0c80d9 https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/283f4461c18f2b873460c4cde8f43d03ac4e3c03/rats-search/rats_search_connector.py /opt/connectors/rats-search/rats_search_connector.py
# Bundle model data at build time; checksum verification never executes downloaded code.
ADD --chmod=0444 --checksum=sha256:074c80803dc5cbc426059553bd7a5577e4c66fb5685dc4ae33568b713e08fa4a https://huggingface.co/NorviTech/armiger-ultra/resolve/b9f70a0448fc98f4113126552f46bbd42eaecf0e/config.json /models/config.json
ADD --chmod=0444 --checksum=sha256:c10afea430188e0207752e93bf0b340fbb585858684037c5134a477cbc0dd7bb https://huggingface.co/NorviTech/armiger-ultra/resolve/b9f70a0448fc98f4113126552f46bbd42eaecf0e/vocab.json /models/vocab.json
ADD --chmod=0444 --checksum=sha256:92aa6aac07a9868f7a54fc0089eec6b06aa86c759723dd0731e0afd13e617cdd https://huggingface.co/NorviTech/armiger-ultra/resolve/b9f70a0448fc98f4113126552f46bbd42eaecf0e/calibration.json /models/calibration.json
ADD --chmod=0444 --checksum=sha256:70286459da47287eb5d984cd67251e0f80f511ceee0ad6ad34c7c2334c9223ec https://huggingface.co/NorviTech/armiger-ultra/resolve/b9f70a0448fc98f4113126552f46bbd42eaecf0e/armiger_ultra.fp32.onnx /models/armiger_ultra.fp32.onnx
ADD --chmod=0444 --checksum=sha256:4e4c7a2dd59fb91a234e6627e7ade35062f99268d3f05370432bb9f2a2f9b1c9 https://huggingface.co/NorviTech/armiger-ultra/resolve/b9f70a0448fc98f4113126552f46bbd42eaecf0e/swarm_adapter.pt /models/swarm_adapter.pt
ADD --chmod=0444 --checksum=sha256:8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26 https://huggingface.co/NorviTech/armiger-ultra/resolve/b9f70a0448fc98f4113126552f46bbd42eaecf0e/validated_zero_cue_catalog.json /models/validated_zero_cue_catalog.json
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
