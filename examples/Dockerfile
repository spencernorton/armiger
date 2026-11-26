FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir ".[ultra]" \
    && useradd --create-home --uid 10001 armiger
# Immutable connector source. Model downloads contain data only.
ADD --chmod=0444 --checksum=sha256:5798537d28c1644bcaa5d3f77dbc4ebdd7e9191d2dc4ad6c585bb7bb7e2d04fb https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/99dcc59b8e733b4063bf09f61bcb80b73e802bd1/bitmagnet/bitmagnet_connector.py /opt/connectors/bitmagnet/bitmagnet_connector.py
ADD --chmod=0444 --checksum=sha256:1193569b699fa6b211189bd0797e4831c19c269347dc2e944f0c21455a0c80d9 https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/99dcc59b8e733b4063bf09f61bcb80b73e802bd1/rats-search/rats_search_connector.py /opt/connectors/rats-search/rats_search_connector.py
# Bundle model data at build time; checksum verification never executes downloaded code.
ADD --chmod=0444 --checksum=sha256:b9bb4a0f3bebcaf2dd489c41807c9026962c649036830496f9db50e5fe3d282a https://huggingface.co/NorviTech/armiger-ultra/resolve/13e972ac637debf07979dc0ac4268367e562c5f1/config.json /models/config.json
ADD --chmod=0444 --checksum=sha256:353bffcd4e8de78eb6aeb098a7f0fc11e5f674358cb2df25de8fd07f6901cef7 https://huggingface.co/NorviTech/armiger-ultra/resolve/13e972ac637debf07979dc0ac4268367e562c5f1/vocab.json /models/vocab.json
ADD --chmod=0444 --checksum=sha256:c311b6b95db96a0c7166b93e5a2119f90c92c9d420e565747b01dfb991123978 https://huggingface.co/NorviTech/armiger-ultra/resolve/13e972ac637debf07979dc0ac4268367e562c5f1/calibration.json /models/calibration.json
ADD --chmod=0444 --checksum=sha256:d77547768dd293d3630ea1bda802db272221f2499222e4464f642564c9344cbe https://huggingface.co/NorviTech/armiger-ultra/resolve/13e972ac637debf07979dc0ac4268367e562c5f1/armiger_ultra.fp32.onnx /models/armiger_ultra.fp32.onnx
ADD --chmod=0444 --checksum=sha256:4e4c7a2dd59fb91a234e6627e7ade35062f99268d3f05370432bb9f2a2f9b1c9 https://huggingface.co/NorviTech/armiger-ultra/resolve/13e972ac637debf07979dc0ac4268367e562c5f1/swarm_adapter.pt /models/swarm_adapter.pt
ADD --chmod=0444 --checksum=sha256:8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26 https://huggingface.co/NorviTech/armiger-ultra/resolve/13e972ac637debf07979dc0ac4268367e562c5f1/validated_zero_cue_catalog.json /models/validated_zero_cue_catalog.json
# Remote ADD applies --chmod to newly created parent directories too. Keep
# directories searchable while the downloaded scripts and model files stay 0444.
RUN chmod 0755 /opt/connectors /opt/connectors/bitmagnet /opt/connectors/rats-search /models \
    && python -m armiger download --directory /models --variant ultra --offline \
    && chown -R armiger:armiger /app
USER 10001:10001
ENV ARMIGER_MODEL_DIR=/models ARMIGER_VARIANT=ultra ARMIGER_AUTO_DOWNLOAD=true PYTHONUNBUFFERED=1
# Fail the image build if its runtime user cannot read either connector or model.
RUN python -c "from pathlib import Path; from armiger.artifacts import download_model; \
    [compile(Path(p).read_bytes(), p, 'exec') for p in \
    ('/opt/connectors/bitmagnet/bitmagnet_connector.py', '/opt/connectors/rats-search/rats_search_connector.py')]; \
    download_model(variant='ultra', offline=True)"
EXPOSE 8096
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/healthz', timeout=3).read()"]
CMD ["python", "-m", "armiger", "serve", "--host", "0.0.0.0", "--port", "8096"]
