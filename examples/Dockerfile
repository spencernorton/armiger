# syntax=docker/dockerfile:1
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/spencernorton/armiger" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="Armiger Ultra"
WORKDIR /opt/armiger
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[ultra]' && useradd --create-home --uid 10001 armiger
# Immutable connector source. Model downloads contain data only.
ADD --chmod=0444 --checksum=sha256:7a468557832c21b6645f7e205fcdf9c8262b2c45d39215b18cca459f2d3eb9df https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/a18596f8a371afcd7df339f3fe6fb288749f55fe/bitmagnet/bitmagnet_connector.py /opt/connectors/bitmagnet/bitmagnet_connector.py
ADD --chmod=0444 --checksum=sha256:d6eba3e9c6d17c5e14adcbab86cb4b913e82da7315b269344488070d2326b70c https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/a18596f8a371afcd7df339f3fe6fb288749f55fe/rats-search/rats_search_connector.py /opt/connectors/rats-search/rats_search_connector.py
# Bundle model data at build time; checksum verification never executes downloaded code.
ADD --chmod=0444 --checksum=sha256:504538ab038cf1733bb1f9f3d8f442495d104af78620b20dbc4f4cf0920269cd https://huggingface.co/NorviTech/armiger-ultra/resolve/284d9ac5cf23eb090731097624451b49f36c8260/config.json /models/config.json
ADD --chmod=0444 --checksum=sha256:c10afea430188e0207752e93bf0b340fbb585858684037c5134a477cbc0dd7bb https://huggingface.co/NorviTech/armiger-ultra/resolve/284d9ac5cf23eb090731097624451b49f36c8260/vocab.json /models/vocab.json
ADD --chmod=0444 --checksum=sha256:e5b58fc6a8380b982f09296822b0244b28aa5b9b38c32d690c9e25160a8860bb https://huggingface.co/NorviTech/armiger-ultra/resolve/284d9ac5cf23eb090731097624451b49f36c8260/calibration.json /models/calibration.json
ADD --chmod=0444 --checksum=sha256:6459a20d7bf45d46d59efbcb50ffb8a519bacda14ae36403dcc510eceaa38f44 https://huggingface.co/NorviTech/armiger-ultra/resolve/284d9ac5cf23eb090731097624451b49f36c8260/armiger_ultra.fp32.onnx /models/armiger_ultra.fp32.onnx
ADD --chmod=0444 --checksum=sha256:85effcb9e70af7bff6047d4ae45d2155444e9b691dc84431f19ad14f9daee9ea https://huggingface.co/NorviTech/armiger-ultra/resolve/284d9ac5cf23eb090731097624451b49f36c8260/swarm_adapter.pt /models/swarm_adapter.pt
ADD --chmod=0444 --checksum=sha256:8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26 https://huggingface.co/NorviTech/armiger-ultra/resolve/284d9ac5cf23eb090731097624451b49f36c8260/validated_zero_cue_catalog.json /models/validated_zero_cue_catalog.json
RUN chmod -R a+rX /opt/connectors /models \
    && python -m armiger download --directory /models --offline \
    && chown -R armiger:armiger /models
ENV ARMIGER_MODEL_DIR=/models ARMIGER_VARIANT=ultra ARMIGER_AUTO_DOWNLOAD=true PYTHONUNBUFFERED=1
USER armiger
EXPOSE 8096
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=20 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('ARMIGER_CONNECTOR_PORT', '8096') + '/healthz', timeout=4)"
CMD ["python", "-m", "armiger", "serve", "--host", "0.0.0.0"]
