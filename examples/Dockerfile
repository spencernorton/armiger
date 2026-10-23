# syntax=docker/dockerfile:1
FROM python:3.12-slim
WORKDIR /opt/armiger
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[ultra]' && useradd --create-home --uid 10001 armiger
# Immutable connector source. Model downloads contain data only.
ADD --chmod=0444 --checksum=sha256:e3ab7d7a0fb1d20ce4871103d639117c5502db5832532388b1e4bdd76605896f https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/97efadfbee6a4b7c8b179e412122bc799f0dd377/bitmagnet/bitmagnet_connector.py /opt/connectors/bitmagnet/bitmagnet_connector.py
ADD --chmod=0444 --checksum=sha256:daccc692be45d6810918dab809f46f59d0c6d670aed8867dc1ca1575cadea4f7 https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/97efadfbee6a4b7c8b179e412122bc799f0dd377/rats-search/rats_search_connector.py /opt/connectors/rats-search/rats_search_connector.py
RUN chmod -R a+rX /opt/connectors && mkdir -p /models && chown armiger:armiger /models
ENV ARMIGER_MODEL_DIR=/models ARMIGER_VARIANT=ultra ARMIGER_AUTO_DOWNLOAD=true PYTHONUNBUFFERED=1
USER armiger
CMD ["python", "/opt/connectors/bitmagnet/bitmagnet_connector.py"]
