FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[ultra]" && useradd --create-home --uid 10001 armiger
# Immutable connector source. Model downloads contain data only.
ADD --chmod=0444 --checksum=sha256:86adfd358b09d4964ae51145aa3dfac121d9972f20f26ca5ad0bb588978caf2b https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/9b00089a06df24497e45d2f2144eb14a1bb03ac2/bitmagnet/bitmagnet_connector.py /opt/connectors/bitmagnet/bitmagnet_connector.py
ADD --chmod=0444 --checksum=sha256:ecb2c234da0c5654fe9edb11dc0426b0a2f0c6ea27d825e8e544cbc741b801ae https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/9b00089a06df24497e45d2f2144eb14a1bb03ac2/rats-search/rats_search_connector.py /opt/connectors/rats-search/rats_search_connector.py
# Bundle model data at build time; checksum verification never executes downloaded code.
ADD --chmod=0444 --checksum=sha256:da965d165bc45b5dafb2ae2fcb45781b1c37cebd9fc3026604c12d060d6e523f https://huggingface.co/NorviTech/armiger-ultra/resolve/18ef5418b3858ee4f70c137540febb5343b8754e/config.json /models/config.json
ADD --chmod=0444 --checksum=sha256:c10afea430188e0207752e93bf0b340fbb585858684037c5134a477cbc0dd7bb https://huggingface.co/NorviTech/armiger-ultra/resolve/18ef5418b3858ee4f70c137540febb5343b8754e/vocab.json /models/vocab.json
ADD --chmod=0444 --checksum=sha256:300387779dc16292dd9e6be038ce9dd60e5bdb0ee929b383dcd4e51715f8b5f8 https://huggingface.co/NorviTech/armiger-ultra/resolve/18ef5418b3858ee4f70c137540febb5343b8754e/calibration.json /models/calibration.json
ADD --chmod=0444 --checksum=sha256:6f94aef676e5bec4968378437769cfdc768fefcc6a21854b8319b33456ebd2ec https://huggingface.co/NorviTech/armiger-ultra/resolve/18ef5418b3858ee4f70c137540febb5343b8754e/armiger_ultra.fp32.onnx /models/armiger_ultra.fp32.onnx
ADD --chmod=0444 --checksum=sha256:4e4c7a2dd59fb91a234e6627e7ade35062f99268d3f05370432bb9f2a2f9b1c9 https://huggingface.co/NorviTech/armiger-ultra/resolve/18ef5418b3858ee4f70c137540febb5343b8754e/swarm_adapter.pt /models/swarm_adapter.pt
ADD --chmod=0444 --checksum=sha256:8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26 https://huggingface.co/NorviTech/armiger-ultra/resolve/18ef5418b3858ee4f70c137540febb5343b8754e/validated_zero_cue_catalog.json /models/validated_zero_cue_catalog.json
RUN python -m armiger download --directory /models --variant ultra --offline && chown -R armiger:armiger /app
USER 10001:10001
ENV ARMIGER_MODEL_DIR=/models ARMIGER_VARIANT=ultra ARMIGER_AUTO_DOWNLOAD=true ARMIGER_DHT_AUTO_UPDATE=true PYTHONUNBUFFERED=1
EXPOSE 8096
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/healthz', timeout=3).read()"]
CMD ["python", "-m", "armiger", "serve", "--host", "0.0.0.0", "--port", "8096"]
