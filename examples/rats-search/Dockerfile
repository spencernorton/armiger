# syntax=docker/dockerfile:1
# Console backend built from the upstream v2.3.5 source and pinned submodules.
FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates qt6-base-dev \
    qt6-websockets-dev qt6-tools-dev libgl1-mesa-dev libxkbcommon-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init && git remote add origin https://github.com/librats/rats-search.git \
    && git fetch --depth 1 origin 8415a208025d3827f2f380f3bfe530557a10eaba \
    && git checkout --detach FETCH_HEAD && git submodule update --init --recursive --depth 1
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DRATS_SEARCH_USE_SYSTEM_LIBRATS=OFF -DRATS_SEARCH_BUILD_TESTS=OFF \
    && cmake --build build --config Release --parallel 2

FROM ubuntu:24.04
LABEL org.opencontainers.image.source="https://github.com/spencernorton/armiger" \
      org.opencontainers.image.title="Rats Search backend for Armiger" \
      org.opencontainers.image.licenses="MIT AND GPL-3.0-or-later" \
      org.opencontainers.image.description="Upstream Rats Search 2.3.5 console server; Manticore source included in /usr/share/doc/manticore"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libqt6core6t64 libqt6gui6t64 libqt6widgets6t64 libqt6network6t64 \
    libqt6sql6t64 libqt6sql6-mysql libqt6concurrent6t64 libqt6websockets6 \
    libgl1 libxkbcommon0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/build/bin/RatsSearch ./
COPY --from=builder /src/imports/linux/x64/searchd /src/imports/linux/x64/indexer /src/imports/linux/x64/indextool ./
COPY --from=builder /src/LICENSE /usr/share/doc/rats-search/LICENSE
# Corresponding source, build scripts and component licenses for bundled Manticore
# 17.5.1 2c59803fe. This archive is also accessible without running the image.
ADD --chmod=0444 --checksum=sha256:55526381c020ae6e8597dd01b086e3cb013a7bd9167727156f7e5eddbc9da5d5 https://codeload.github.com/manticoresoftware/manticoresearch/tar.gz/2c59803fefb5512dd868fbf42aa759d78e617e74 /usr/share/doc/manticore/source.tar.gz
RUN tar -xzf /usr/share/doc/manticore/source.tar.gz -C /usr/share/doc/manticore --strip-components=1 --wildcards '*/LICENSE' '*/component-licenses/*' \
    && chmod +x /app/RatsSearch /app/searchd /app/indexer /app/indextool
RUN <<'SH'
cat > /usr/local/bin/start-rats <<'SCRIPT'
#!/bin/sh
set -eu
if [ ! -f /data/rats.json ]; then
    case "$RATS_INDEXER" in true|false) ;; *) echo 'RATS_INDEXER must be true or false' >&2; exit 1 ;; esac
    printf '{"restApi":true,"httpPort":8095,"indexer":%s,"upnp":false}' "$RATS_INDEXER" > /data/rats.json
fi
exec /app/RatsSearch --console --max-peers 30 --data-dir /data "$@"
SCRIPT
chmod +x /usr/local/bin/start-rats
SH
ENV QT_QPA_PLATFORM=offscreen RATS_INDEXER=true
VOLUME /data
EXPOSE 8095
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=30 CMD bash -c 'echo > /dev/tcp/127.0.0.1/8095'
ENTRYPOINT ["/usr/local/bin/start-rats"]
