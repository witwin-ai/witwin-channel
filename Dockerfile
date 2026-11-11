# Copyright Xingyu Chen.
# Builds the witwin-channel wheel from this checkout and a runtime image that installs it.
#
#   docker build -t witwin-channel .                                  # all release GPU targets
#   docker build --build-arg CUDA_ARCHITECTURES=89-real -t witwin-channel .   # one GPU, faster
#   docker build --build-context rayd=../RayD -t witwin-channel .     # a local RayD checkout
#   docker run --rm --gpus all witwin-channel                         # prints the doctor report
#
# The build context must be a Git checkout (the build records its commit). RayD is cloned at the
# commit in dependencies/rayd.lock.json unless a `rayd` build context replaces that stage; it must
# be a Git checkout of that commit with its origin set to the locked repository URL. The OptiX 8.1
# SDK headers are fetched from NVIDIA/optix-dev at build time; no OptiX library is installed.
#
# At run time the OptiX runtime comes from the host driver. The NVIDIA container toolkit mounts it
# only with the `graphics` driver capability, which this image requests. Without it, or with a
# driver older than R555, scenes trace on the slower pure-CUDA backend.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04

FROM ${CUDA_IMAGE} AS toolchain
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

FROM toolchain AS rayd
COPY dependencies/rayd.lock.json /tmp/rayd.lock.json
RUN set -eu; \
    url=$(python3 -c "import json; print(json.load(open('/tmp/rayd.lock.json'))['repository_url'])"); \
    commit=$(python3 -c "import json; print(json.load(open('/tmp/rayd.lock.json'))['commit'])"); \
    git init -q /rayd && cd /rayd && git remote add origin "$url" \
    && git fetch -q --depth 1 origin "$commit" && git checkout -q FETCH_HEAD

FROM toolchain AS optix
RUN git clone -q --depth 1 --branch v8.1.0 https://github.com/NVIDIA/optix-dev.git /optix

FROM toolchain AS build
ARG TORCH_VERSION=2.10.0
ARG CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
ARG BUILD_JOBS=4
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --no-cache-dir "cmake>=4.3,<4.4" "ninja>=1.13,<1.14" "scikit-build-core>=0.12,<0.13" \
    && pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
COPY --from=rayd /rayd /rayd
COPY --from=optix /optix/include /optix/include
COPY . /src
WORKDIR /src
RUN git config --global --add safe.directory /src && git config --global --add safe.directory /rayd \
    && CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
       CMAKE_ARGS="-DRAYD_SOURCE_DIR=/rayd -DOPTIX_INCLUDE_DIR=/optix/include -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
       pip wheel . --no-build-isolation --no-deps -w /dist

FROM ${RUNTIME_IMAGE} AS runtime
ARG TORCH_VERSION=2.10.0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
COPY --from=build /dist /dist
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir /dist/*.whl
CMD ["python", "-m", "witwin.channel", "doctor"]
