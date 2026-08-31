FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDA_OXIDE_LLC=/usr/bin/llc-21 \
    LIBCLANG_PATH=/usr/lib/llvm-21/lib \
    LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 \
    LLVM_CONFIG=/usr/local/bin/llvm-config-7 \
    LLVM_LINK_STATIC=1 \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/home/ubuntu/.cargo \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_TOOL_DIR=/opt/uv/tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_PROJECT_ENVIRONMENT=/opt/astrodrift-venv \
    PATH=/home/ubuntu/.cargo/bin:/home/ubuntu/.local/bin:/usr/lib/llvm-21/bin:/usr/local/cuda/bin:${PATH} \
    TERM=xterm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gnupg \
        libedit-dev \
        libffi-dev \
        libfontconfig-dev \
        libglib2.0-0 \
        libgsl-dev \
        libncurses-dev \
        libssl-dev \
        libx11-xcb-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxml2-dev \
        libxrandr-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        sudo \
        wget \
        xz-utils \
        zlib1g-dev \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
        > /etc/apt/sources.list.d/llvm-toolchain-noble-21.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-21 \
        libclang-common-21-dev \
        lld-21 \
        llvm-21 \
        llvm-21-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Rust-CUDA emits the legacy LLVM 7 NVVM dialect. Install it separately so
# cuda-oxide continues to use the LLVM 21 tools above.
WORKDIR /tmp/llvm-7-build
RUN curl -fsSLO https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz \
    && tar -xf llvm-7.1.0.src.tar.xz \
    && cmake -S llvm-7.1.0.src -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm-7 \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_ENABLE_ZLIB=ON \
        -DLLVM_ENABLE_TERMINFO=ON \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build \
    && ln -s /opt/llvm-7/bin/llvm-config /usr/local/bin/llvm-config-7 \
    && cd / \
    && rm -rf /tmp/llvm-7-build

ARG UV_VERSION=0.12.7
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /tmp/uv-installer.sh
RUN UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-installer.sh \
    && rm /tmp/uv-installer.sh

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb
RUN wget -q "${NSYS_URL}${NSYS_PKG}" \
    && dpkg -i "${NSYS_PKG}" \
    && rm "${NSYS_PKG}"

RUN curl -fsSL https://sh.rustup.rs \
        | sh -s -- --default-toolchain none --profile minimal -y \
    && rustup toolchain install nightly-2026-08-28 \
        --profile minimal \
        --component rust-src \
        --component rustc-dev \
        --component rust-analyzer \
        --component rustfmt \
        --component clippy \
        --component llvm-tools

# Separately pinned Rust toolchain from the manifest.
WORKDIR /tmp/astrodrift-toolchain
RUN --mount=type=bind,source=rust-toolchain.toml,target=/tmp/astrodrift-toolchain/rust-toolchain.toml \
    rustup show >/dev/null

RUN uv python install 3.13.15 \
    && uv tool install maturin@1.15.0 \
    && uv tool install ruff@0.16.5 \
    && uv tool install ty@0.0.75

WORKDIR /tmp/astrodrift-python
RUN install -d -m 0777 /opt/astrodrift-venv
RUN --mount=type=bind,source=uv.lock,target=/tmp/astrodrift-python/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/tmp/astrodrift-python/pyproject.toml \
    umask 000 \
    && uv sync --locked --no-install-project

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000
RUN set -eux; \
    existing_group="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    if [[ -z "$existing_group" ]]; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    elif [[ "$existing_group" != "${USERNAME}" ]]; then \
        groupmod -n "${USERNAME}" "$existing_group"; \
    fi; \
    existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    if [[ -z "$existing_user" ]]; then \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" \
            --home-dir /home/ubuntu --create-home --shell /bin/bash "${USERNAME}"; \
    elif [[ "$existing_user" != "${USERNAME}" ]]; then \
        usermod -l "${USERNAME}" -d /home/ubuntu -m "$existing_user"; \
    fi; \
    usermod -g "${USER_GID}" -s /bin/bash -d /home/ubuntu "${USERNAME}"; \
    install -d -m 0755 -o "${USER_UID}" -g "${USER_GID}" /home/ubuntu; \
    chown -R "${USER_UID}:${USER_GID}" /home/ubuntu/.cargo; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

USER ubuntu
ENV HOME=/home/ubuntu

WORKDIR /workspaces/astrodrift
CMD ["/bin/bash"]
