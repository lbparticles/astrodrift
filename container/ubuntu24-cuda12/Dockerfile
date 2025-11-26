FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
SHELL ["/bin/bash","-c"]

WORKDIR /data/home/leo

COPY container/shared/apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
        grep -vE '^\s*#|^\s*$' /tmp/apt-packages.txt \
        | xargs -r apt-get -qq -y install \
    && rm -rf /var/lib/apt/lists/*

# Download and build LLVM 7.1.0 for all architectures
WORKDIR /data/llvm7
RUN curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz && \
    tar -xf llvm-7.1.0.src.tar.xz && \
    cd llvm-7.1.0.src && \
    mkdir build && cd build && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        TARGETS="X86;NVPTX"; \
    else \
        TARGETS="AArch64;NVPTX"; \
    fi && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="$TARGETS" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_ENABLE_ZLIB=ON \
        -DLLVM_ENABLE_TERMINFO=ON \
        -DCMAKE_INSTALL_PREFIX=/usr \
        .. && \
    ninja -j$(nproc) && \
    ninja install && \
    cd ../.. && \
    rm -rf llvm-7.1.0.src* && \
    ln -s /usr/bin/llvm-config /usr/bin/llvm-config-7

# Get Rustup
ENV HOME=/data/home/leo
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- --default-toolchain none -y
ENV PATH="/data/home/leo/.cargo/bin:$PATH"

# Get NSight Systems
ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb
RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

# Get uv
ADD https://astral.sh/uv/0.9.7/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/data/home/leo/.local/bin/:$PATH"

# Install Python
RUN uv python install 3.13.9
RUN uv tool install maturin@1.9.6
RUN uv tool install ruff@0.14.4
RUN uv tool install ty@0.0.1a25

# Download Rust toolchain
WORKDIR /data/astrodrift
RUN --mount=type=bind,source=rust-toolchain.toml,target=/data/astrodrift/rust-toolchain.toml \
    rustup show

# Sync uv packages
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
# ENV RUST_LOG=info

# Create a non-root user for devcontainers
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    existing_group="$(getent group ${USER_GID} | cut -d: -f1 || true)"; \
    if [ -z "$existing_group" ]; then \
        echo "Creating group ${USERNAME} with GID ${USER_GID}"; \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    else \
        if [ "$existing_group" != "${USERNAME}" ]; then \
            echo "Renaming group $existing_group to ${USERNAME}"; \
            groupmod -n "${USERNAME}" "$existing_group"; \
        else \
            echo "Group ${USERNAME} already exists"; \
        fi; \
    fi; \
    \
    existing_user="$(getent passwd ${USER_UID} | cut -d: -f1 || true)"; \
    if [ -z "$existing_user" ]; then \
        echo "Creating user ${USERNAME} with UID ${USER_UID}"; \
        useradd \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --home-dir /data/home/leo \
            --create-home \
            --shell /bin/bash \
            "${USERNAME}"; \
    else \
        if [ "$existing_user" != "${USERNAME}" ]; then \
            echo "Renaming user $existing_user to ${USERNAME}"; \
            usermod -l "${USERNAME}" "$existing_user"; \
        else \
            echo "User ${USERNAME} already exists"; \
        fi; \
        usermod -g "${USER_GID}" "${USERNAME}"; \
        if [ ! -d /data/home/leo ]; then \
            usermod -d /data/home/leo -m "${USERNAME}"; \
        else \
            usermod -d /data/home/leo "${USERNAME}"; \
        fi; \
    fi; \
    \
    mkdir -p /data/home/leo; \
    chown -R "${USER_UID}:${USER_GID}" /data/home/leo
ENV TERM=xterm
