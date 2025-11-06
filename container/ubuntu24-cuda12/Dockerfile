FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
SHELL ["/bin/bash","-c"]

WORKDIR /data/home/leo

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    clang \
    curl \
    libssl-dev \
    libtinfo-dev \
    pkg-config \
    xz-utils \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Needed to build `path_tracer`, `optix/ex03_window` example
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    cmake \
    libfontconfig-dev \
    libx11-xcb-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev && \
    rm -rf /var/lib/apt/lists/*

# Get LLVM 7
WORKDIR /data/llvm7

# Install dependencies for building LLVM
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    libffi-dev \
    libedit-dev \
    libncurses5-dev \
    libxml2-dev \
    python3 \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Download and build LLVM 7.1.0 for all architectures
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

# Get Rust
ENV HOME=/data/home/leo
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- --default-toolchain none -y
ENV PATH="/data/home/leo/.cargo/bin:$PATH"

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb

RUN apt-get update && apt install -y wget libglib2.0-0
RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

ADD https://astral.sh/uv/0.9.7/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/data/home/leo/.local/bin/:$PATH"
RUN uv python install 3.13
RUN uv tool install maturin

WORKDIR /data/astrodrift
RUN --mount=type=bind,source=rust-toolchain.toml,target=/data/astrodrift/rust-toolchain.toml \
    rustup show


RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# RUN --mount=type=bind,source=Cargo.toml,target=Cargo.toml \
#     cargo fetch  # Or cargo build --release for a binary
# WORKDIR /data/astrodrift/kernels
# RUN --mount=type=bind,source=kernels/Cargo.toml,target=Cargo.toml \
#     cargo fetch # Or cargo build --release for a binary


# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
# ENV RUST_LOG=info
