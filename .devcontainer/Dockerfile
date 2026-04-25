FROM ubuntu:questing AS build

# Pull in uv from the official repo
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /uvx /bin/

ENV DEBIAN_FRONTEND="noninteractive"
ENV PATH="/opt/python-ft/bin:$PATH"

RUN apt-get update \
    \
    && `# Start with all normal ubuntu packages` \
    && apt-get install -y --no-install-recommends apt-utils \
         ca-certificates \
         software-properties-common \
         lsb-release \
         gnupg2 \
         wget \
         git \
         make \
         ninja-build \
         cmake \
         valgrind \
         strace \
         heaptrack \
         cppcheck \
         libc6-dev \
         zlib1g-dev \
         libfmt-dev \
         libzstd-dev \
         python3 \
         python3-pip \
         python3-dev \
         python3-virtualenv \
         libbpf-dev \
         libbpfcc-dev \
         linux-tools-common \
         linux-tools-generic \
         linux-perf \
         libbenchmark-dev \
         gcc-15 \
         g++-15 \
         gdb \
         llvm-21 \
         clang-21 \
         lldb-21 \
         python3-clang-21 \
         llvm-21-dev \
         libclang-21-dev \
         libclang-rt-21-dev \
         liblldb-21-dev \
         clang-tidy-21 \
         clang-format-21 \
         clang-tools-21 \
         googletest \
    \
     `# Configure the latest version of GCC as the default` \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 1000 \
    && update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-15 1000 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 1000 \
    \
    && `# Configure the latest version of LLVM as the default` \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 1000 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 1000 \
    && update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-21 1000 \
    && update-alternatives --install /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-21 1000 \
    && update-alternatives --install /usr/bin/opt opt /usr/bin/opt-21 1000 \
    \
    && `# Final apt cleanup` \
    && rm -rf /var/lib/apt/lists/* \
    \
    && `# Record versions in the build` \
    && gcc --version \
    && clang --version \
    && uv --version \
    && cmake --version


COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --timeout 3000 \
    -r /tmp/requirements.txt

RUN uv python install 3.14 3.14+freethreaded
