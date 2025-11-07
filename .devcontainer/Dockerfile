#--- From : https://github.com/PyMesh/PyMesh/issues/365
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1

# Update package list and install dependencies for building Python
RUN sed -i '1i deb-src http://archive.ubuntu.com/ubuntu/ jammy main' /etc/apt/sources.list
RUN apt-get update
RUN apt-get build-dep -y python3
RUN apt-get install -y \
    build-essential gdb lcov pkg-config \
    libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma lzma-dev tk-dev uuid-dev zlib1g-dev libmpdec-dev libzstd-dev
RUN apt-get install -y wget
RUN rm -rf /var/lib/apt/lists/*

# Download and install Python 3.11.13 from source (install to /usr/ with --prefix=/usr)
RUN wget https://www.python.org/ftp/python/3.11.13/Python-3.11.13.tgz \
    && tar -xf Python-3.11.13.tgz \
    && cd Python-3.11.13 \
    && ./configure --prefix=/usr --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && cd .. \
    && rm -rf Python-3.11.13 Python-3.11.13.tgz

# Set Python 3.11.13 as the default Python version (use higher priority to override system)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 100

# Ensure pip is installed and upgraded for Python 3.11
RUN python -m ensurepip \
    && python -m pip install --upgrade pip

#--------- Build and Install pymesh
# Install system dependencies for PyMesh and general use (minimized based on docs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone and init submodules (separated for caching)
RUN git clone https://github.com/PyMesh/PyMesh.git /tmp/PyMesh \
    && cd /tmp/PyMesh \
    && git submodule update --init

# Update Pybind11 submodule to v3.0.0 for Python 3.11 compatibility
RUN cd /tmp/PyMesh/third_party/pybind11 \
    && git remote add upstream https://github.com/pybind/pybind11.git \
    && git fetch upstream --tags \
    && git checkout v2.10.4 \
    && cd /tmp/PyMesh

# Patch setup.py to force CMake to use exact Python 3.11 executable and version
# Also patch setup.cfg to fix SetuptoolsDeprecationWarning (replace dash with underscore)
RUN sed -i "s/ -DPythonLibsNew_FIND_VERSION={}/ -DPython_FIND_VERSION=3.11 -DPython_FIND_VERSION_EXACT=ON -DPython_EXECUTABLE=\\/usr\\/bin\\/python3.11 -DPython_FIND_STRATEGY=LOCATION/g" /tmp/PyMesh/setup.py \
    && sed -i 's/description-file/description_file/g' /tmp/PyMesh/setup.cfg

# Patch and build third-party (with fixes for Draco and MMG)
RUN cd /tmp/PyMesh \
    && sed -i '1i #include <cstddef>' third_party/draco/src/draco/core/hash_utils.h \
    && sed -i '2i #include <cstdint>' third_party/draco/src/draco/core/hash_utils.h \
    && sed -i '1i #include <limits>' third_party/draco/src/draco/io/parser_utils.h \
    && sed -i '1i #include <limits>' third_party/draco/src/draco/io/parser_utils.cc \
    && cd third_party \
    && CFLAGS="-fcommon" CXXFLAGS="-fcommon" ./build.py all \
    && cd .. \
    && rm -rf third_party/build/*  # Clean third-party build dir to save space

# Main PyMesh build with -fcommon to fix multiple definitions in Triangle/Predicates
RUN cd /tmp/PyMesh \
    && CFLAGS="-fcommon" CXXFLAGS="-fcommon" python setup.py build \
    && python setup.py install \
    && cd / \
    && rm -rf /tmp/PyMesh

# Apply compatibility patches for Python 3.11
# RUN sed -i '1i import collections' /usr/lib/python3.11/site-packages/pymesh/__init__.py \
#     && sed -i '2i import collections.abc' /usr/lib/python3.11/site-packages/pymesh/__init__.py \
#     && sed -i '3i collections.Callable = collections.abc.Callable' /usr/lib/python3.11/site-packages/pymesh/__init__.py \
#     && sed -i '4i import numpy as np' /usr/lib/python3.11/site-packages/pymesh/__init__.py \
#     && sed -i '5i np.float = float' /usr/lib/python3.11/site-packages/pymesh/__init__.py
RUN sed -i '1i import collections' /usr/lib/python3.11/site-packages/pymesh2*/pymesh/__init__.py \
    && sed -i '2i import collections.abc' /usr/lib/python3.11/site-packages/pymesh*/pymesh/__init__.py \
    && sed -i '3i collections.Callable = collections.abc.Callable' /usr/lib/python3.11/site-packages/pymesh*/pymesh/__init__.py \
    && sed -i '4i import numpy as np' /usr/lib/python3.11/site-packages/pymesh*/pymesh/__init__.py \
    && sed -i '5i np.float = float' /usr/lib/python3.11/site-packages/pymesh*/pymesh/__init__.py

# Upgrade pip, setuptools, wheel, and install Python dependencies for PyMesh (NumPy, SciPy, Nose)
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
    numpy==1.24 \
    scipy==1.11 \
    nose \
    matplotlib==3.10.3

# Run tests to verify pymesh installation
RUN python -c "import pymesh; pymesh.test()"

#--------- Install other dependancies
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /volume

# Default command (e.g., start a shell or Jupyter if needed)
CMD ["bash"]
