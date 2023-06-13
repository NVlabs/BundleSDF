ROOT=$(pwd)
cd /kaolin && pip install -e .
cd ${ROOT}/mycuda && rm -rf build *egg* && pip install -e .
cd ${ROOT}/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j11