FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y wget bzip2 ca-certificates curl git vim tmux g++ gcc build-essential cmake checkinstall gfortran libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev cmake-curses-gui


RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install

RUN cd / &&\
    git clone https://github.com/opencv/opencv &&\
    cd opencv &&\
    git checkout 3.4.15 &&\
    cd / && git clone https://github.com/opencv/opencv_contrib.git &&\
    cd opencv_contrib &&\
    git checkout 3.4.15 &&\
    cd /opencv &&\
    mkdir build


RUN cd /opencv/build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_STUBS=OFF -DBUILD_DOCS=OFF -DWITH_MATLAB=OFF -Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF -DCUDA_FAST_MATH=ON  -DMKL_WITH_OPENMP=ON -DOPENCV_ENABLE_NONFREE=ON -DWITH_OPENMP=ON -DWITH_QT=ON -WITH_OPENEXR=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_opencv_cudacodec=OFF -DINSTALL_PYTHON_EXAMPLES=OFF  -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DCMAKE_CXX_FLAGS=-std=c++11 -DENABLE_CXX11=OFF  -DBUILD_opencv_xfeatures2d=ON -DOPENCV_DNN_OPENCL=OFF -DWITH_CUDA=ON -DWITH_OPENCL=OFF &&\
    make -j6 &&\
    make install &&\
    cd /opencv/build && make install

RUN cd / &&\
    git clone https://github.com/PointCloudLibrary/pcl

RUN cd /pcl &&\
    git checkout pcl-1.10.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_apps=OFF -DBUILD_GPU=OFF  -DBUILD_CUDA=OFF -DBUILD_examples=OFF -DBUILD_global_tests=OFF -DBUILD_simulation=OFF -DCUDA_BUILD_EMULATION=OFF -DCMAKE_CXX_FLAGS=-std=c++11 -DPCL_ENABLE_SSE=ON -DPCL_SHARED_LIBS=ON &&\
    make -j6 &&\
    make install

RUN apt install -y libzmq3-dev freeglut3-dev

RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install

RUN cd / && git clone https://github.com/jbeder/yaml-cpp &&\
    cd yaml-cpp && git checkout yaml-cpp-0.7.0 &&\
    mkdir build && cd build && cmake .. -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DINSTALL_GTEST=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_BUILD_SHARED_LIBS=ON &&\
    make -j6 && make install

SHELL ["/bin/bash", "--login", "-c"]

RUN cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    /bin/bash -c "source ~/.bashrc" && \
    /opt/conda/bin/conda update -n base -c defaults conda -y &&\
    /opt/conda/bin/conda config --set ssl_verify no && \
    /opt/conda/bin/conda config --add channels conda-forge &&\
    /opt/conda/bin/conda create -n py38 python=3.8


ENV PATH $PATH:/opt/conda/envs/py38/bin


RUN conda init bash &&\
    echo "conda activate py38" >> ~/.bashrc &&\
    conda activate py38 &&\
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 &&\
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" &&\
    pip install trimesh opencv-python wandb matplotlib imageio tqdm open3d ruamel.yaml sacred kornia pymongo pyrender jupyterlab ninja &&\
    conda install -y -c anaconda scipy


RUN cd / && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin


ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
ENV OPENCV_IO_ENABLE_OPENEXR=1


RUN imageio_download_bin freeimage

RUN conda activate py38 && cd /kaolin &&\
    # sed -i "223i\    extra_compile_args['nvcc'] += ['-gencode=arch=compute_52,code=sm_52', '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61', '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_80,code=sm_80', '-gencode=arch=compute_80,code=compute_80']" setup.py &&\
    FORCE_CUDA=1 python setup.py develop

#### Kaolin will change numpy version
RUN pip install transformations einops scikit-image awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui pytinyrenderer PyQt5 cython-npm chardet openpyxl

RUN apt-get update --fix-missing && \
    apt install -y rsync lbzip2 pigz zip p7zip-full p7zip-rar
