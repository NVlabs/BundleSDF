docker rm -f bundlesdf
DIR=$(pwd)/../
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name bundlesdf  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE nvcr.io/nvidian/bundlesdf:latest bash
