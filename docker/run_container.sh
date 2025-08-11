docker rm -f bundlesdf
DIR=$(pwd)/../
xhost +  && docker run --gpus all --device /dev/bus/usb:/dev/bus/usb --privileged --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name bundlesdf  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE yufeiyanggao/push_tracking:latest bash
