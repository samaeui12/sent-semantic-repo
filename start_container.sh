#!/bin/bash

set -e
TODAY=$(date +'%Y%m%d')

:<<'END'
    $0: 매개 변수의 총 개수
    -gt: greater than
    shift: 첫번째 파라미터 버리고 $# <- ($#-1)
END

function usage() {
    echo "usage: $0 [option]"
    echo "-h    help"
    echo "-i   image name "
    echo "-c   container name "
    echo "-ngpu    gpu number "
    echo "-d    target date"
    exit 1
}

while test $# -gt 0;
do
    case "$1" in
        -h)
            usage
        ;;
        -iname) 
            shift
            IMAGE_NAME=$1
            shift ;;
        -cname) shift
            CONTAINER_NAME=$1
            shift ;;
        -ngpu)
            shift
            GPU_NUMBER=$1
            shift ;;
        -mode)
            shift
            MODE=$1
            shift ;;
    esac
done

if [[ -z ${IMAGE_NAME} ]]; then
    IMAGE_NAME='registry.tde.sktelecom.com/aisearch/docker-model-images/semantic-search-lib:latest'
fi

if [[ -z ${MODE} ]]; then
    MODE='stg'
fi

if [[ -z ${GPU_NUMBER} ]]; then
    GPU_NUMBER=-1
fi

if [[ -z ${DATE} ]]; then
    DATE=$TODAY
fi

if [[ -z ${CONTAINER_NAME} ]]; then
    CONTAINER_NAME='hwan'
fi

TODAY_CONTAINER_NAME="if_kill_war_begin"
MEMORY='80g'

echo "#######################"
echo $IMAGE_NAME
echo $GPU_NUMBER
echo $CONTAINER_NAME
echo $DATE
echo $TODAY_CONTAINER_NAME
echo "#######################"

{ # try

    docker rm $TODAY_CONTAINER_NAME && echo "Successfully rm container $TODAY_CONTAINER_NAME"
}   || { # catch
    echo "CHECKED NO CONTAINER [$TODAY_CONTAINER_NAME] EXISTENCE"
}


sudo docker run -it --name $TODAY_CONTAINER_NAME \
-e NVIDIA_VISIBLE_DEVICES='0,1' \
--ip 0.0.0.0 \
--memory='80g'\
-p 40004:6006 \
-p 40002:8888 \
-v /app/data/air-cupid:/app/data -v /app/service/semantic-search-lib:/app/code \
$IMAGE_NAME


{
    jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=8888
}
