# Define a builder image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

LABEL name="Hwan"
LABEL version="1.0"
LABEL repository="semantic-search-lib"

## SHELL
SHELL ["/bin/bash", "-c"]


ENV LANG C.UTF-8

## PROXY 설정

ENV http_proxy=http://172.18.171.132:3128
ENV https_proxy=http://172.18.171.132:3128

## WORKDIR

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip

COPY ./requirements.txt ${WORKDIR}/requirements.txt
RUN pip install --no-cache-dir --upgrade -r ${WORKDIR}/requirements.txt 

RUN git clone https://github.com/NVIDIA/apex

# Change the directory to apex and install using pip
RUN cd apex && \
    pip3 install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install requirements txt 

ENV CODE_DIR /app/code
ENV PYTHONPATH "${PYTHONPATH}:${CODE_DIR}"
WORKDIR $CODE_DIR

## python package

CMD [ "/bin/bash" ]