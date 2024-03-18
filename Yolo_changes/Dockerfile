FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    poppler-utils \
    python3-dev

# add debian backports repo and install tesseract

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN mkdir /project
RUN git config --global --add safe.directory /project
WORKDIR /project

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
#RUN pip3 install --upgrade h5py

COPY classify/ classify/
COPY data/ data/
COPY models/ models/
COPY segment/ segment/
COPY utils/ utils/

EXPOSE 443
