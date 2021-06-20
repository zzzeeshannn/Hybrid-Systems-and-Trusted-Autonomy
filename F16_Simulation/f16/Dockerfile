# Dockerfile for f16 hw
#
# To build image:
# docker build . -t f16_image
#
# To run scripts and exit:
# docker run f16_image
#
# To get an interactive shell after building the image:
# docker run -i f16_image bash
#
# To delete docker container use: docker rm f16_image

FROM python:3.6

COPY ./requirements.txt /work/requirements.txt

# set working directory
WORKDIR /work

# install python package dependencies
RUN pip3 install -r requirements.txt

# copy remaining files to docker
COPY . /work

