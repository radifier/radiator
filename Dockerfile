FROM nvidia/cuda:11.7.0-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y git automake libssl-dev libcurl4-openssl-dev libjansson-dev
WORKDIR /root/radiator
