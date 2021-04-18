FROM mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-pytorch1.4-torchvision0.5-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest

ENV PYTHONIOENCODING=utf-8
ENV LC_CTYPE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV https_proxy="star-proxy.oa.com:3128"
ENV http_proxy="star-proxy.oa.com:3128"

ADD torch-1.5.1+cu101-cp36-cp36m-linux_x86_64.whl torch-1.5.1+cu101-cp36-cp36m-linux_x86_64.whl

RUN pip3 install --upgrade pip && \
    pip3 install sacrebleu && \
    pip3 install transformers==2.11.0 && \
    pip3 install faiss-gpu==1.6.1 && \
    pip3 install jsonlines && \
    pip3 install regex && \
    pip3 install sklearn && \
    pip3 install scipy && \
    pip3 install service_streamer && \
    pip3 install gunicorn && \
    pip3 install cached_property && \
    pip3 install networkx && \
    pip3 install penman>=1.1.0 && \
    pip3 install pytorch-ignite==0.4.4 && \
    pip3 install regex && \
    pip3 install smatch && \
    pip3 install transformers==4.4.2 && \
    pip3 install PyYAML>=5.1 && \
    pip3 install torch-1.5.1+cu101-cp36-cp36m-linux_x86_64.whl
    

