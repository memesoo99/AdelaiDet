FROM nvidia/cuda:10.2-devel-ubuntu18.04
 
RUN apt update && apt install -y libglib2.0-0 && apt clean
RUN apt install -y wget gcc-7 g++-7 vim curl git libsm6 libgl1-mesa-glx libxext6 libxrender-dev lsb-core libglib2.0-0 libjpeg-dev zlib1g-dev software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
ARG DEBIAN_FRONTEND=noninterative
RUN apt install -y python3.9 python3.9-distutils python3.9-dev libpython3.9-dev python3-setuptools

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install pythran --upgrade setuptools
RUN python3.9 -m pip install 'git+https://github.com/lazybuttrying/detectron2.git'

WORKDIR /root
RUN mkdir code
WORKDIR code

RUN git clone https://github.com/lazybuttrying/AdelaiDet.git adet
WORKDIR adet
RUN python3.9 setup.py build develop
RUN wget -P ./training_dir/BoxInst_MS_R_50_1x/ https://modelfiles-bucket.s3.ap-northeast-2.amazonaws.com/model_final.pth

RUN pip install opencv-python skimage fastapi pydantic uvicorn
# fastapi>=0.68.0 pydantic>=1.8.0 uvicorn>=0.15.0
