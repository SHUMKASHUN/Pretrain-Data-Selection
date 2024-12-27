FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    wget \ 
    git-lfs \
    && git lfs install \
    && apt-get clean

RUN pip install nltk sentencepiece transformers==4.40.1 wandb -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh  \
    && bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /root/anaconda3 \
    && rm Anaconda3-2024.10-1-Linux-x86_64.sh 

ENV PATH="${PATH}:/root/anaconda3/bin"


RUN conda create -n datatrove python=3.10 -y \
    && conda init bash \
    && source activate datatrove \
    && pip install datatrove[processing] orjson -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda create -n lm-eval python=3.10 -y \
    && source activate lm-eval \
    && pip install datasets==2.19.0 transformers==4.41.2 tqdm torch==2.3.0 wandb numpy sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && source deactivate
