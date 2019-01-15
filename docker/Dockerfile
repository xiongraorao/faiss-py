FROM ubuntu:16.04

ENV TZ=Asia/Shanghai HOME=/home/app

WORKDIR $HOME

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y \
    && apt install -y git

COPY miniconda3.tar.gz $HOME

RUN tar -xvzf miniconda3.tar.gz && rm miniconda3.tar.gz

RUN git clone https://github.com/xiongraorao/faiss-py.git

VOLUME ["$HOME/faiss-py/index/"]

EXPOSE 2344

WORKDIR $HOME/faiss-py

ENTRYPOINT ["/home/app/miniconda3/bin/python", "app.py"]