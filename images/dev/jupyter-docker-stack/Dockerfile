FROM  quay.io/jupyter/pytorch-notebook:cuda12-latest
LABEL maintainer="Motuzov <motuzov@gmail.com>"

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    vim tree && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# COPY setup.sh setup.sh


WORKDIR "${HOME}"
