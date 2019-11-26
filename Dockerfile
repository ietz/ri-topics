FROM continuumio/miniconda3:4.7.12-alpine

WORKDIR /home/anaconda
COPY ./environment.yml ./
RUN /opt/conda/bin/conda env create -f environment.yml

COPY . .

EXPOSE 8888
CMD [ "/opt/conda/envs/ri-topics/bin/python", "./main.py" ]
