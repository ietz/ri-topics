FROM continuumio/miniconda3:4.7.12-alpine

WORKDIR /usr/cmd/app
COPY ./environment.yml ./
RUN /opt/conda/bin/conda env create -f environment.yml

COPY . .

EXPOSE 8888
CMD [ "/opt/conda/bin/conda", "run", "-n", "ri-topics", "python", "./ri_topics/server.py" ]
