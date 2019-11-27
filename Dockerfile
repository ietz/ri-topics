# Fetch pretrained SBERT model separately
# Decouples changing the SBERT model from changes in environment.yml
FROM alpine:latest
ARG SBERT_MODEL_NAME=bert-base-wikipedia-sections-mean-tokens
RUN wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/$SBERT_MODEL_NAME.zip -O /root/sbert_model.zip && \
    mkdir /root/sbert_model && \
    unzip /root/sbert_model.zip -d /root/sbert_model && \
    rm /root/sbert_model.zip


FROM continuumio/miniconda3:4.7.12-alpine
# Setup conda environment
WORKDIR /home/anaconda
COPY ./environment.yml ./
RUN /opt/conda/bin/conda env create -f environment.yml

# Copy SBERT model
COPY --from=0 /root/sbert_model /home/anaconda/sbert_model
ENV SBERT_MODEL=/home/anaconda/sbert_model

# Copy code
COPY . .

# Run
EXPOSE 8888
CMD [ "/opt/conda/envs/ri-topics/bin/python", "./main.py" ]
