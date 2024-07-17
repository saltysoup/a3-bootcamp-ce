FROM nvcr.io/nvidia/nemo:24.05
#FROM nvcr.io/nvidia/nemo:24.03.01.framework
WORKDIR /workspace

# GCSfuse components (used to provide shared storage, not intended for high performance)
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
  && echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get install --yes google-cloud-cli \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && mkdir /gcs

RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

COPY enable-step-times-2405.patch /opt/NeMo/enable-step-times-2405.patch
RUN cd /opt/NeMo/ && git apply enable-step-times-2405.patch 
