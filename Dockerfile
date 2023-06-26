FROM python:3.8

EXPOSE 5006

RUN apt-get update &&  apt-get install libjpeg-dev zlib1g-dev \
  && apt-get -yq autoremove \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install nodejs (see: https://askubuntu.com/a/720814)
RUN curl -sL https://deb.nodesource.com/setup_18.x | bash \
    && apt-get install nodejs \
    && apt-get -yq autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf ~/.cache/pip

# Add entrypoint (this allows variable expansion)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ORIGIN="incognitive.paris-reinforce.epu.ntua.gr" PORT="5006" PREFIX="" LOG_LEVEL="info"

COPY ./app /app
ENTRYPOINT ["./entrypoint.sh"]