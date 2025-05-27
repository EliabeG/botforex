# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. Instalar dependências de sistema necessárias para compilação e outras bibliotecas
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
        gcc \
        g++ \
        make \
        curl \
        libhdf5-dev \
        libxml2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Baixar e compilar TA-Lib v0.6.4 a partir do código-fonte
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# 3. Copiar o arquivo de requirements.txt
COPY ./requirements.txt /app/requirements.txt

# 4. Instalar as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo o código do projeto para /app/ no container
COPY . /app/

# 6. Definir a variável de ambiente para o diretório base da aplicação
ENV APP_BASE_DIR=/app

# 7. Comando padrão para executar quando o container iniciar
CMD ["python", "main.py"]