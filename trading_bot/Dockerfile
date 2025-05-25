# Dockerfile
FROM python:3.12-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    python3-dev \
    build-essential \
    libta-lib0-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p /app/data/ticks /app/logs /app/models

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Expor porta para métricas Prometheus
EXPOSE 9090

# Comando padrão
CMD ["python", "main.py", "--mode", "live"]