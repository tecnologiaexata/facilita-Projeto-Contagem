# Facilita Projeto Coffee Backend

Worker Python puro para rodar na `Vast.ai`, sem Docker.

## Papel deste repositorio

- atuar como worker da fila assincrona
- buscar jobs no frontend/control plane
- baixar entradas do `Blob`
- processar importacao de anotacoes, treino e inferencia
- subir os artefatos finais direto no `Blob`
- reportar `complete` ou `fail` para o frontend

O frontend principal da arquitetura fica no repositorio separado `Facilita-Projeto-Coffee-Frontend`.

## Convencao de anotacao

- `0 = fundo`
- `1 = coffee`
- `planta` nunca e anotada manualmente

Tudo o que nao for anotado como `fundo` ou `coffee` vira `planta` por exclusao.

## Endpoints disponiveis

- `GET /api/health`
- `GET /api/meta`
- `GET /api/worker`
- `GET /api/monitoring`

## Como rodar sem Docker

1. Crie um ambiente virtual.
2. Instale as dependencias de [backend/requirements.txt](/home/renatoolegrio/Documentos/GitHub/facilita-Projeto-Coffee/backend/requirements.txt).
3. Copie [.env.example](/home/renatoolegrio/Documentos/GitHub/facilita-Projeto-Coffee/.env.example) para `.env`.
4. Suba o worker com:

```bash
python3 run_worker.py --port 8050
```

Se preferir, tambem pode controlar por ambiente:

```bash
HOST=0.0.0.0 PORT=8050 python3 run_worker.py
```

Use `venv` por padrao. Rodar fora dela pode misturar `numpy/scipy/sklearn` do sistema com pacotes instalados pelo `pip`, o que costuma quebrar o worker.

## Variaveis principais

- `HOST`: host HTTP do worker
- `PORT`: porta HTTP do worker
- `RELOAD`: ativa reload local
- `LOG_LEVEL`: nivel de log do Uvicorn
- `CONTROL_PLANE_URL`: URL do frontend/control plane
- `WORKER_PUBLIC_URL`: URL publica do worker usada no registro/heartbeat
- `WORKER_SHARED_TOKEN`: token compartilhado com o frontend
- `BLOB_READ_WRITE_TOKEN`: token do Blob para leitura/escrita dos artefatos
- `BLOB_ACCESS`: `public` ou `private`

## Fluxo

1. O frontend recebe upload e salva a entrada no Blob.
2. O frontend cria o job no Postgres.
3. O worker envia heartbeat e faz polling da fila.
4. O worker busca o `context` do job, baixa o necessario, processa e sobe os resultados no Blob.
5. O worker reporta o resultado final para o frontend.
