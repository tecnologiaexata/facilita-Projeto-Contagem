# Facilita Projeto Coffee Backend

Worker de processamento para rodar na `Vast.ai`.

## Papel deste repositorio

- recebe importacao de galeria e inferencia
- processa imagens locais ou remotas por URL
- treina o modelo de segmentacao
- mantem heartbeat com o frontend/control plane na `Vercel`
- faz polling assincrono da fila de jobs do control plane
- sobe artefatos finais direto no `Blob`

## Convencao de anotacao

As classes agora seguem esta regra:

- `0 = fundo`
- `1 = coffee`
- `planta` nao e anotada manualmente

Tudo o que nao estiver anotado como `fundo` ou `coffee` passa a ser tratado como `planta` por exclusao.

## Endpoints principais

- `GET /api/health`
- `GET /api/meta`
- `GET /api/worker`
- `POST /api/gallery`
- `POST /api/gallery/from-url`
- `POST /api/inference`
- `POST /api/inference/from-url`
- `POST /api/training/run`

## Modelo assincrono

- o frontend na Vercel cria jobs no Postgres
- o worker da Vast.ai envia heartbeat para se registrar como `online`
- o mesmo worker consulta a fila em polling, busca o `context` do job e processa os jobs pendentes
- os arquivos de entrada sao baixados temporariamente e os artefatos finais vao para o `Blob`
- ao concluir, ele reporta `complete` ou `fail` para o control plane, que persiste apenas metadados/manifests no Postgres

## Ambiente

Copie as variaveis de `.env.example` e ajuste:

- `CONTROL_PLANE_URL`: URL do frontend na Vercel
- `WORKER_PUBLIC_URL`: URL publica atual do worker na Vast.ai
- `WORKER_SHARED_TOKEN`: token compartilhado com a Vercel
- `WORKER_JOB_POLL_ENABLED`: habilita o consumo assincrono da fila
- `WORKER_JOB_POLL_INTERVAL_SECONDS`: intervalo de polling dos jobs
- `BLOB_READ_WRITE_TOKEN`: token de escrita/leitura do Blob
- `BLOB_ACCESS`: `public` ou `private` para os artefatos definitivos
- `REMOTE_FETCH_ALLOWED_HOSTS`: lista opcional de hosts permitidos para baixar imagens/TXT

## Observacao

O diretório `frontend/` dentro deste repositorio continua aqui apenas como legado/local. O frontend principal da arquitetura nova fica no repositorio separado `Facilita-Projeto-Coffee-Frontend`.
