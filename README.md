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
- `WORKER_DEFAULT_YOLO_DEVICE`: GPU padrao usada pelo YOLO, por exemplo `0`
- `WORKER_DEFAULT_YOLO_MODEL`: caminho local opcional para um checkpoint `.pt` ou para uma pasta com pesos; pode ser usado como base no treino e como fallback na inferencia
- `.worker-default-yolo-model`: arquivo versionado opcional na raiz do repositorio com um caminho local de pesos; e usado quando `WORKER_DEFAULT_YOLO_MODEL` nao vier definido no `.env`
- `INFERENCE_PROVIDER`: provider padrao da inferencia, `local_yolo` ou `roboflow`; a tela tambem pode enviar o provider por job
- `ROBOFLOW_API_KEY`, `ROBOFLOW_API_URL`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW`: configuracao do Workflow Roboflow usado quando o provider for `roboflow`
- `ROBOFLOW_CLASSES`, `ROBOFLOW_CLASSES_PARAMETER`, `ROBOFLOW_CONFIDENCE`, `ROBOFLOW_CONFIDENCE_PARAMETER`, `ROBOFLOW_MAX_IMAGE_SIDE`, `ROBOFLOW_USE_CACHE`, `ROBOFLOW_TIMEOUT_SECONDS`: parametros enviados ao Workflow e pre-processamento da imagem
- `PYTORCH_INSTALL_MODE`: `cuda`, `auto`, `cpu` ou `skip` para controlar como o `workerctl.sh` instala o PyTorch. O padrao atual do projeto e `cuda`
- `PYTORCH_INDEX_URL`: indice PyTorch usado quando o modo CUDA estiver ativo. Se ficar vazio, o `workerctl.sh` escolhe automaticamente `cu128` em GPUs Blackwell/B200 e `cu124` nas demais GPUs NVIDIA
- `WORKER_SHARED_TOKEN`: token compartilhado com o frontend
- `BLOB_READ_WRITE_TOKEN`: token do Blob para leitura/escrita dos artefatos
- `BLOB_ACCESS`: `public` ou `private`
- `WORKER_JOB_STUCK_AFTER_SECONDS`: limite sem progresso para marcar o job como provavelmente travado no `/api/worker`

Treino exige GPU CUDA funcional. A inferencia local YOLO usa GPU quando disponivel e pode cair para CPU; a inferencia Roboflow roda o modelo no Roboflow e pode ser processada pelo worker sem GPU.

Se voce ja baixou os pesos do YOLO manualmente, pode apontar o worker para eles no `.env`, por exemplo:

```env
WORKER_DEFAULT_YOLO_MODEL=C:\Users\Michael - Facilita\Desktop\pesos\pesos_atualizado
```

Tambem e possivel versionar um default para a equipe na raiz do repositorio, usando o arquivo `.worker-default-yolo-model`. O worker so usa esse arquivo quando o caminho existir localmente e quando `WORKER_DEFAULT_YOLO_MODEL` nao tiver sido definido no ambiente.

Com isso:

- o treino usa esse checkpoint local como modelo base quando `base_model` nao vier no job
- a inferencia consegue usar esse `.pt` como fallback quando nao houver modelo ativo vindo do control plane
- se `WORKER_DEFAULT_YOLO_MODEL` apontar para uma pasta, o worker prioriza o `weights.pt` mais recente dentro dela; se nao existir `weights.pt`, usa o `.pt` mais novo encontrado

## Fluxo

1. O frontend recebe upload e salva a entrada no Blob.
2. O frontend cria o job no Postgres.
3. O worker envia heartbeat e faz polling da fila.
4. O worker busca o `context` do job, baixa o necessario, processa e sobe os resultados no Blob.
5. O worker reporta o resultado final para o frontend.

## Operacao automatizada na Vast.ai

Agora o repositorio tem scripts para preparar e subir o worker automaticamente lendo o `.env` na raiz do projeto.

Arquivos principais:

- [workerctl.sh](/home/renatoolegrio/Documentos/GitHub/facilita-Projeto-Coffee/scripts/workerctl.sh)
- [vast_onstart.sh](/home/renatoolegrio/Documentos/GitHub/facilita-Projeto-Coffee/scripts/vast_onstart.sh)
- [onstart.sh](/home/renatoolegrio/Documentos/GitHub/facilita-Projeto-Coffee/onstart.sh)

Comandos uteis:

```bash
bash scripts/workerctl.sh bootstrap
bash scripts/workerctl.sh start
bash scripts/workerctl.sh stop
bash scripts/workerctl.sh restart
bash scripts/workerctl.sh status
bash scripts/workerctl.sh health
bash scripts/workerctl.sh logs
```

O `workerctl.sh`:

- le o `.env` automaticamente
- cria `.venv` se ainda nao existir
- instala `torch/torchvision/torchaudio` com wheel CUDA quando houver GPU detectada ou quando `PYTORCH_INSTALL_MODE=cuda`
- em GPUs Blackwell como `B200`, promove automaticamente o indice oficial do PyTorch para `cu128` mesmo se o `.env` antigo ainda estiver apontando para `cu124`
- instala dependencias apenas quando `backend/requirements.txt` mudar
- sobe o worker em background
- grava logs em `logs/worker.log`
- guarda o pid em `run/worker.pid`

### Auto-start quando a instancia ligar de novo

Na Vast.ai, o caminho recomendado e configurar o comando de `On-start script` da instancia para apontar para:

```bash
bash /workspace/facilita-Projeto-Coffee/onstart.sh
```

Se o repositorio estiver em outro diretorio, ajuste o caminho. O `onstart.sh` chama `scripts/vast_onstart.sh`, que por sua vez usa `workerctl.sh start`.

Assim, quando a instancia voltar a ligar, o worker sobe automaticamente de novo com o `.env` ja presente na pasta do projeto.

### O que voce precisa deixar pronto na instancia

- o repositorio clonado
- o `.env` na raiz do projeto
- a porta do worker exposta na Vast.ai, por exemplo `8050`
- `WORKER_PUBLIC_URL` apontando para o IP publico e porta da instancia

Exemplo:

```env
WORKER_PUBLIC_URL=http://SEU_IP_PUBLICO:8050
```
