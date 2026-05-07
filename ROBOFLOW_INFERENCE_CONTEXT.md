# Contexto: inferencia via Roboflow

Comando para retomar: "retomar inferencia Roboflow".

Atualizado em 07/05/2026.

## Estado Atual Validado

O caminho aprovado para producao esta no repositorio `Facilita-Projeto-Coffee-Frontend`.

- A aba `Inferencia Roboflow` chama a rota `/api/v1/inferences/roboflow-direct`.
- A mesma aba tambem possui o bloco `Inferencia Roboflow por lote GPU`.
- Esse fluxo e separado da inferencia YOLO local.
- A inferencia YOLO local continua usando fila/job/worker.
- A inferencia Roboflow direta usa a Vercel como servidor intermediario e chama a API do Roboflow diretamente.
- A inferencia Roboflow por lote usa Roboflow Batch Processing API com GPU explicita.
- A rota direta nao cria job e nao aciona worker YOLO.
- O lote GPU tambem nao cria job local, nao aciona Vast e nao aciona worker YOLO.
- O resultado e persistido diretamente em `inference_runs`.
- O overlay correto e gerado como PNG a partir da mascara completa, igual ao padrao validado no localhost.
- A classe `planta` e inferida por exclusao: a mascara inicia como `planta`, e os poligonos retornados pelo Roboflow sobrescrevem `fundo` e `coffee`.
- O visualizador mostra identificacoes, percentuais e pixels de `coffee`, `planta` e `fundo`.

Commits importantes no frontend:

```text
fcfda44 Add Roboflow inference tab
5ef30d1 Separate direct Roboflow inference flow
3585c1f Log direct Roboflow inference steps
9cce137 Resize direct Roboflow images before upload
663175b Generate Roboflow overlays from exclusion mask
7465ecb Retry Roboflow direct calls with smaller images
c5866a8 Show Roboflow detection and pixel metrics
a34a704 Add Roboflow GPU batch inference
4cff3e6 Allow Roboflow batch blob uploads
6c47f55 Use minimal Roboflow batch job payload
81e4b78 Retry Roboflow GPU batch payload variants
43a3739 Import Roboflow batch results
edf7e14 Document Roboflow GPU batch flow
10c1089 Enable manual and automatic batch sync
6bd0ff5 Read Roboflow export batch ids
230c5e0 Handle multipart Roboflow batch exports
addffb9 Import compressed Roboflow batch results
48ce0c1 Extract Roboflow tar batch results
ea29bef Import Roboflow batch results incrementally
```

## Estado do Worker Original

Tambem foi implementado suporte Roboflow por provider no worker original:

- Commit `e0e0321 Add Roboflow inference provider`.
- O worker aceita inferencia `local_yolo` ou `roboflow` por job via `inference_provider`.
- A logica Roboflow foi integrada em `backend/app/services/roboflow_inference.py`.
- O worker chama o Workflow Roboflow por HTTP direto, sem `inference-sdk`, para evitar conflito de `numpy` com Ultralytics/YOLO.
- Esse caminho fica como alternativa/fallback arquitetural, mas o fluxo que ficou aprovado pelo usuario foi o Roboflow direto no frontend/Vercel.

## Objetivo

Estudar e implementar um modo em que a inferencia pesada seja feita no Roboflow, enquanto este backend continua responsavel por gerar mascaras, calcular pixels por classe e salvar artefatos no Blob/control plane.

## Arquitetura atual aprovada

Fluxo Roboflow direto:

1. Usuario envia uma imagem na aba `Inferencia Roboflow`.
2. O frontend redimensiona a imagem antes de subir ao Blob, com lado maximo inicial de 2560 px.
3. A rota `/api/v1/inferences/roboflow-direct` baixa a imagem do Blob.
4. A rota chama o Workflow API do Roboflow.
5. Se o Roboflow retornar `502`, `503` ou `504`, a rota tenta novamente e usa fallbacks menores (`1920`, `1280`, `960` por padrao).
6. A rota extrai `predictions` com poligonos.
7. A rota rasteriza os poligonos e cria a mascara local:
   - `0 = fundo`
   - `1 = coffee`
   - `2 = planta`
8. A rota calcula metricas de pixels por classe.
9. A rota gera e salva no Blob:
   - `overlay.png`
   - `color-mask.png`
   - `mask.png`
   - `dataset/labels/<runId>.txt`
   - `roboflow-result.json`
10. A rota persiste o resultado em `inference_runs`.

Fluxo Roboflow por lote GPU:

1. Usuario seleciona varias imagens ou uma pasta no bloco `Inferencia Roboflow por lote GPU`.
2. O frontend prepara/redimensiona imagens e sobe entradas para o Vercel Blob.
3. A rota `/api/v1/inferences/roboflow-batch` baixa as imagens do Blob e envia ao Roboflow Data Staging.
4. A rota cria o job de Batch Processing com `computeConfiguration.machineType = "gpu"`.
5. O Roboflow processa o lote e disponibiliza os resultados/exportacoes.
6. A UI tenta sincronizar automaticamente os resultados do lote.
7. A rota `/api/v1/inferences/roboflow-batch-sync` consulta o job, baixa JSONL/resultados, extrai `predictions`, rasteriza poligonos e recria a mascara local:
   - `0 = fundo`
   - `1 = coffee`
   - `2 = planta`
8. A mesma rota calcula metricas, gera `overlay.png`, `color-mask.png`, `mask.png`, TXT YOLO e `roboflow-result.json`.
9. Cada imagem importada do lote e persistida em `inference_runs`, aparecendo na aba `Resultados`.
10. Se o lote ainda estiver processando, a UI permite usar `Sincronizar ultimo lote`.
11. A sincronizacao importa em blocos pequenos para evitar timeout da Vercel:
   - body aceita `startIndex` e `importLimit`;
   - resposta inclui `nextIndex`, `totalRecords` e `hasMore`;
   - UI continua chamando ate `hasMore=false`.

Lote testado em producao:

- Batch ID: `rf_batch_moulldlk`
- Job ID: `rfgpumoulldlk`
- Cuidado: o ID correto tem dois `l` antes de `dk`; com um `l` o Roboflow retorna 404 `Could not find requested batch`.
- Apos importacao do lote, a API de inferencias em producao confirmou `total=102`.

Fluxo YOLO local:

1. Frontend/control plane cria job de inferencia.
2. Worker Python baixa/carrega a imagem.
3. Worker roda YOLO local.
4. Worker salva artefatos e metricas.

Manter esses dois fluxos separados para evitar conflito de logica.

## Classes locais

Padrao atual do projeto:

- `0 = fundo`
- `1 = coffee`
- `2 = planta`

A classe `planta` continua sendo inferida por exclusao quando o Roboflow retornar apenas `fundo` e `coffee`.

## Pontos de codigo relevantes

- `backend/app/services/worker_jobs.py`
  - `_process_inference(...)` hoje carrega modelo YOLO local, roda predicao e salva artefatos.
  - Ponto natural para escolher provider `local_yolo` ou `roboflow`.
- `backend/app/services/yolo_segmentation.py`
  - `predict_sample_class_mask(...)` gera a mascara local atual.
  - `build_yolo_annotation_text_from_mask(...)` pode continuar sendo usado depois da mascara final.
- `backend/app/services/annotation.py`
  - `compute_pixel_distribution(...)` calcula os pixels por classe.
  - `build_color_mask(...)` e `build_overlay(...)` geram visualizacoes.
- `backend/app/services/modeling.py`
  - `calculate_inference_payload(...)` monta o payload de metricas final.

## Variaveis relevantes

```env
ROBOFLOW_API_KEY=
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_WORKSPACE=topografia-exata-s-workspace
ROBOFLOW_WORKFLOW=segmentacao-cafe-instance-segmentation-1777720897351
ROBOFLOW_IMAGE_INPUT=image
ROBOFLOW_CLASSES=fundo,coffee
ROBOFLOW_CLASSES_PARAMETER=classes
ROBOFLOW_CONFIDENCE=0.05
ROBOFLOW_CONFIDENCE_PARAMETER=confidence
ROBOFLOW_USE_CACHE=true
ROBOFLOW_TIMEOUT_SECONDS=120
ROBOFLOW_FALLBACK_MAX_SIDES=1920,1280,960
ROBOFLOW_BATCH_API_URL=https://api.roboflow.com
ROBOFLOW_BATCH_WORKERS_PER_MACHINE=1
ROBOFLOW_BATCH_TIMEOUT_SECONDS=3600
```

Nao colar API key no chat. Configurar apenas no `.env` local/deploy.

## Conversao esperada do retorno Roboflow

Nao usar `annotated_image` para metricas. Ela e apenas visual.

Preferencias:

1. Se o workflow retornar mascara semantica (`segmentation_mask` base64 + `class_map`), decodificar e remapear para os IDs locais.
2. Se retornar `predictions` com poligonos/`points`, rasterizar com OpenCV (`cv2.fillPoly`) para criar a mascara.
3. Se retornar somente `annotated_image`, pedir ajuste no workflow, porque nao e confiavel para contagem de pixels.

## Decisoes tomadas

- A inferencia Roboflow aprovada roda pela rota direta na Vercel, nao pela fila YOLO.
- A inferencia Roboflow por lote GPU aprovada roda pela API de Batch Processing do Roboflow, tambem fora da fila YOLO.
- A planta e sempre por exclusao quando Roboflow retorna apenas `fundo` e `coffee`.
- O overlay aprovado e PNG calculado pixel a pixel com alpha, nao SVG de poligonos.
- Os logs da rota direta registram etapas, payloads sanitizados, respostas, tentativas e variantes de imagem.
- Os logs do lote usam os prefixos `[roboflow-batch]` e `[roboflow-batch-sync]`.
- Resultados antigos mantem os artefatos antigos; para ver mudancas e preciso gerar nova inferencia.

## Aprendizado do lote GPU - qualidade e confianca

- A metrica local do lote usa a mesma base conceitual do direto: rasterizacao de `predictions`, pixels por classe e `planta` por exclusao.
- A diferenca de qualidade observada veio da configuracao efetiva do Roboflow Batch Processing, nao da formula local de metrica.
- No fluxo direto, a rota envia parametros dinamicos para o Workflow API, como `classes` e `confidence`.
- No Batch Processing API, o Roboflow aceitou o payload minimo/oficial do job; na pratica o batch depende do workflow publicado no Roboflow.
- Comparativo real em producao para `DSC00003`:
  - Batch GPU `rfbatch-rfgpumoulldlk-1-0001_dsc00003`: `confidence=0.05`, `70` identificacoes, `Coffee 1.31%`.
  - Direto antigo `roboflow-1778087437334`: `confidence=0`, `297` identificacoes, `Coffee 3.59%`.
- Alterar o workflow publicado para `Custom Confidence = 0` fez um teste de 5 imagens falhar no Roboflow: 5/5 arquivos falharam no fragmento `fragmento-0`.
- Recomendacao para proximo teste do batch:
  - `Custom Confidence`: `0.01` ou `0.02`, evitar `0`.
  - `Workers por maquina`: `1`.
  - `Max Detections`: `1000`.
  - `Max Candidates`: `1000` ou `1500`.
  - `Mask Decode Mode`: `accurate`.
  - Testar primeiro 1 a 3 imagens.
- Se o Roboflow retornar `0 resultado(s) importados` apos falha do job, baixar a secao `falhas` do export para ler a mensagem real.

## Observacoes

- Sim, nesse desenho a inferencia acontece dentro do Roboflow.
- O worker nao participa do fluxo Roboflow direto.
- O worker tambem nao participa do fluxo Roboflow por lote GPU.
- O lote GPU depende do formato JSONL/exportacao do Roboflow; se mudar o formato, revisar `pages/api/v1/inferences/roboflow-batch-sync.js` no frontend.
- E plausivel o Roboflow estar gerando resultado visual melhor por diferenca de modelo, checkpoint, preprocessing, thresholds, resolucao ou pos-processamento.
- Se voltar 502 do Roboflow, consultar logs da Vercel procurando `roboflow_attempt_failed`, `roboflow_image_variants_ready` e `roboflow_attempt_succeeded`.
