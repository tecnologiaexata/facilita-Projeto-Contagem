# Contexto: inferencia via Roboflow

Comando para retomar: "retomar inferencia Roboflow".

Atualizado em 05/05/2026.

## Estado Atual Validado

O caminho aprovado para producao esta no repositorio `Facilita-Projeto-Coffee-Frontend`.

- A aba `Inferencia Roboflow` chama a rota `/api/v1/inferences/roboflow-direct`.
- Esse fluxo e separado da inferencia YOLO local.
- A inferencia YOLO local continua usando fila/job/worker.
- A inferencia Roboflow direta usa a Vercel como servidor intermediario e chama a API do Roboflow diretamente.
- A rota direta nao cria job e nao aciona worker YOLO.
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
- A planta e sempre por exclusao quando Roboflow retorna apenas `fundo` e `coffee`.
- O overlay aprovado e PNG calculado pixel a pixel com alpha, nao SVG de poligonos.
- Os logs da rota direta registram etapas, payloads sanitizados, respostas, tentativas e variantes de imagem.
- Resultados antigos mantem os artefatos antigos; para ver mudancas e preciso gerar nova inferencia.

## Observacoes

- Sim, nesse desenho a inferencia acontece dentro do Roboflow.
- O worker nao participa do fluxo Roboflow direto.
- E plausivel o Roboflow estar gerando resultado visual melhor por diferenca de modelo, checkpoint, preprocessing, thresholds, resolucao ou pos-processamento.
- Se voltar 502 do Roboflow, consultar logs da Vercel procurando `roboflow_attempt_failed`, `roboflow_image_variants_ready` e `roboflow_attempt_succeeded`.
