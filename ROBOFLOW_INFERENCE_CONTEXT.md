# Contexto: inferencia via Roboflow

Comando para retomar: "retomar inferencia Roboflow".

## Objetivo

Estudar e implementar um modo em que a inferencia pesada seja feita no Roboflow, enquanto este backend continua responsavel por gerar mascaras, calcular pixels por classe e salvar artefatos no Blob/control plane.

## Arquitetura desejada

Fluxo proposto:

1. Frontend/control plane cria job de inferencia.
2. Worker Python baixa/carrega a imagem.
3. Worker chama o Workflow API do Roboflow usando chave no backend.
4. Roboflow executa a inferencia.
5. Worker recebe `outputs[0].predictions` ou uma mascara bruta.
6. Worker converte o resultado para `class_mask` no padrao local.
7. Worker reaproveita o pipeline atual para gerar `mask.png`, `color-mask.png`, `overlay.png`, TXT YOLO e metricas de pixels.

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

## Variaveis sugeridas

Adicionar ao `.env.example` quando implementar:

```env
INFERENCE_PROVIDER=local_yolo
ROBOFLOW_API_KEY=
ROBOFLOW_API_URL=https://detect.roboflow.com
ROBOFLOW_WORKSPACE=topografia-exata-s-workspace
ROBOFLOW_WORKFLOW=general-segmentation-api-10
ROBOFLOW_CLASSES=fundo,coffee
ROBOFLOW_TIMEOUT_SECONDS=60
```

Nao colar API key no chat. Configurar apenas no `.env` local/deploy.

## Conversao esperada do retorno Roboflow

Nao usar `annotated_image` para metricas. Ela e apenas visual.

Preferencias:

1. Se o workflow retornar mascara semantica (`segmentation_mask` base64 + `class_map`), decodificar e remapear para os IDs locais.
2. Se retornar `predictions` com poligonos/`points`, rasterizar com OpenCV (`cv2.fillPoly`) para criar a mascara.
3. Se retornar somente `annotated_image`, pedir ajuste no workflow, porque nao e confiavel para contagem de pixels.

## Decisoes sugeridas

- Implementar `INFERENCE_PROVIDER=local_yolo|roboflow`.
- Manter YOLO local como fallback.
- Criar servico novo, por exemplo `backend/app/services/roboflow_inference.py`.
- Reaproveitar todo o pos-processamento atual depois de obter `class_mask`.
- Testar primeiro com uma imagem unica e comparar:
  - Roboflow -> mascara -> metricas locais
  - YOLO local -> mascara -> metricas locais

## Observacoes

- Sim, nesse desenho a inferencia acontece dentro do Roboflow.
- O worker pode rodar sem GPU para esse modo, pois o Roboflow executa o modelo.
- E plausivel o Roboflow estar gerando resultado visual melhor por diferenca de modelo, checkpoint, preprocessing, thresholds, resolucao ou pos-processamento.
- Antes de implementar, validar o schema do workflow para confirmar exatamente quais outputs estao disponiveis.
