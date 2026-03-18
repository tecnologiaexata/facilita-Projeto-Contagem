import { useEffect, useState } from "react";
import ImagePreviewFigure from "../components/ImagePreviewFigure";
import { deleteInference, getInferences, getTraining, runInference, runTraining } from "../lib/api";

function PercentCard({ label, value }) {
  return (
    <article className="stat-card">
      <span>{label}</span>
      <strong>{value}%</strong>
    </article>
  );
}

function readCoffeeMetrics(item) {
  return {
    cafe: item.cafe_percentual_na_imagem ?? item.area_cafe_percentual ?? 0,
    descarte: item.descarte_percentual_na_imagem ?? item.fundo_percentual_na_imagem ?? 0,
    frutoNoCafe: item.fruto_percentual_no_cafe ?? item.fruto_percentual_na_area_cafe ?? 0,
    folhagemNoCafe: item.folhagem_percentual_no_cafe ?? item.folhagem_percentual_na_area_cafe ?? 0,
  };
}

async function fetchInferencePageData() {
  const [trainingPayload, inferencePayload] = await Promise.all([getTraining(), getInferences()]);
  return {
    training: trainingPayload,
    history: inferencePayload.items,
  };
}

function buildTrainingProgressMessage(job, prefix = "Treino em background") {
  const phase = job?.task?.phase || job?.phase;
  return phase ? `${prefix}: ${phase}.` : `${prefix} em andamento.`;
}

export default function InferencePage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [training, setTraining] = useState(null);
  const [status, setStatus] = useState({ kind: "idle", message: "" });
  const [deletingId, setDeletingId] = useState("");
  const trainingJob = training?.job;
  const isTrainingRunning = Boolean(trainingJob?.is_active);

  useEffect(() => {
    loadPage();
  }, []);

  useEffect(() => {
    if (!trainingJob?.is_active) {
      return undefined;
    }

    let cancelled = false;
    let timer = null;

    async function pollTraining() {
      try {
        const trainingPayload = await getTraining();
        if (cancelled) return;

        setTraining(trainingPayload);

        const nextJob = trainingPayload.job;
        if (nextJob?.is_active) {
          setStatus({
            kind: "loading",
            message: buildTrainingProgressMessage(nextJob),
          });
          timer = window.setTimeout(pollTraining, 3000);
          return;
        }

        if (nextJob?.status === "failed") {
          setStatus({
            kind: "error",
            message: nextJob.error || "Falha ao treinar o modelo.",
          });
          return;
        }

        const pageData = await fetchInferencePageData();
        if (cancelled) return;

        setTraining(pageData.training);
        setHistory(pageData.history);
        setStatus({
          kind: "success",
          message: "Treino concluido. Agora voce pode rodar a inferencia.",
        });
      } catch (error) {
        if (cancelled) return;
        setStatus({ kind: "error", message: error.message });
      }
    }

    timer = window.setTimeout(pollTraining, 3000);
    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [trainingJob?.id, trainingJob?.status]);

  async function loadPage() {
    try {
      const pageData = await fetchInferencePageData();
      setTraining(pageData.training);
      setHistory(pageData.history);
      if (pageData.training?.job?.is_active) {
        setStatus({
          kind: "loading",
          message: buildTrainingProgressMessage(pageData.training.job),
        });
      }
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    }
  }

  async function handleInference() {
    if (!selectedFile || !training?.has_model || isTrainingRunning) return;
    setStatus({ kind: "loading", message: "Processando imagem e calculando percentuais..." });
    try {
      const payload = await runInference(selectedFile);
      setResult(payload.item);
      await loadPage();
      setStatus({
        kind: "success",
        message: "Inferencia concluida. O resultado tambem foi adicionado na galeria de treino com XML CVAT para revisao e edicao.",
      });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    }
  }

  async function handleTrainFirst() {
    setStatus({ kind: "loading", message: "Iniciando treino em background..." });
    try {
      const payload = await runTraining();
      setTraining(payload.item);

      if (payload.item?.job?.is_active) {
        setStatus({
          kind: "loading",
          message: buildTrainingProgressMessage(
            payload.item.job,
            payload.started ? "Treino iniciado em background" : "Treino ja em andamento",
          ),
        });
        return;
      }

      await loadPage();
      setStatus({ kind: "success", message: "Treino concluido. Agora voce pode rodar a inferencia." });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    }
  }

  async function handleDeleteInference(item) {
    if (!window.confirm(`Excluir a inferencia ${item.original_filename}? A anotacao da galeria de treino sera mantida.`)) {
      return;
    }
    setDeletingId(item.id);
    setStatus({
      kind: "loading",
      message: `Excluindo ${item.original_filename} do historico de inferencia...`,
    });
    try {
      await deleteInference(item.id);
      if (result?.id === item.id) {
        setResult(null);
      }
      await loadPage();
      setStatus({
        kind: "success",
        message: "Inferencia excluida do historico. A anotacao correspondente na galeria de treino foi preservada.",
      });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    } finally {
      setDeletingId("");
    }
  }

  return (
    <section className="page-grid">
      <div className="panel panel--sticky">
        <div className="panel__header">
          <p className="eyebrow">4. Inferencia</p>
          <h2>Suba uma imagem nova e receba a composicao do cafe, descartando o fundo.</h2>
        </div>

        <label className="field">
          <span>Imagem para inferencia</span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
          />
        </label>

        <div className="callout">
          <strong>Modelo atual</strong>
          <p>
            {isTrainingRunning
              ? buildTrainingProgressMessage(trainingJob)
              : training?.has_model
              ? `Treinado em ${new Date(training.latest_report?.trained_at || Date.now()).toLocaleString("pt-BR")}`
              : "Ainda nao existe modelo salvo."}
          </p>
        </div>

        <div className={`status status--${status.kind}`}>
          {status.message
            || (isTrainingRunning
              ? buildTrainingProgressMessage(trainingJob)
              : training?.has_model
                ? "Envie uma imagem para segmentar."
                : "Treine o modelo antes de rodar a inferencia.")}
        </div>

        <div className="button-row">
          <button
            type="button"
            className="button"
            onClick={handleInference}
            disabled={!selectedFile || !training?.has_model || isTrainingRunning || status.kind === "loading"}
          >
            Rodar inferencia
          </button>
          <button
            type="button"
            className="button button--ghost"
            onClick={handleTrainFirst}
            disabled={status.kind === "loading" || isTrainingRunning || Boolean(deletingId)}
          >
            {isTrainingRunning ? "Treino em andamento..." : "Treinar agora"}
          </button>
        </div>
      </div>

      <div className="stack">
        <div className="panel">
          <div className="panel__header">
            <p className="eyebrow">Resultado visual</p>
            <h2>Fundo como descarte e cafe como fruto + folhas.</h2>
          </div>

          {!result ? (
            <div className="empty-state">
              <p>Nenhuma inferencia executada nesta sessao ainda.</p>
            </div>
          ) : (
            (() => {
              const metrics = readCoffeeMetrics(result);
              return (
                <div className="stack">
              <div className="stats-grid">
                <PercentCard label="Cafe na imagem" value={metrics.cafe} />
                <PercentCard label="Descarte na imagem" value={metrics.descarte} />
                <PercentCard label="Fruto no cafe" value={metrics.frutoNoCafe} />
                <PercentCard label="Folhas no cafe" value={metrics.folhagemNoCafe} />
              </div>

              {result.training_annotation_id ? (
                <div className="callout">
                  <strong>Tambem entrou na galeria de treino</strong>
                  <p>
                    A inferencia gerou a anotacao <code>{result.training_annotation_id}</code> com overlay e XML CVAT
                    para voce revisar, editar e incluir no proximo treino.
                  </p>
                </div>
              ) : null}

              <div className="gallery-card__images">
                <ImagePreviewFigure
                  src={result.image_url}
                  previewSrc={result.image_preview_url}
                  alt={result.original_filename}
                  caption="Original"
                  loading="eager"
                />
                <ImagePreviewFigure
                  src={result.overlay_url}
                  previewSrc={result.overlay_preview_url}
                  alt={`overlay-${result.original_filename}`}
                  caption="Overlay previsto"
                  loading="eager"
                />
              </div>
                </div>
              );
            })()
          )}
        </div>

        <div className="panel">
          <div className="panel__header">
            <p className="eyebrow">Historico</p>
            <h2>Resultados ja processados pelo backend.</h2>
          </div>

          {!history.length ? (
            <div className="empty-state">
              <p>O historico de inferencias aparecera aqui depois do primeiro processamento.</p>
            </div>
          ) : (
            <div className="gallery-grid">
              {history.map((item) => {
                const metrics = readCoffeeMetrics(item);
                return (
                  <article key={item.id} className="gallery-card">
                  <header className="gallery-card__header">
                    <div>
                      <strong>{item.original_filename}</strong>
                      <small>{new Date(item.created_at).toLocaleString("pt-BR")}</small>
                    </div>
                    <div className="gallery-card__actions">
                      <span>Modelo: {new Date(item.trained_at).toLocaleDateString("pt-BR")}</span>
                      <button
                        type="button"
                        className="button button--ghost button--danger button--small"
                        onClick={() => handleDeleteInference(item)}
                        disabled={deletingId === item.id}
                      >
                        {deletingId === item.id ? "Excluindo..." : "Excluir"}
                      </button>
                    </div>
                  </header>
                  <div className="gallery-card__images">
                    <ImagePreviewFigure
                      src={item.image_url}
                      previewSrc={item.image_preview_url}
                      alt={item.original_filename}
                      caption="Entrada"
                    />
                    <ImagePreviewFigure
                      src={item.overlay_url}
                      previewSrc={item.overlay_preview_url}
                      alt={`infer-${item.original_filename}`}
                      caption="Segmentacao"
                    />
                  </div>
                  <div className="gallery-card__stats">
                    <span>Cafe: {metrics.cafe}%</span>
                    <span>Descarte: {metrics.descarte}%</span>
                    <span>Fruto no cafe: {metrics.frutoNoCafe}%</span>
                    <span>Folhas no cafe: {metrics.folhagemNoCafe}%</span>
                  </div>
                  {item.training_annotation_id ? (
                    <div className="gallery-card__stats">
                      <span>Galeria treino: {item.training_annotation_id}</span>
                    </div>
                  ) : null}
                  </article>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
