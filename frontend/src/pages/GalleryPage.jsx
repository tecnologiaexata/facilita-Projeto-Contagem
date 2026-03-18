import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import ImagePreviewFigure from "../components/ImagePreviewFigure";
import { deleteAnnotation, getAnnotations, getMeta, getTraining, runTraining } from "../lib/api";

const GALLERY_PAGE_SIZE = 12;

function MetricBlock({ label, value }) {
  return (
    <article className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function readCoffeeMetrics(stats) {
  const counts = stats?.counts || {};
  const totalPixels = stats?.total_pixels || 0;
  const coffeeMetrics = stats?.coffee_metrics;
  const cafePixels = coffeeMetrics?.cafe_pixels ?? (counts.folhagem || 0) + (counts.fruto || 0);
  const descartePixels = coffeeMetrics?.descarte_pixels ?? (counts.fundo || 0);

  return {
    cafe: coffeeMetrics?.cafe_percentual_na_imagem ?? (totalPixels ? Number(((cafePixels / totalPixels) * 100).toFixed(2)) : 0),
    descarte:
      coffeeMetrics?.descarte_percentual_na_imagem ??
      (totalPixels ? Number(((descartePixels / totalPixels) * 100).toFixed(2)) : 0),
    frutoNoCafe:
      coffeeMetrics?.fruto_percentual_no_cafe ??
      (cafePixels ? Number((((counts.fruto || 0) / cafePixels) * 100).toFixed(2)) : 0),
    folhagemNoCafe:
      coffeeMetrics?.folhagem_percentual_no_cafe ??
      (cafePixels ? Number((((counts.folhagem || 0) / cafePixels) * 100).toFixed(2)) : 0),
  };
}

function formatBytes(bytes) {
  if (!bytes) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

async function fetchGalleryPageData(pageIndex = 0) {
  const offset = pageIndex * GALLERY_PAGE_SIZE;
  const [annotationPayload, metaPayload] = await Promise.all([
    getAnnotations({ offset, limit: GALLERY_PAGE_SIZE }),
    getMeta(),
  ]);
  return {
    annotations: annotationPayload.items,
    totalAnnotations: annotationPayload.total ?? annotationPayload.items.length,
    meta: metaPayload,
  };
}

function buildTrainingProgressMessage(job, prefix = "Treino em background") {
  const phase = job?.task?.phase || job?.phase;
  return phase ? `${prefix}: ${phase}.` : `${prefix} em andamento.`;
}

export default function GalleryPage() {
  const [annotations, setAnnotations] = useState([]);
  const [meta, setMeta] = useState(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [totalAnnotations, setTotalAnnotations] = useState(0);
  const [trainingState, setTrainingState] = useState({ kind: "idle", message: "" });
  const [deletingId, setDeletingId] = useState("");
  const trainingJob = meta?.training?.job;
  const isTrainingRunning = Boolean(trainingJob?.is_active);

  useEffect(() => {
    loadPage(pageIndex);
  }, [pageIndex]);

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

        setMeta((current) => (current ? { ...current, training: trainingPayload } : { training: trainingPayload }));

        const nextJob = trainingPayload.job;
        if (nextJob?.is_active) {
          setTrainingState({
            kind: "loading",
            message: buildTrainingProgressMessage(nextJob),
          });
          timer = window.setTimeout(pollTraining, 3000);
          return;
        }

        if (nextJob?.status === "failed") {
          setTrainingState({
            kind: "error",
            message: nextJob.error || "Falha ao treinar o modelo.",
          });
          return;
        }

        const pageData = await fetchGalleryPageData(pageIndex);
        if (cancelled) return;

        if (pageData.totalAnnotations > 0 && pageIndex > 0 && !pageData.annotations.length) {
          setPageIndex(Math.max(Math.ceil(pageData.totalAnnotations / GALLERY_PAGE_SIZE) - 1, 0));
          return;
        }

        setAnnotations(pageData.annotations);
        setTotalAnnotations(pageData.totalAnnotations);
        setMeta(pageData.meta);
        setTrainingState({
          kind: "success",
          message: "Treino concluido. O modelo ja esta disponivel para inferencia.",
        });
      } catch (error) {
        if (cancelled) return;
        setTrainingState({ kind: "error", message: error.message });
      }
    }

    timer = window.setTimeout(pollTraining, 3000);
    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [trainingJob?.id, trainingJob?.status, pageIndex]);

  async function loadPage(nextPageIndex = pageIndex) {
    try {
      const pageData = await fetchGalleryPageData(nextPageIndex);
      if (pageData.totalAnnotations > 0 && nextPageIndex > 0 && !pageData.annotations.length) {
        setPageIndex(Math.max(Math.ceil(pageData.totalAnnotations / GALLERY_PAGE_SIZE) - 1, 0));
        return;
      }

      setAnnotations(pageData.annotations);
      setTotalAnnotations(pageData.totalAnnotations);
      setMeta(pageData.meta);
      if (pageData.meta?.training?.job?.is_active) {
        setTrainingState({
          kind: "loading",
          message: buildTrainingProgressMessage(pageData.meta.training.job),
        });
      }
    } catch (error) {
      setTrainingState({ kind: "error", message: error.message });
    }
  }

  async function handleTrain() {
    setTrainingState({ kind: "loading", message: "Iniciando treino em background..." });
    try {
      const payload = await runTraining();
      const trainingPayload = payload.item;
      setMeta((current) => (current ? { ...current, training: trainingPayload } : { training: trainingPayload }));

      if (trainingPayload?.job?.is_active) {
        setTrainingState({
          kind: "loading",
          message: buildTrainingProgressMessage(
            trainingPayload.job,
            payload.started ? "Treino iniciado em background" : "Treino ja em andamento",
          ),
        });
        return;
      }

      await loadPage(pageIndex);
      setTrainingState({
        kind: "success",
        message: "Treino concluido. O modelo ja esta disponivel para inferencia.",
      });
    } catch (error) {
      setTrainingState({ kind: "error", message: error.message });
    }
  }

  async function handleDelete(item) {
    if (!window.confirm(`Excluir a anotacao ${item.original_filename}?`)) return;
    setDeletingId(item.id);
    setTrainingState({
      kind: "loading",
      message: `Excluindo ${item.original_filename} da galeria de treino...`,
    });
    try {
      await deleteAnnotation(item.id);
      const nextPageIndex = annotations.length === 1 && pageIndex > 0 ? pageIndex - 1 : pageIndex;
      if (nextPageIndex !== pageIndex) {
        setPageIndex(nextPageIndex);
      } else {
        await loadPage(nextPageIndex);
      }
      setTrainingState({
        kind: "success",
        message: "Anotacao excluida. Se o modelo ja estava treinado, rode um novo treino para refletir a base atual.",
      });
    } catch (error) {
      setTrainingState({ kind: "error", message: error.message });
    } finally {
      setDeletingId("");
    }
  }

  function handleDownloadModel() {
    const downloadUrl = meta?.training?.download_url;
    if (!downloadUrl) return;
    const anchor = document.createElement("a");
    anchor.href = downloadUrl;
    anchor.download = meta?.training?.model_filename || "facilita-coffee-model.joblib";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  const latestReport = meta?.training?.latest_report;
  const summaryMetrics = readCoffeeMetrics(meta?.summary);
  const totalPages = Math.max(1, Math.ceil(totalAnnotations / GALLERY_PAGE_SIZE));
  const pageStart = totalAnnotations ? pageIndex * GALLERY_PAGE_SIZE + 1 : 0;
  const pageEnd = totalAnnotations
    ? Math.min(totalAnnotations, pageIndex * GALLERY_PAGE_SIZE + annotations.length)
    : 0;

  return (
    <section className="stack">
      <div className="page-grid">
        <div className="panel panel--sticky">
          <div className="panel__header">
            <p className="eyebrow">2. Dataset + Treino</p>
            <h2>Visualize o material anotado e gere o modelo leve do MVP.</h2>
          </div>

          <div className="stats-grid">
            <MetricBlock label="Anotacoes" value={meta?.summary?.total_annotations ?? 0} />
            <MetricBlock label="Cafe na base" value={`${summaryMetrics.cafe}%`} />
            <MetricBlock label="Fruto no cafe" value={`${summaryMetrics.frutoNoCafe}%`} />
            <MetricBlock label="Folhas no cafe" value={`${summaryMetrics.folhagemNoCafe}%`} />
          </div>

          <div className="callout">
            <strong>Split automatico</strong>
            <p>
              O backend organiza a base em treino, validacao e teste dentro de
              `storage/dataset/` cada vez que o treino roda.
            </p>
          </div>

          <div className="button-row">
            <button
              type="button"
              className="button"
              onClick={handleTrain}
              disabled={!totalAnnotations || isTrainingRunning || Boolean(deletingId)}
            >
              {isTrainingRunning ? "Treino em andamento..." : "Treinar modelo"}
            </button>
            <button
              type="button"
              className="button button--ghost"
              onClick={handleDownloadModel}
              disabled={!meta?.training?.download_url || isTrainingRunning || Boolean(deletingId)}
            >
              Baixar modelo
            </button>
          </div>
          <div className={`status status--${trainingState.kind}`}>
            {trainingState.message || (isTrainingRunning ? buildTrainingProgressMessage(trainingJob) : "Modelo ainda nao treinado.")}
          </div>

          {latestReport && (
            <div className="result-card">
              <h3>Ultimo treino</h3>
              <p>{new Date(latestReport.trained_at).toLocaleString("pt-BR")}</p>
              <p>Treino: {latestReport.splits.train} imagens</p>
              <p>Validacao: {latestReport.splits.val} imagens</p>
              <p>Teste: {latestReport.splits.test} imagens</p>
              <p>mIoU validacao: {latestReport.val_metrics?.mean_iou ?? "-"}</p>
              <p>mIoU teste: {latestReport.test_metrics?.mean_iou ?? "-"}</p>
              <p>Arquivo: {meta?.training?.model_filename ?? "-"}</p>
              <p>Tamanho: {formatBytes(meta?.training?.model_size_bytes)}</p>
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel__header">
            <p className="eyebrow">3. Galeria anotada</p>
            <h2>Todas as imagens, mascaras e overlays salvos no fluxo.</h2>
          </div>

          {totalAnnotations ? (
            <>
              <p className="eyebrow">
                Mostrando {pageStart}-{pageEnd} de {totalAnnotations} anotacoes
              </p>
              {totalPages > 1 ? (
                <div className="button-row">
                  <button
                    type="button"
                    className="button button--ghost"
                    onClick={() => setPageIndex((current) => Math.max(current - 1, 0))}
                    disabled={pageIndex === 0 || Boolean(deletingId)}
                  >
                    Anterior
                  </button>
                  <button
                    type="button"
                    className="button button--ghost"
                    onClick={() => setPageIndex((current) => Math.min(current + 1, totalPages - 1))}
                    disabled={pageIndex >= totalPages - 1 || Boolean(deletingId)}
                  >
                    Proxima
                  </button>
                </div>
              ) : null}
            </>
          ) : null}

          {!totalAnnotations ? (
            <div className="empty-state">
              <p>Nenhuma anotacao salva ainda. Comece pela aba Anotar + CVAT.</p>
            </div>
          ) : (
            <div className="gallery-grid">
              {annotations.map((item) => {
                const metrics = readCoffeeMetrics(item.pixel_stats);
                return (
                  <article key={item.id} className="gallery-card">
                  <header className="gallery-card__header">
                    <div>
                      <strong>{item.original_filename}</strong>
                      <small>{new Date(item.updated_at || item.created_at).toLocaleString("pt-BR")}</small>
                    </div>
                    <div className="gallery-card__actions">
                      <Link
                        className="button button--ghost button--small"
                        to={`/anotar?sample=${item.id}`}
                      >
                        Editar
                      </Link>
                      <a
                        className="button button--ghost button--small"
                        href={item.image_url}
                        download={`${item.file_label || item.id}.png`}
                      >
                        Imagem
                      </a>
                      <a
                        className="button button--ghost button--small"
                        href={item.cvat_url}
                        download={`${item.file_label || item.id}.xml`}
                      >
                        XML CVAT
                      </a>
                      <a
                        className="button button--ghost button--small"
                        href={item.package_url}
                        download
                      >
                        ZIP
                      </a>
                      <button
                        type="button"
                        className="button button--ghost button--danger button--small"
                        onClick={() => handleDelete(item)}
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
                      caption="Original"
                    />
                    <ImagePreviewFigure
                      src={item.overlay_url}
                      previewSrc={item.overlay_preview_url}
                      alt={`overlay-${item.original_filename}`}
                      caption="Overlay"
                    />
                  </div>
                  <div className="gallery-card__stats">
                    <span>Cafe: {metrics.cafe}%</span>
                    <span>Descarte: {metrics.descarte}%</span>
                    <span>Fruto no cafe: {metrics.frutoNoCafe}%</span>
                    <span>Folhas no cafe: {metrics.folhagemNoCafe}%</span>
                  </div>
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
