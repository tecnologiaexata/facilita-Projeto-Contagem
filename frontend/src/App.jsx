import { startTransition, useEffect, useState } from "react";
import {
  deleteGalleryItem,
  deleteInference,
  getGallery,
  getInferences,
  getMeta,
  runInference,
  runTraining,
  uploadGalleryItem,
} from "./lib/api";

const GALLERY_PAGE_SIZE = 36;

function formatDate(value) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString("pt-BR");
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatBytes(value) {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let current = value;
  let unitIndex = 0;
  while (current >= 1024 && unitIndex < units.length - 1) {
    current /= 1024;
    unitIndex += 1;
  }
  return `${current.toFixed(current >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function buildTrainingMessage(training) {
  if (!training) return "Ainda nao existe modelo treinado.";
  if (training.job?.is_active) {
    return training.job.phase || "Treino em andamento.";
  }
  if (training.latest_report?.trained_at) {
    return `Modelo treinado em ${formatDate(training.latest_report.trained_at)}.`;
  }
  return "Ainda nao existe modelo treinado.";
}

function statusTone(kind) {
  if (kind === "error") return "error";
  if (kind === "success") return "success";
  if (kind === "loading") return "loading";
  return "neutral";
}

function StatusBanner({ state }) {
  if (!state?.message) return null;
  return <div className={`status-banner status-banner--${statusTone(state.kind)}`}>{state.message}</div>;
}

function MetricTile({ label, value, helper }) {
  return (
    <article className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{helper}</small>
    </article>
  );
}

function PreviewFigure({ src, alt, caption }) {
  if (!src) return null;
  return (
    <figure className="preview-figure">
      <div className="preview-figure__frame">
        <img src={src} alt={alt} loading="lazy" />
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

function GalleryCard({ item, busy, onDelete }) {
  const stats = item.pixel_stats?.coffee_metrics ?? {};
  const sourceLabel = item.source_type === "external_txt" ? "TXT externo" : "Mascara interna";

  return (
    <article className="record-card">
      <header className="record-card__header">
        <div>
          <h3>{item.original_filename}</h3>
          <p>{formatDate(item.created_at)}</p>
        </div>
        <div className="record-card__actions">
          <span className="pill">{sourceLabel}</span>
          {item.annotation_txt_url ? (
            <a className="ghost-button" href={item.annotation_txt_url} target="_blank" rel="noreferrer">
              TXT
            </a>
          ) : null}
          <button type="button" className="ghost-button ghost-button--danger" disabled={busy} onClick={onDelete}>
            {busy ? "Excluindo..." : "Excluir"}
          </button>
        </div>
      </header>

      <div className="record-card__media">
        <PreviewFigure
          src={item.image_preview_url || item.image_url}
          alt={`original-${item.original_filename}`}
          caption="Original"
        />
        <PreviewFigure
          src={item.overlay_preview_url || item.overlay_url}
          alt={`overlay-${item.original_filename}`}
          caption="Overlay"
        />
      </div>

      <div className="record-card__stats">
        <span>Cafe: {formatPercent(stats.cafe_percentual_na_imagem)}</span>
        <span>Planta: {formatPercent(stats.planta_percentual_na_imagem)}</span>
        <span>Area mapeada: {formatPercent(stats.area_mapeada_percentual_na_imagem)}</span>
        <span>Fundo: {formatPercent(stats.fundo_percentual_na_imagem)}</span>
      </div>

      <div className="record-card__footer">
        <span>Formato: {item.annotation_format || "mascara"}</span>
        <span>Formas: {item.annotation_shape_count ?? "-"}</span>
        <span>
          Classes: {Array.isArray(item.annotation_classes) && item.annotation_classes.length ? item.annotation_classes.join(", ") : "-"}
        </span>
      </div>
    </article>
  );
}

function InferenceCard({ item, busy, onDelete }) {
  return (
    <article className="record-card record-card--analysis">
      <header className="record-card__header">
        <div>
          <h3>{item.original_filename}</h3>
          <p>{formatDate(item.created_at)}</p>
        </div>
        <div className="record-card__actions">
          <span className="pill">Modelo {formatDate(item.trained_at)}</span>
          <button type="button" className="ghost-button ghost-button--danger" disabled={busy} onClick={onDelete}>
            {busy ? "Excluindo..." : "Excluir"}
          </button>
        </div>
      </header>

      <div className="record-card__media">
        <PreviewFigure
          src={item.image_preview_url || item.image_url}
          alt={`inferencia-${item.original_filename}`}
          caption="Imagem analisada"
        />
        <PreviewFigure
          src={item.overlay_preview_url || item.overlay_url}
          alt={`resultado-${item.original_filename}`}
          caption="Resultado"
        />
      </div>

      <div className="record-card__stats">
        <span>Cafe: {formatPercent(item.cafe_percentual_na_imagem)}</span>
        <span>Planta: {formatPercent(item.planta_percentual_na_imagem)}</span>
        <span>Area mapeada: {formatPercent(item.area_mapeada_percentual_na_imagem)}</span>
        <span>Fundo: {formatPercent(item.fundo_percentual_na_imagem)}</span>
      </div>
    </article>
  );
}

export default function App() {
  const [dashboard, setDashboard] = useState({
    isLoading: true,
    error: "",
    meta: null,
    gallery: [],
    galleryTotal: 0,
    inferences: [],
  });
  const [uploadState, setUploadState] = useState({ kind: "idle", message: "" });
  const [trainingState, setTrainingState] = useState({ kind: "idle", message: "" });
  const [inferenceState, setInferenceState] = useState({ kind: "idle", message: "" });
  const [uploadFormVersion, setUploadFormVersion] = useState(0);
  const [inferenceFormVersion, setInferenceFormVersion] = useState(0);
  const [deletingGalleryId, setDeletingGalleryId] = useState("");
  const [deletingInferenceId, setDeletingInferenceId] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard(showLoading = true) {
      if (showLoading) {
        setDashboard((current) => ({
          ...current,
          isLoading: current.meta === null,
          error: "",
        }));
      }

      try {
        const [metaPayload, galleryPayload, inferencePayload] = await Promise.all([
          getMeta(),
          getGallery({ offset: 0, limit: GALLERY_PAGE_SIZE }),
          getInferences(),
        ]);

        if (cancelled) return;

        startTransition(() => {
          setDashboard({
            isLoading: false,
            error: "",
            meta: metaPayload,
            gallery: galleryPayload.items ?? [],
            galleryTotal: galleryPayload.total ?? galleryPayload.items?.length ?? 0,
            inferences: inferencePayload.items ?? [],
          });
        });
      } catch (error) {
        if (cancelled) return;
        setDashboard((current) => ({
          ...current,
          isLoading: false,
          error: error.message || "Falha ao carregar o painel.",
        }));
      }
    }

    void loadDashboard(true);

    if (!dashboard.meta?.training?.job?.is_active) {
      return () => {
        cancelled = true;
      };
    }

    const intervalId = window.setInterval(() => {
      void loadDashboard(false);
    }, 4000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [refreshKey, dashboard.meta?.training?.job?.id, dashboard.meta?.training?.job?.status]);

  async function handleUploadSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const imageFile = formData.get("image");
    const annotationFile = formData.get("annotation_txt");

    if (!(imageFile instanceof File) || !imageFile.size || !(annotationFile instanceof File) || !annotationFile.size) {
      setUploadState({ kind: "error", message: "Selecione uma foto e um TXT de anotacao antes de enviar." });
      return;
    }

    setUploadState({ kind: "loading", message: "Convertendo o TXT e salvando o item na galeria..." });

    try {
      const payload = await uploadGalleryItem(imageFile, annotationFile);
      setUploadState({
        kind: "success",
        message: `${payload.item.original_filename} foi salvo na galeria com sucesso.`,
      });
      setUploadFormVersion((current) => current + 1);
      setRefreshKey((current) => current + 1);
    } catch (error) {
      setUploadState({ kind: "error", message: error.message || "Falha ao salvar o item." });
    }
  }

  async function handleRunTraining() {
    setTrainingState({ kind: "loading", message: "Preparando o treino do modelo..." });

    try {
      const payload = await runTraining();
      const message = payload.started
        ? "Treino iniciado. O painel sera atualizado automaticamente."
        : payload.item?.job?.phase || "Ja existe um treino em andamento.";
      setTrainingState({ kind: "success", message });
      setRefreshKey((current) => current + 1);
    } catch (error) {
      setTrainingState({ kind: "error", message: error.message || "Falha ao iniciar o treino." });
    }
  }

  async function handleInferenceSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const imageFile = formData.get("image");

    if (!(imageFile instanceof File) || !imageFile.size) {
      setInferenceState({ kind: "error", message: "Selecione uma imagem para analisar." });
      return;
    }

    setInferenceState({ kind: "loading", message: "Rodando a analise com o modelo treinado..." });

    try {
      const payload = await runInference(imageFile);
      setInferenceState({
        kind: "success",
        message: `${payload.item.original_filename} foi analisada com sucesso.`,
      });
      setInferenceFormVersion((current) => current + 1);
      setRefreshKey((current) => current + 1);
    } catch (error) {
      setInferenceState({ kind: "error", message: error.message || "Falha ao processar a imagem." });
    }
  }

  async function handleDeleteGalleryItem(item) {
    const confirmed = window.confirm(`Excluir "${item.original_filename}" da galeria?`);
    if (!confirmed) return;

    setDeletingGalleryId(item.id);
    try {
      await deleteGalleryItem(item.id);
      setRefreshKey((current) => current + 1);
    } catch (error) {
      setUploadState({ kind: "error", message: error.message || "Falha ao excluir o item da galeria." });
    } finally {
      setDeletingGalleryId("");
    }
  }

  async function handleDeleteInference(item) {
    const confirmed = window.confirm(`Excluir o historico da analise "${item.original_filename}"?`);
    if (!confirmed) return;

    setDeletingInferenceId(item.id);
    try {
      await deleteInference(item.id);
      setRefreshKey((current) => current + 1);
    } catch (error) {
      setInferenceState({ kind: "error", message: error.message || "Falha ao excluir a analise." });
    } finally {
      setDeletingInferenceId("");
    }
  }

  const training = dashboard.meta?.training;
  const summary = dashboard.meta?.summary;
  const latestReport = training?.latest_report;
  const classes = dashboard.meta?.classes ?? [];

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Facilita Coffee Counter</p>
          <h1>Galeria, treino e analise em um fluxo so.</h1>
          <p className="hero__text">
            Esta versao opera com <strong>foto + TXT externo</strong> no novo modelo semantico: <strong>0 = cafe</strong> e{" "}
            <strong>1 = planta</strong>.
            Tudo que entra no sistema vira item de galeria, pode treinar o modelo e depois servir de base para novas analises.
          </p>
        </div>

        <div className="hero__aside">
          <MetricTile
            label="Itens na galeria"
            value={dashboard.galleryTotal}
            helper={`${summary?.source_breakdown?.external_txt ?? 0} com TXT externo`}
          />
          <MetricTile
            label="Modelo"
            value={training?.has_model ? "Pronto" : "Pendente"}
            helper={buildTrainingMessage(training)}
          />
          <MetricTile
            label="Analises salvas"
            value={training?.inference_runs ?? dashboard.inferences.length}
            helper={dashboard.inferences.length ? `Ultima em ${formatDate(dashboard.inferences[0].created_at)}` : "Sem historico"}
          />
        </div>
      </section>

      {dashboard.error ? <div className="page-alert page-alert--error">{dashboard.error}</div> : null}
      {dashboard.isLoading ? <div className="page-alert">Carregando painel...</div> : null}

      <section className="workspace-grid">
        <article className="panel">
          <div className="panel__header">
            <p className="eyebrow">1. Subir para galeria</p>
            <h2>Foto + anotacao externa</h2>
            <p>
              Envie a imagem final e o TXT exportado do site terceiro. O backend interpreta o arquivo no padrao numerico
              do projeto e converte tudo para a mascara interna usada no treino.
            </p>
          </div>

          <form key={uploadFormVersion} className="form-stack" onSubmit={handleUploadSubmit}>
            <label className="field">
              <span>Foto</span>
              <input type="file" name="image" accept="image/*" />
            </label>

            <label className="field">
              <span>TXT de anotacao</span>
              <input type="file" name="annotation_txt" accept=".txt,text/plain" />
            </label>

            <div className="hint-card">
              <strong>Formatos aceitos</strong>
              <p>`0 x1 y1 x2 y2 x3 y3 ...` para poligono de cafe.</p>
              <p>`1 cx cy w h` para bounding box YOLO de planta.</p>
              <p>`# class-map: 0=cafe, 1=planta` quando o TXT vier com cabecalho explicito.</p>
            </div>

            <button type="submit" className="primary-button">
              Salvar na galeria
            </button>
          </form>

          <StatusBanner state={uploadState} />
        </article>

        <article className="panel">
          <div className="panel__header">
            <p className="eyebrow">2. Treinar modelo</p>
            <h2>Treino manual e controlado</h2>
            <p>O modelo so e atualizado quando voce dispara o treino. Nao existe mais treino implicito ao tentar analisar uma imagem.</p>
          </div>

          <div className="detail-grid">
            <div className="detail-card">
              <span>Ultimo treino</span>
              <strong>{formatDate(latestReport?.trained_at)}</strong>
              <small>{latestReport ? `${latestReport.splits.train} treino / ${latestReport.splits.val} validacao` : "Sem relatorio ainda"}</small>
            </div>
            <div className="detail-card">
              <span>Arquivo do modelo</span>
              <strong>{training?.model_filename || "-"}</strong>
              <small>{formatBytes(training?.model_size_bytes)}</small>
            </div>
          </div>

          <button
            type="button"
            className="primary-button"
            onClick={handleRunTraining}
            disabled={!dashboard.galleryTotal || Boolean(training?.job?.is_active)}
          >
            {training?.job?.is_active ? "Treinando..." : "Treinar modelo"}
          </button>

          {training?.download_url ? (
            <a className="ghost-button ghost-button--full" href={training.download_url}>
              Baixar ultimo modelo
            </a>
          ) : null}

          <StatusBanner state={trainingState.kind === "idle" ? { kind: "neutral", message: buildTrainingMessage(training) } : trainingState} />
        </article>

        <article className="panel">
          <div className="panel__header">
            <p className="eyebrow">3. Analisar novas imagens</p>
            <h2>Inferencia com historico</h2>
            <p>Envie uma nova imagem para processar com o ultimo modelo treinado. O resultado fica salvo apenas no historico de analises.</p>
          </div>

          <form key={inferenceFormVersion} className="form-stack" onSubmit={handleInferenceSubmit}>
            <label className="field">
              <span>Nova imagem</span>
              <input type="file" name="image" accept="image/*" />
            </label>

            <button type="submit" className="primary-button" disabled={!training?.has_model || Boolean(training?.job?.is_active)}>
              Analisar imagem
            </button>
          </form>

          <StatusBanner state={inferenceState.kind === "idle" ? { kind: "neutral", message: training?.has_model ? "Modelo pronto para analise." : "Treine um modelo antes de analisar novas imagens." } : inferenceState} />
        </article>
      </section>

      <section className="panel panel--wide">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Galeria</p>
            <h2>Base armazenada para treino</h2>
          </div>
          <div className="legend">
            {classes.map((item) => (
              <span key={item.id} className="legend__item">
                <i style={{ backgroundColor: `rgb(${item.color.join(",")})` }} />
                {item.label}
              </span>
            ))}
          </div>
        </div>

        {dashboard.gallery.length ? (
          <div className="record-grid">
            {dashboard.gallery.map((item) => (
              <GalleryCard
                key={item.id}
                item={item}
                busy={deletingGalleryId === item.id}
                onDelete={() => handleDeleteGalleryItem(item)}
              />
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <h3>Nenhum item salvo ainda.</h3>
            <p>Assim que uma foto com TXT for enviada, ela aparece aqui com overlay e metricas de cafe, planta e fundo.</p>
          </div>
        )}
      </section>

      <section className="panel panel--wide">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Historico</p>
            <h2>Ultimas analises executadas</h2>
          </div>
        </div>

        {dashboard.inferences.length ? (
          <div className="record-grid">
            {dashboard.inferences.map((item) => (
              <InferenceCard
                key={item.id}
                item={item}
                busy={deletingInferenceId === item.id}
                onDelete={() => handleDeleteInference(item)}
              />
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <h3>Sem analises recentes.</h3>
            <p>Depois do primeiro treino, use o bloco de inferencia para gerar o historico.</p>
          </div>
        )}
      </section>
    </main>
  );
}
