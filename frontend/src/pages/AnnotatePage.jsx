import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import AnnotationBoard from "../components/AnnotationBoard";
import { deleteAnnotation, getAnnotation, getSam2Status, saveAnnotation } from "../lib/api";

const WIZARD_STEPS = [
  { id: "image", shortLabel: "Imagem" },
  { id: "fruto", shortLabel: "Fruto", classSlug: "fruto" },
  { id: "folhagem", shortLabel: "Folhas", classSlug: "folhagem" },
  { id: "preview", shortLabel: "Preview" },
];

function buildInitialStepState() {
  return {
    fruto: { dirty: false, completed: false },
    folhagem: { dirty: false, completed: false },
  };
}

function stepHasSelection(stepState, stepId) {
  const item = stepState[stepId];
  return Boolean(item?.dirty || item?.completed);
}

function stepBadge(step, stepState, currentAnnotationId, hasPendingSave, maxReachedStepIndex) {
  if (step.id === "preview") {
    if (currentAnnotationId && !hasPendingSave) return { tone: "success", label: "Salvo" };
    if (maxReachedStepIndex >= 3) return { tone: "idle", label: "Revisar" };
    return { tone: "idle", label: "" };
  }

  const item = stepState[step.classSlug];
  if (!item) return { tone: "idle", label: "" };
  if (currentAnnotationId && !hasPendingSave && item.completed) {
    return { tone: "success", label: "Salvo" };
  }
  if (item.completed) return { tone: "success", label: "Pronto" };
  if (item.dirty) return { tone: "warning", label: "Marcado" };
  return { tone: "idle", label: "Pendente" };
}

function blockedReason(index, selectedFile, maxReachedStepIndex, stepState) {
  if (index === 0) return "";
  if (!selectedFile) return "Selecione uma imagem antes de avancar no wizard.";
  if (index === 2 && !stepHasSelection(stepState, "fruto")) {
    return "Marque o Fruto antes de abrir a etapa de Folhas.";
  }
  if (index === 3 && !stepHasSelection(stepState, "fruto")) {
    return "Marque o Fruto antes de abrir o Preview.";
  }
  if (index === 3 && !stepHasSelection(stepState, "folhagem")) {
    return "Marque as Folhas antes de abrir o Preview.";
  }
  if (index > maxReachedStepIndex + 1) {
    return "Siga a ordem do wizard: Fruto, Folhas e depois Preview.";
  }
  return "";
}

function WizardStepCard({ step, index, active, disabled, badge, onClick }) {
  return (
    <button
      type="button"
      className={[
        "wizard-step",
        "wizard-step--card",
        active ? "wizard-step--active" : "",
        disabled ? "wizard-step--disabled" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onClick={onClick}
      disabled={disabled}
    >
      <span className="wizard-step__index">{index + 1}</span>
      <span className="wizard-step__body">
        <strong>{step.shortLabel}</strong>
        {badge?.label ? (
          <small className={`wizard-step__badge wizard-step__badge--${badge.tone}`}>{badge.label}</small>
        ) : null}
      </span>
    </button>
  );
}

function ImageStepCard({ active, fileLabel, inputKey, onFileChange }) {
  return (
    <label
      className={[
        "wizard-step",
        "wizard-step--card",
        "wizard-step--upload",
        active ? "wizard-step--active" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <span className="wizard-step__index">1</span>
      <span className="wizard-step__body">
        <strong>Imagem</strong>
        <small className="wizard-step__file">{fileLabel}</small>
        <span className="button button--ghost button--small wizard-step__upload-button">
          Escolher arquivo
        </span>
      </span>
      <input
        key={inputKey}
        className="wizard-step__file-input"
        type="file"
        accept="image/*"
        onChange={onFileChange}
      />
    </label>
  );
}

function ClearModal({ open, busy, onConfirm, onCancel }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="clear-modal-title">
        <p className="eyebrow">Limpar anotacao</p>
        <h2 id="clear-modal-title">Deseja limpar toda a anotacao atual?</h2>
        <p>
          Isso remove o progresso local e, se a imagem ja tiver sido salva, tambem exclui esse item
          do servidor.
        </p>
        <div className="modal-card__actions">
          <button type="button" className="button button--ghost" onClick={onCancel} disabled={busy}>
            Cancelar
          </button>
          <button type="button" className="button button--danger" onClick={onConfirm} disabled={busy}>
            {busy ? "Limpando..." : "Limpar tudo"}
          </button>
        </div>
      </div>
    </div>
  );
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Nao foi possivel carregar a mascara salva para edicao."));
    image.src = src;
  });
}

async function fetchFileFromUrl(url, filename) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Nao foi possivel carregar a imagem original para editar a anotacao.");
  }
  const blob = await response.blob();
  return new File([blob], filename || "imagem.png", {
    type: blob.type || "image/png",
  });
}

async function inspectStoredMask(maskUrl) {
  const image = await loadImageElement(maskUrl);
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;

  let hasFruto = false;
  let hasFolhagem = false;
  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index] === 2) hasFruto = true;
    if (pixels[index] === 1) hasFolhagem = true;
    if (hasFruto && hasFolhagem) break;
  }

  return { fruto: hasFruto, folhagem: hasFolhagem };
}

export default function AnnotatePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedFile, setSelectedFile] = useState(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [maxReachedStepIndex, setMaxReachedStepIndex] = useState(0);
  const [brushSize, setBrushSize] = useState(24);
  const [sam2Status, setSam2Status] = useState(null);
  const [stepState, setStepState] = useState(buildInitialStepState);
  const [status, setStatus] = useState({ kind: "idle", message: "" });
  const [isSaving, setIsSaving] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [clearModalOpen, setClearModalOpen] = useState(false);
  const [currentAnnotationId, setCurrentAnnotationId] = useState("");
  const [fileInputKey, setFileInputKey] = useState(0);
  const [hasPendingSave, setHasPendingSave] = useState(false);
  const [initialMaskUrl, setInitialMaskUrl] = useState("");

  useEffect(() => {
    loadSam2Status();
  }, []);

  const sampleToEdit = searchParams.get("sample") || "";

  useEffect(() => {
    if (!sampleToEdit) return;
    loadAnnotationForEditing(sampleToEdit);
  }, [sampleToEdit]);

  const activeStep = WIZARD_STEPS[stepIndex];
  const isPreviewStep = activeStep.id === "preview";
  const selectedClass = activeStep.classSlug || "folhagem";
  const samAllowed = activeStep.classSlug === "folhagem" && !isPreviewStep;
  const lockedClassSlugs = useMemo(() => {
    if (activeStep.id === "folhagem" || activeStep.id === "preview") {
      return ["fruto"];
    }
    return [];
  }, [activeStep.id]);

  const selectedFileLabel = useMemo(() => {
    if (!selectedFile) return "Nenhuma imagem selecionada";
    return `${selectedFile.name} · ${Math.round(selectedFile.size / 1024)} KB`;
  }, [selectedFile]);

  async function loadSam2Status() {
    try {
      const payload = await getSam2Status();
      setSam2Status(payload);
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    }
  }

  function resetWizardState() {
    setSelectedFile(null);
    setStepIndex(0);
    setMaxReachedStepIndex(0);
    setBrushSize(24);
    setStepState(buildInitialStepState());
    setCurrentAnnotationId("");
    setHasPendingSave(false);
    setInitialMaskUrl("");
    setFileInputKey((current) => current + 1);
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] ?? null;
    setSelectedFile(nextFile);
    setCurrentAnnotationId("");
    setStepState(buildInitialStepState());
    setHasPendingSave(false);
    setInitialMaskUrl("");
    setStatus({ kind: "idle", message: "" });
    setStepIndex(nextFile ? 1 : 0);
    setMaxReachedStepIndex(nextFile ? 1 : 0);
    setSearchParams({});
  }

  async function loadAnnotationForEditing(sampleId) {
    setStatus({ kind: "loading", message: "Carregando anotacao salva para edicao..." });
    try {
      const payload = await getAnnotation(sampleId);
      const item = payload.item;

      const [imageFile, occupancy] = await Promise.all([
        fetchFileFromUrl(item.image_url, item.original_filename),
        inspectStoredMask(item.mask_url),
      ]);

      const nextStepState = {
        fruto: { dirty: occupancy.fruto, completed: occupancy.fruto },
        folhagem: { dirty: occupancy.folhagem, completed: occupancy.folhagem },
      };
      const nextStepIndex = occupancy.fruto && occupancy.folhagem ? 3 : occupancy.fruto ? 2 : 1;

      setSelectedFile(imageFile);
      setCurrentAnnotationId(item.id);
      setInitialMaskUrl(item.mask_url);
      setStepState(nextStepState);
      setHasPendingSave(false);
      setStepIndex(nextStepIndex);
      setMaxReachedStepIndex(nextStepIndex);
      setStatus({
        kind: "success",
        message: "Anotacao carregada. Ajuste a mascara e salve novamente quando terminar.",
      });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    }
  }

  function markCurrentStepDirty(isDirty) {
    const stepId = activeStep.classSlug;
    if (!stepId) return;
    setStepState((current) => {
      const next = current[stepId];
      if (!next || next.dirty === isDirty) return current;
      return {
        ...current,
        [stepId]: {
          ...next,
          dirty: isDirty,
          completed: isDirty ? next.completed : false,
        },
      };
    });
  }

  function markCurrentStepEdited() {
    if (!activeStep.classSlug) return;
    setHasPendingSave(true);
    setStatus({
      kind: "idle",
      message: `${activeStep.shortLabel} atualizado localmente. O save final acontece apenas no Preview.`,
    });
  }

  async function handleExportMask(maskBlob) {
    if (!selectedFile || !isPreviewStep) return;
    setIsSaving(true);
    setStatus({
      kind: "loading",
      message: "Salvando anotacao completa da foto...",
    });
    try {
      const payload = await saveAnnotation(selectedFile, maskBlob, currentAnnotationId);
      setCurrentAnnotationId(payload.item.id);
      setHasPendingSave(false);
      setStepState({
        fruto: { dirty: true, completed: true },
        folhagem: { dirty: true, completed: true },
      });
      setStatus({
        kind: "success",
        message: "Anotacao completa salva com sucesso. A galeria recebeu apenas 1 item dessa foto.",
      });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
      throw error;
    } finally {
      setIsSaving(false);
    }
  }

  function canOpenStep(index) {
    return !blockedReason(index, selectedFile, maxReachedStepIndex, stepState);
  }

  function goToStep(index) {
    const reason = blockedReason(index, selectedFile, maxReachedStepIndex, stepState);
    if (reason) {
      setStatus({ kind: "error", message: reason });
      return;
    }

    if (index > stepIndex && activeStep.classSlug) {
      if (!stepHasSelection(stepState, activeStep.classSlug)) {
        setStatus({
          kind: "error",
          message: `Marque ${activeStep.shortLabel} antes de avancar no wizard.`,
        });
        return;
      }
      setStepState((current) => ({
        ...current,
        [activeStep.classSlug]: {
          ...current[activeStep.classSlug],
          completed: true,
        },
      }));
    }

    setStepIndex(index);
    setMaxReachedStepIndex((current) => Math.max(current, index));
    setStatus(
      index === 3
        ? { kind: "idle", message: "Revise a imagem colorida e salve a anotacao completa da foto." }
        : { kind: "idle", message: "" }
    );
  }

  async function handleClearAll() {
    setIsClearing(true);
    setStatus({ kind: "loading", message: "Limpando anotacao atual..." });
    try {
      if (currentAnnotationId) {
        await deleteAnnotation(currentAnnotationId);
      }
      resetWizardState();
      setSearchParams({});
      setStatus({ kind: "success", message: "Anotacao atual limpa. Voce pode iniciar novamente." });
      setClearModalOpen(false);
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    } finally {
      setIsClearing(false);
    }
  }

  return (
    <section className="stack">
      <div className="panel annotate-wizard-panel">
        <div className="annotate-wizard-panel__topline">
          <p className="eyebrow">1. Wizard de anotacao</p>
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={() => setClearModalOpen(true)}
            disabled={!selectedFile && !currentAnnotationId}
          >
            Limpar tudo
          </button>
        </div>

        <div className="wizard-steps wizard-steps--row">
          <ImageStepCard
            active={stepIndex === 0}
            fileLabel={selectedFileLabel}
            inputKey={fileInputKey}
            onFileChange={handleFileChange}
          />

          {WIZARD_STEPS.slice(1).map((step, offset) => {
            const index = offset + 1;
            return (
              <WizardStepCard
                key={step.id}
                step={step}
                index={index}
                active={index === stepIndex}
                disabled={!canOpenStep(index)}
                badge={stepBadge(step, stepState, currentAnnotationId, hasPendingSave, maxReachedStepIndex)}
                onClick={() => goToStep(index)}
              />
            );
          })}
        </div>

        {status.message ? (
          <div className={`status status--${status.kind} status--inline`}>
            {status.kind === "loading" ? <span className="button-spinner" aria-hidden="true" /> : null}
            <span>{status.message}</span>
          </div>
        ) : null}
      </div>

      <div className="panel">
        <AnnotationBoard
          imageFile={selectedFile}
          initialMaskUrl={initialMaskUrl}
          selectedClass={selectedClass}
          lockedClassSlugs={lockedClassSlugs}
          brushSize={brushSize}
          sam2Status={sam2Status}
          samAllowed={samAllowed}
          isPreviewMode={isPreviewStep}
          canSave={isPreviewStep}
          isSaving={isSaving}
          onBrushSizeChange={setBrushSize}
          onDirtyChange={markCurrentStepDirty}
          onStepEdit={markCurrentStepEdited}
          onExportMask={handleExportMask}
        />
      </div>

      <ClearModal
        open={clearModalOpen}
        busy={isClearing}
        onConfirm={handleClearAll}
        onCancel={() => setClearModalOpen(false)}
      />
    </section>
  );
}
