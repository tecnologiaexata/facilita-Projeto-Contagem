async function readJson(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return { detail: await response.text() };
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await readJson(response);
  if (!response.ok) {
    const message = payload?.detail || payload?.message || "Falha na requisicao.";
    throw new Error(message);
  }
  return payload;
}

function withQuery(path, params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    query.set(key, String(value));
  });
  const serialized = query.toString();
  return serialized ? `${path}?${serialized}` : path;
}

export function getMeta() {
  return request("/api/meta");
}

export function getMonitoring() {
  return request("/api/monitoring");
}

export function getSam2Status() {
  return request("/api/sam2/status");
}

export function getAnnotations(params = {}) {
  return request(withQuery("/api/annotations", params));
}

export function getAnnotation(sampleId) {
  return request(`/api/annotations/${sampleId}`);
}

export function saveAnnotation(originalImage, maskBlob, sampleId = "") {
  const formData = new FormData();
  formData.append("original_image", originalImage);
  formData.append("mask_image", maskBlob, "mask.png");
  if (sampleId) {
    formData.append("sample_id", sampleId);
  }
  return request("/api/annotations", {
    method: "POST",
    body: formData,
  });
}

export function deleteAnnotation(sampleId) {
  return request(`/api/annotations/${sampleId}`, {
    method: "DELETE",
  });
}

export function runTraining() {
  return request("/api/training/run", { method: "POST" });
}

export function getTraining() {
  return request("/api/training");
}

export function runInference(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);
  return request("/api/inference", {
    method: "POST",
    body: formData,
  });
}

export function getInferences() {
  return request("/api/inferences");
}

export function deleteInference(runId) {
  return request(`/api/inferences/${runId}`, {
    method: "DELETE",
  });
}

export function createSam2Session(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);
  return request("/api/sam2/sessions", {
    method: "POST",
    body: formData,
  });
}

export function predictSam2Mask(sessionId, payload) {
  return request(`/api/sam2/sessions/${sessionId}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}
