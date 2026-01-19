const API_BASE = "https://chairul01-deepfake-backend.hf.space";

export async function health() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("Health check gagal");
  return res.json();
}

export async function predictFile(file, maxFrames = 12) {
  const form = new FormData();
  form.append("file", file);
  form.append("max_frames", String(maxFrames));

  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Predict gagal");
  return data;
}

export async function getHistory(limit = 20) {
  const res = await fetch(`${API_BASE}/api/history?limit=${limit}`);
  if (!res.ok) throw new Error("History gagal");
  return res.json();
}

export function uploadUrl(filename) {
  return `${API_BASE}/uploads/${encodeURIComponent(filename)}`;
}