import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { getHistory, health, predictFile, uploadUrl } from "./api";

function formatPct(x) {
  return `${(x * 100).toFixed(2)}%`;
}

function safeText(x) {
  return x == null ? "" : String(x);
}

export default function App() {
  const [server, setServer] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const [isDragging, setIsDragging] = useState(false);

  const [q, setQ] = useState("");
  const [filterLabel, setFilterLabel] = useState("ALL");

  const isVideo = useMemo(() => {
    if (!file) return false;
    return file.type?.startsWith("video/");
  }, [file]);

  useEffect(() => {
    (async () => {
      try {
        const h = await health();
        setServer(h);
      } catch (e) {
        setErr(String(e.message || e));
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const h = await getHistory(30);
        setHistory(h.items || []);
      } catch {
        // ignore
      }
    })();
  }, [result]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const filteredHistory = useMemo(() => {
    let items = [...history];

    if (filterLabel !== "ALL") {
      items = items.filter((it) => safeText(it.label).toUpperCase() === filterLabel);
    }

    if (q.trim()) {
      const s = q.trim().toLowerCase();
      items = items.filter((it) => safeText(it.filename).toLowerCase().includes(s));
    }

    return items;
  }, [history, q, filterLabel]);

  function onPickFile(f) {
    setErr("");
    setResult(null);
    setFile(f || null);
  }

  function onDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onPickFile(f);
  }

  async function onSubmit(e) {
    e.preventDefault();
    setErr("");
    setResult(null);

    if (!file) {
      setErr("Pilih file dulu (gambar / video).");
      return;
    }

    try {
      setLoading(true);
      const data = await predictFile(file, 12);
      setResult(data);
    } catch (e2) {
      setErr(String(e2.message || e2));
    } finally {
      setLoading(false);
    }
  }

  const statusText = server?.status === "ok" ? "OK" : "Belum terhubung";
  const modelLoaded = Boolean(server?.model_loaded);

  const label = result?.result?.label ? String(result.result.label).toUpperCase() : "";
  const fakeProb = result?.result?.fake_prob ?? null;
  const threshold = result?.result?.threshold ?? null;

  const badgeClass =
    label === "REAL" ? "badge badgeReal" : label === "FAKE" ? "badge badgeFake" : "badge";

  const progressWidth = fakeProb == null ? "0%" : `${Math.max(0, Math.min(1, fakeProb)) * 100}%`;

  return (
    <div className="container">
      <div className="topbar">
        <div className="brand">
          <h1>Deepfake HoaxBuster</h1>
          <p>
            Deteksi Real vs Fake (deepfake) berbasis MobileNetV2. Jalankan inferensi cepat dari gambar/video dan simpan riwayat hasil.
          </p>
        </div>

        <div className="chips">
          <div className="chip">
            Backend: <b>{statusText}</b>
          </div>
          <div className="chip">
            Model: <b>{modelLoaded ? "loaded" : "not loaded"}</b>
          </div>
          {result?.model?.img_size != null && (
            <div className="chip">
              Input: <b>{result.model.img_size}</b>
            </div>
          )}
          {result?.timing?.latency_ms != null && (
            <div className="chip">
              Latency: <b>{result.timing.latency_ms} ms</b>
            </div>
          )}
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h2>1) Upload & Predict</h2>
          <p className="muted">
            Unggah gambar/video wajah. Untuk hasil lebih stabil, gunakan wajah yang terlihat jelas dan tidak blur.
          </p>

          <div
            className={`dropzone ${isDragging ? "dropzoneActive" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            <div className="row">
              <div className="fileName">{file ? file.name : "Drag & drop file di sini atau pilih file."}</div>
              <label className="btn">
                Pilih File
                <input
                  type="file"
                  accept="image/*,video/*"
                  style={{ display: "none" }}
                  onChange={(e) => onPickFile(e.target.files?.[0] || null)}
                />
              </label>
            </div>

            <div className="row">
              <button className="btn btnPrimary" onClick={onSubmit} disabled={loading}>
                {loading ? "Memproses..." : "Analisis"}
              </button>

              <div className="muted">
                Maks frame video: <b>12</b>
              </div>
            </div>
          </div>

          {previewUrl && (
            <div className="preview">
              {!isVideo ? (
                <img src={previewUrl} alt="preview" />
              ) : (
                <video src={previewUrl} controls />
              )}
            </div>
          )}

          {err && (
            <div className="error">
              <b>Error:</b> {err}
            </div>
          )}
        </div>

        <div className="card">
          <h2>2) Result</h2>

          {!result ? (
            <p className="muted">Belum ada hasil. Upload file lalu klik Analisis.</p>
          ) : (
            <>
              <div className={badgeClass}>
                Label: <span>{label}</span>
              </div>

              <div className="progressWrap">
                <div className="muted">
                  Fake probability: <b>{fakeProb == null ? "-" : fakeProb.toFixed(6)}</b>
                </div>
                <div className="progressBar" aria-label="fake probability bar">
                  <div className="progressFill" style={{ width: progressWidth }} />
                </div>
              </div>

              <div className="kv">
                <div className="k">Confidence</div>
                <div>{formatPct(result.result.confidence)}</div>

                <div className="k">Threshold</div>
                <div>{threshold}</div>

                <div className="k">Model</div>
                <div>
                  {result.model.version} (img {result.model.img_size})
                </div>

                <div className="k">Created</div>
                <div className="small">{result.created_at}</div>
              </div>

              <div style={{ marginTop: 12 }}>
                <div className="muted" style={{ marginBottom: 8 }}>
                  Saved media (backend):
                </div>
                <div className="small" style={{ wordBreak: "break-all" }}>
                  {uploadUrl(result.filename)}
                </div>

                <div className="preview" style={{ marginTop: 10 }}>
                  {result.media_type === "image" ? (
                    <img src={uploadUrl(result.filename)} alt="uploaded" />
                  ) : (
                    <video src={uploadUrl(result.filename)} controls />
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h2>3) History</h2>

        <div className="tools">
          <input
            className="input"
            placeholder="Search filename..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />

          <select className="select" value={filterLabel} onChange={(e) => setFilterLabel(e.target.value)}>
            <option value="ALL">All</option>
            <option value="REAL">REAL</option>
            <option value="FAKE">FAKE</option>
          </select>
        </div>

        {filteredHistory.length === 0 ? (
          <p className="muted">Belum ada data history atau tidak ada yang cocok dengan filter.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Preview</th>
                <th>Filename</th>
                <th>Label</th>
                <th>Confidence</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {filteredHistory.map((it) => {
                const itLabel = safeText(it.label).toUpperCase();
                const itBadge =
                  itLabel === "REAL" ? "badge badgeReal" : itLabel === "FAKE" ? "badge badgeFake" : "badge";

                return (
                  <tr key={it.id}>
                    <td className="small">{it.id}</td>
                    <td>
                      <div className="thumb">
                        {it.media_type === "image" ? (
                          <img src={uploadUrl(it.filename)} alt="thumb" />
                        ) : (
                          <img
                            src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='96' height='68'%3E%3Crect width='100%25' height='100%25' fill='%23000000'/%3E%3Cpath d='M38 22 L60 34 L38 46 Z' fill='%23ffffff' opacity='0.7'/%3E%3C/svg%3E"
                            alt="video"
                          />
                        )}
                      </div>
                    </td>
                    <td style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {it.filename}
                    </td>
                    <td>
                      <span className={itBadge}>{itLabel}</span>
                    </td>
                    <td>{formatPct(it.confidence)}</td>
                    <td className="small">{it.created_at}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}