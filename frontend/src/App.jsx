import { useEffect, useMemo, useState } from "react";
import { getHistory, health, predictFile, uploadUrl } from "./api";

export default function App() {
  const [server, setServer] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const isVideo = useMemo(() => {
    if (!file) return false;
    return file.type.startsWith("video/");
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
        const h = await getHistory(20);
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

  return (
    <div style={{ maxWidth: 980, margin: "24px auto", padding: 16, fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ marginBottom: 6 }}>Deepfake HoaxBuster</h1>
      <p style={{ marginTop: 0, color: "#444" }}>
        Backend status: <b>{server?.status === "ok" ? "OK" : "Belum terhubung"}</b>{" "}
        {server?.model_loaded ? "(model loaded)" : ""}
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 16 }}>
          <h2 style={{ marginTop: 0 }}>1) Upload & Predict</h2>

          <form onSubmit={onSubmit}>
            <input
              type="file"
              accept="image/*,video/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <div style={{ marginTop: 12 }}>
              <button type="submit" disabled={loading} style={{ padding: "10px 12px" }}>
                {loading ? "Memproses..." : "Analisis"}
              </button>
            </div>
          </form>

          {previewUrl && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ margin: "8px 0" }}>Preview</h3>
              {!isVideo ? (
                <img src={previewUrl} alt="preview" style={{ maxWidth: "100%", borderRadius: 10 }} />
              ) : (
                <video src={previewUrl} controls style={{ width: "100%", borderRadius: 10 }} />
              )}
            </div>
          )}

          {err && (
            <p style={{ color: "crimson", marginTop: 12 }}>
              <b>Error:</b> {err}
            </p>
          )}
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 16 }}>
          <h2 style={{ marginTop: 0 }}>2) Result</h2>

          {!result ? (
            <p style={{ color: "#555" }}>Belum ada hasil. Upload file lalu klik Analisis.</p>
          ) : (
            <>
              <div style={{ padding: 12, border: "1px solid #eee", borderRadius: 10 }}>
                <p style={{ margin: 0 }}>
                  <b>Label:</b> {result.result.label}
                </p>
                <p style={{ margin: "6px 0" }}>
                  <b>Confidence:</b> {(result.result.confidence * 100).toFixed(2)}%
                </p>
                <p style={{ margin: "6px 0" }}>
                  <b>Fake prob:</b> {result.result.fake_prob.toFixed(6)}
                </p>
                <p style={{ margin: "6px 0" }}>
                  <b>Threshold:</b> {result.result.threshold}
                </p>
                <p style={{ margin: "6px 0" }}>
                  <b>Latency:</b> {result.timing.latency_ms} ms
                </p>
                <p style={{ margin: "6px 0" }}>
                  <b>Model:</b> {result.model.version} (img {result.model.img_size})
                </p>
                <p style={{ margin: "6px 0" }}>
                  <b>Created:</b> {result.created_at}
                </p>
              </div>

              <div style={{ marginTop: 12 }}>
                <h3 style={{ margin: "8px 0" }}>Saved Media (from backend)</h3>
                <p style={{ marginTop: 0, color: "#555" }}>
                  <code>{uploadUrl(result.filename)}</code>
                </p>

                {result.media_type === "image" ? (
                  <img
                    src={uploadUrl(result.filename)}
                    alt="uploaded"
                    style={{ maxWidth: "100%", borderRadius: 10, border: "1px solid #eee" }}
                  />
                ) : (
                  <video
                    src={uploadUrl(result.filename)}
                    controls
                    style={{ width: "100%", borderRadius: 10, border: "1px solid #eee" }}
                  />
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16, border: "1px solid #ddd", borderRadius: 10, padding: 16 }}>
        <h2 style={{ marginTop: 0 }}>3) History</h2>
        {history.length === 0 ? (
          <p style={{ color: "#555" }}>Belum ada data history.</p>
        ) : (
          <table width="100%" cellPadding="8" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#f6f6f6" }}>
                <th align="left">ID</th>
                <th align="left">Type</th>
                <th align="left">Filename</th>
                <th align="left">Label</th>
                <th align="left">Confidence</th>
                <th align="left">Created</th>
              </tr>
            </thead>
            <tbody>
              {history.map((it) => (
                <tr key={it.id} style={{ borderTop: "1px solid #eee" }}>
                  <td>{it.id}</td>
                  <td>{it.media_type}</td>
                  <td>{it.filename}</td>
                  <td>{it.label}</td>
                  <td>{(it.confidence * 100).toFixed(2)}%</td>
                  <td style={{ fontSize: 12, color: "#555" }}>{it.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}