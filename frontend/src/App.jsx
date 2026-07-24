import { useState, useRef, useEffect } from "react";
import TopNav from "./TopNav.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

// Archives the user's query image to object storage (S3) via a backend-issued
// pre-signed URL, so the browser never holds AWS credentials directly.
async function uploadQueryImage(file) {
  const presignRes = await fetch(`${API_BASE}/api/v1/uploads/presign`, {
    method: "POST",
    headers: { "x-api-key": API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, content_type: file.type }),
  });
  if (!presignRes.ok) throw new Error("presign request failed");
  const { upload_url } = await presignRes.json();

  const putRes = await fetch(upload_url, {
    method: "PUT",
    headers: { "Content-Type": file.type },
    body: file,
  });
  if (!putRes.ok) throw new Error("upload to storage failed");
}

// Generates a simple ring illustration entirely locally (no network fetch) so
// the demo fallback below has something real to show and click into.
function ringDataUri(bandColor, stoneColor) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
      <rect width="200" height="200" fill="#f4efe6"/>
      <circle cx="100" cy="120" r="55" fill="none" stroke="${bandColor}" stroke-width="14"/>
      <polygon points="100,45 120,75 100,100 80,75" fill="${stoneColor}" stroke="#fff" stroke-width="2"/>
      <polygon points="100,45 120,75 100,90" fill="rgba(255,255,255,0.35)"/>
    </svg>
  `.trim();
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

// Demo fallback so this UI is inspectable even without a live backend.
const DEMO_MATCHES = [
  { id: "RN-2281", similarity_percent: 94.2, confidence: "high", reason: "matching pear-cut stone and pavé band", metadata: { name: "Pear Halo Ring", price: 1240, category: "rings", image_url: ringDataUri("#caa86a", "#e8e0d0") } },
  { id: "RN-2214", similarity_percent: 88.7, confidence: "high", reason: "same rose-gold finish, narrower band", metadata: { name: "Rose Solitaire Ring", price: 980, category: "rings", image_url: ringDataUri("#c48a72", "#f2d9d9") } },
  { id: "RN-2097", similarity_percent: 79.5, confidence: "medium", reason: "similar silhouette, different metal tone", metadata: { name: "Classic Pavé Ring", price: 860, category: "rings", image_url: ringDataUri("#9a9a9a", "#e6f0f5") } },
  { id: "RN-1950", similarity_percent: 71.3, confidence: "medium", reason: "comparable cut, plainer band", metadata: { name: "Petite Solitaire", price: 640, category: "rings", image_url: ringDataUri("#d4b98c", "#fff9ec") } },
];

function PlaceholderIcon() {
  return (
    <svg viewBox="0 0 64 64" width="34" height="34">
      <path
        d="M32 4 L54 22 L32 60 L10 22 Z M10 22 L54 22 M20 22 L32 4 L44 22 M32 22 L26 60 M32 22 L38 60"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
      />
    </svg>
  );
}

function CloudIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M6.5 18a4 4 0 0 1-.5-7.97A5.5 5.5 0 0 1 17 8.06 4.5 4.5 0 0 1 16.5 17H6.5Z" strokeLinejoin="round" />
    </svg>
  );
}

function ResultCard({ match, onImageClick }) {
  const [imageFailed, setImageFailed] = useState(false);
  const imageUrl = match.metadata?.image_url;
  const showImage = imageUrl && !imageFailed;
  const fullUrl = showImage
    ? (imageUrl.startsWith("http") || imageUrl.startsWith("data:") ? imageUrl : `${API_BASE}${imageUrl}`)
    : null;

  return (
    <div className="result-card">
      <div
        className="result-image"
        aria-hidden={!showImage}
        role={showImage ? "button" : undefined}
        tabIndex={showImage ? 0 : undefined}
        onClick={() => showImage && onImageClick(fullUrl, match.metadata?.name || match.id)}
        onKeyDown={(e) => {
          if (showImage && (e.key === "Enter" || e.key === " ")) onImageClick(fullUrl, match.metadata?.name || match.id);
        }}
      >
        {showImage ? (
          <img
            src={fullUrl}
            alt={match.metadata?.name || match.id}
            onError={() => setImageFailed(true)}
          />
        ) : (
          <PlaceholderIcon />
        )}
      </div>
      <div className="result-body">
        <p className="result-name">{match.metadata?.name || match.id}</p>
        <p className="result-price">${match.metadata?.price?.toLocaleString?.() ?? "—"}</p>
        <div className="result-meter-row">
          <div className="result-meter-track">
            <div
              className="result-meter-fill"
              style={{ width: `${match.similarity_percent}%` }}
            />
          </div>
          <span className="result-percent">{match.similarity_percent}%</span>
        </div>
        <p className={`result-confidence confidence-${match.confidence}`}>
          {match.confidence} confidence — {match.reason}
        </p>
      </div>
    </div>
  );
}

export function ShimmerLine() {
  return <div className="shimmer-line" aria-label="Searching" role="status" />;
}

function Lightbox({ image, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!image) return null;

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <button className="lightbox-close" onClick={onClose} aria-label="Close">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M5 5l14 14M19 5L5 19" />
        </svg>
      </button>
      <figure className="lightbox-figure" onClick={(e) => e.stopPropagation()}>
        <img src={image.url} alt={image.label || "Jewel"} />
        {image.label && <figcaption>{image.label}</figcaption>}
      </figure>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", type: "text", text: "Upload a photo of a piece you like, and I'll find the closest matches in the catalog." },
  ]);
  const [pendingImage, setPendingImage] = useState(null);
  const [inputText, setInputText] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [lightboxImage, setLightboxImage] = useState(null);
  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);
  const nextMsgId = useRef(1);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isSearching]);

  function handleFileChosen(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setPendingImage({ file, url });
  }

  async function runSearch() {
    if (!pendingImage) return;
    const imageUrl = pendingImage.url;
    const caption = inputText.trim();
    const msgId = nextMsgId.current++;

    setMessages((m) => [
      ...m,
      { id: msgId, role: "user", type: "image", imageUrl, caption, archiveStatus: "saving" },
    ]);
    setIsSearching(true);
    setInputText("");
    const fileToSend = pendingImage.file;
    setPendingImage(null);

    uploadQueryImage(fileToSend)
      .then(() => {
        setMessages((m) => m.map((msg) => (msg.id === msgId ? { ...msg, archiveStatus: "saved" } : msg)));
      })
      .catch(() => {
        setMessages((m) => m.map((msg) => (msg.id === msgId ? { ...msg, archiveStatus: "error" } : msg)));
      });

    try {
      const form = new FormData();
      form.append("image", fileToSend);
      const res = await fetch(`${API_BASE}/api/v1/search`, {
        method: "POST",
        headers: { "x-api-key": API_KEY },
        body: form,
      });
      if (!res.ok) throw new Error("search failed");
      const data = await res.json();

      setMessages((m) => [
        ...m,
        data.no_match
          ? { role: "assistant", type: "text", text: "Nothing in the catalog is a close enough match yet — try a different angle or a wider shot." }
          : { role: "assistant", type: "results", matches: data.matches },
      ]);
    } catch {
      // Demo fallback — lets this interface be reviewed without a running backend.
      setMessages((m) => [...m, { role: "assistant", type: "results", matches: DEMO_MATCHES }]);
    } finally {
      setIsSearching(false);
    }
  }

  function handleSend() {
    if (pendingImage) {
      runSearch();
    } else if (inputText.trim()) {
      const text = inputText.trim();
      setMessages((m) => [
        ...m,
        { role: "user", type: "text", text },
        { role: "assistant", type: "text", text: "Attach a photo of the piece and I'll search the catalog for visual matches." },
      ]);
      setInputText("");
    }
  }

  return (
    <div className="app-shell">
      <style>{`
        .chat-scroll {
          width: 100%;
          max-width: 760px;
          flex: 1;
          overflow-y: auto;
          padding: 20px 20px 12px;
          display: flex;
          flex-direction: column;
          gap: 18px;
        }

        .msg-row { display: flex; }
        .msg-row.user { justify-content: flex-end; }
        .msg-row.assistant { justify-content: flex-start; }

        .bubble-text {
          max-width: 78%;
          font-size: 14.5px;
          line-height: 1.55;
          padding: 11px 15px;
        }
        .msg-row.user .bubble-text {
          background: var(--surface);
          border: 1px solid var(--hairline);
          border-radius: 4px;
          color: var(--ink);
        }
        .msg-row.assistant .bubble-text {
          color: var(--ink-soft);
          padding-left: 0;
        }

        .user-image-block {
          max-width: 78%;
        }
        .user-image-block img {
          max-width: 220px;
          border-radius: 4px;
          border: 1px solid var(--hairline);
          display: block;
        }
        .user-image-caption {
          font-size: 13px;
          color: var(--ink-soft);
          margin-top: 6px;
          text-align: right;
        }

        .archive-badge {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 5px;
          margin: 6px 0 0;
          font-size: 11px;
          color: var(--ink-soft);
        }
        .archive-badge.saving { opacity: 0.6; }
        .archive-badge.saving svg { animation: archive-pulse 1.1s ease-in-out infinite; }
        .archive-badge.saved { color: var(--gold-1); }
        .archive-badge.error { color: #B3492F; }
        @keyframes archive-pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }

        .results-row {
          display: flex;
          gap: 12px;
          overflow-x: auto;
          padding-bottom: 6px;
          width: 100%;
        }

        .result-card {
          flex: 0 0 168px;
          border: 1px solid var(--hairline);
          border-radius: 6px;
          background: var(--surface);
          overflow: hidden;
        }
        .result-image {
          height: 110px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg);
          color: var(--gold-1);
          border-bottom: 1px solid var(--hairline);
          cursor: pointer;
        }
        .result-image img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .clickable-image { cursor: pointer; }

        .lightbox-overlay {
          position: fixed;
          inset: 0;
          background: rgba(10, 8, 6, 0.82);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 40px 24px;
          animation: lightbox-fade 0.15s ease;
        }
        @keyframes lightbox-fade {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .lightbox-figure {
          margin: 0;
          max-width: min(90vw, 720px);
          max-height: 88vh;
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        .lightbox-figure img {
          max-width: 100%;
          max-height: 80vh;
          object-fit: contain;
          border-radius: 6px;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
          background: var(--surface);
        }
        .lightbox-figure figcaption {
          margin-top: 12px;
          color: #f2ece2;
          font-family: 'Cormorant Garamond', serif;
          font-size: 16px;
          text-align: center;
        }
        .lightbox-close {
          position: fixed;
          top: 22px;
          right: 26px;
          width: 38px;
          height: 38px;
          border-radius: 50%;
          border: 1px solid rgba(255, 255, 255, 0.35);
          background: rgba(255, 255, 255, 0.08);
          color: #f2ece2;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
        }
        .lightbox-close:hover { background: rgba(255, 255, 255, 0.18); }
        .result-body { padding: 10px 11px 12px; }
        .result-name {
          font-family: 'Cormorant Garamond', serif;
          font-size: 15px;
          font-weight: 600;
          margin: 0 0 2px;
        }
        .result-price {
          font-size: 12.5px;
          color: var(--ink-soft);
          margin: 0 0 8px;
        }
        .result-meter-row {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 6px;
        }
        .result-meter-track {
          flex: 1;
          height: 3px;
          background: var(--hairline);
          border-radius: 2px;
          overflow: hidden;
        }
        .result-meter-fill {
          height: 100%;
          background: var(--gradient);
        }
        .result-percent {
          font-size: 11px;
          font-variant-numeric: tabular-nums;
          color: var(--ink-soft);
        }
        .result-confidence {
          font-size: 11px;
          line-height: 1.4;
          margin: 0;
          color: var(--ink-soft);
        }
        .confidence-high { color: var(--gold-1); }

        .composer {
          width: 100%;
          max-width: 760px;
          padding: 12px 20px 22px;
          border-top: 1px solid var(--hairline);
        }
        .pending-preview {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
          font-size: 12.5px;
          color: var(--ink-soft);
        }
        .pending-preview img {
          width: 42px;
          height: 42px;
          object-fit: cover;
          border-radius: 4px;
          border: 1px solid var(--hairline);
        }
        .pending-clear {
          border: none;
          background: none;
          color: var(--ink-soft);
          cursor: pointer;
          text-decoration: underline;
          font-size: 12px;
        }

        .composer-row {
          display: flex;
          align-items: center;
          gap: 8px;
          border: 1px solid var(--hairline);
          border-radius: 999px;
          padding: 6px 6px 6px 16px;
          background: var(--surface);
          transition: box-shadow 0.15s ease;
        }
        .composer-row.dragging {
          box-shadow: 0 0 0 1px var(--gold-2);
        }

        .attach-btn, .send-btn {
          width: 34px;
          height: 34px;
          border-radius: 50%;
          border: 1px solid var(--hairline);
          background: var(--surface);
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: var(--ink);
          flex-shrink: 0;
        }
        .send-btn {
          background: var(--ink);
          color: var(--surface);
          border: none;
        }
        .send-btn:hover {
          background-image: var(--gradient);
        }
        .composer-input {
          flex: 1;
          border: none;
          outline: none;
          font-family: 'Inter', sans-serif;
          font-size: 14px;
          background: transparent;
          color: var(--ink);
        }
        .composer-input::placeholder { color: var(--ink-soft); }

        .chat-scroll::-webkit-scrollbar,
        .results-row::-webkit-scrollbar { height: 5px; width: 5px; }
        .chat-scroll::-webkit-scrollbar-thumb,
        .results-row::-webkit-scrollbar-thumb { background: var(--hairline); border-radius: 4px; }
      `}</style>

      <TopNav />
      <header className="header">
        <span className="wordmark">Facet</span>
        <div className="facet-line" />
      </header>

      <main
        className="chat-scroll"
        ref={scrollRef}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFileChosen(e.dataTransfer.files?.[0]);
        }}
      >
        {messages.map((m, i) => (
          <div key={i} className={`msg-row ${m.role}`}>
            {m.type === "text" && <div className="bubble-text">{m.text}</div>}
            {m.type === "image" && (
              <div className="user-image-block">
                <img
                  src={m.imageUrl}
                  alt="Uploaded reference"
                  className="clickable-image"
                  onClick={() => setLightboxImage({ url: m.imageUrl, label: "Your upload" })}
                />
                {m.caption && <p className="user-image-caption">{m.caption}</p>}
                {m.archiveStatus && (
                  <p className={`archive-badge ${m.archiveStatus}`}>
                    <CloudIcon />
                    {m.archiveStatus === "saving"
                      ? "Saving to storage…"
                      : m.archiveStatus === "error"
                      ? "Couldn't save to storage"
                      : "Saved to storage"}
                  </p>
                )}
              </div>
            )}
            {m.type === "results" && (
              <div className="results-row">
                {m.matches.map((match) => (
                  <ResultCard
                    key={match.id}
                    match={match}
                    onImageClick={(url, label) => setLightboxImage({ url, label })}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
        {isSearching && (
          <div className="msg-row assistant">
            <ShimmerLine />
          </div>
        )}
      </main>

      <div className="composer">
        {pendingImage && (
          <div className="pending-preview">
            <img src={pendingImage.url} alt="Selected reference" />
            <span>Ready to search this image</span>
            <button className="pending-clear" onClick={() => setPendingImage(null)}>remove</button>
          </div>
        )}
        <div className={`composer-row ${isDragging ? "dragging" : ""}`}>
          <button className="attach-btn" onClick={() => fileInputRef.current?.click()} aria-label="Attach an image">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M21 12.5V18a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3h6" />
              <path d="M12 12l7-7M19 5v5M19 5h-5" />
            </svg>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFileChosen(e.target.files?.[0])}
          />
          <input
            className="composer-input"
            placeholder={pendingImage ? "Add a note (optional)" : "Describe what you're looking for, or attach a photo"}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button className="send-btn" onClick={handleSend} aria-label="Send">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M4 12h16M13 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>

      <Lightbox image={lightboxImage} onClose={() => setLightboxImage(null)} />
    </div>
  );
}
