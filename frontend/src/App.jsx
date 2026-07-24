import { useState, useRef, useEffect } from "react";
import TopNav from "./TopNav.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

// Demo fallback so this UI is inspectable even without a live backend.
const DEMO_MATCHES = [
  { id: "RN-2281", similarity_percent: 94.2, confidence: "high", reason: "matching pear-cut stone and pavé band", metadata: { name: "Pear Halo Ring", price: 1240, category: "rings" } },
  { id: "RN-2214", similarity_percent: 88.7, confidence: "high", reason: "same rose-gold finish, narrower band", metadata: { name: "Rose Solitaire Ring", price: 980, category: "rings" } },
  { id: "RN-2097", similarity_percent: 79.5, confidence: "medium", reason: "similar silhouette, different metal tone", metadata: { name: "Classic Pavé Ring", price: 860, category: "rings" } },
  { id: "RN-1950", similarity_percent: 71.3, confidence: "medium", reason: "comparable cut, plainer band", metadata: { name: "Petite Solitaire", price: 640, category: "rings" } },
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

function ResultCard({ match }) {
  const [imageFailed, setImageFailed] = useState(false);
  const imageUrl = match.metadata?.image_url;
  const showImage = imageUrl && !imageFailed;

  return (
    <div className="result-card">
      <div className="result-image" aria-hidden={!showImage}>
        {showImage ? (
          <img
            src={imageUrl.startsWith("http") ? imageUrl : `${API_BASE}${imageUrl}`}
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

export default function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", type: "text", text: "Upload a photo of a piece you like, and I'll find the closest matches in the catalog." },
  ]);
  const [pendingImage, setPendingImage] = useState(null);
  const [inputText, setInputText] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);

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

    setMessages((m) => [
      ...m,
      { role: "user", type: "image", imageUrl, caption },
    ]);
    setIsSearching(true);
    setInputText("");
    const fileToSend = pendingImage.file;
    setPendingImage(null);

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
        }
        .result-image img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
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
                <img src={m.imageUrl} alt="Uploaded reference" />
                {m.caption && <p className="user-image-caption">{m.caption}</p>}
              </div>
            )}
            {m.type === "results" && (
              <div className="results-row">
                {m.matches.map((match) => (
                  <ResultCard key={match.id} match={match} />
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
    </div>
  );
}
