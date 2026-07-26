import { useEffect, useRef, useState } from "react";
import TopNav from "../TopNav.jsx";
import ShimmerLine from "../components/ShimmerLine.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

const RECENT_LIMIT = 10;
const CATEGORY_OPTIONS = ["ring", "necklace", "earrings", "bracelet"];

// All state below lives only in React state for this session, on purpose —
// no localStorage/sessionStorage. A page refresh mid-upload loses any
// unsubmitted rows; acceptable v1 tradeoff for an internal admin tool.

function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function makeItemId(name) {
  const base = slugify(name || "") || "item";
  const suffix = Math.random().toString(36).slice(2, 8);
  return `${base}-${suffix}`;
}

function validateRow(row) {
  const errors = {};
  if (!row.name.trim()) errors.name = "Required";
  if (!row.category.trim()) errors.category = "Required";
  const priceNum = parseFloat(row.price);
  if (row.price === "" || Number.isNaN(priceNum) || priceNum <= 0) errors.price = "Must be > 0";
  return errors;
}

function resolveImageUrl(url) {
  if (!url) return null;
  return url.startsWith("http") ? url : `${API_BASE}${url}`;
}

function CatalogRow({ row, onChange, onRemove }) {
  return (
    <div className="row-card">
      <img className="row-thumb" src={row.previewUrl} alt={row.name || "preview"} />
      <div className="row-fields">
        <div className="row-fields-grid">
          <div className="field">
            <label>Name *</label>
            <input value={row.name} onChange={(e) => onChange(row.localId, "name", e.target.value)} />
            {row.errors.name && <span className="field-error">{row.errors.name}</span>}
          </div>
          <div className="field">
            <label>Category *</label>
            <input
              list="category-options"
              value={row.category}
              onChange={(e) => onChange(row.localId, "category", e.target.value)}
            />
            {row.errors.category && <span className="field-error">{row.errors.category}</span>}
          </div>
          <div className="field">
            <label>Price *</label>
            <input
              type="number"
              min="0"
              step="0.01"
              value={row.price}
              onChange={(e) => onChange(row.localId, "price", e.target.value)}
            />
            {row.errors.price && <span className="field-error">{row.errors.price}</span>}
          </div>
          <div className="field">
            <label>Material</label>
            <input value={row.material} onChange={(e) => onChange(row.localId, "material", e.target.value)} />
          </div>
        </div>
        <div className="field">
          <label>Caption</label>
          <input value={row.caption} onChange={(e) => onChange(row.localId, "caption", e.target.value)} />
        </div>
        <div className="field">
          <label>Description</label>
          <textarea
            rows={2}
            value={row.description}
            onChange={(e) => onChange(row.localId, "description", e.target.value)}
          />
        </div>
        <div className="field">
          <label>Tags (comma-separated)</label>
          <input value={row.tagsInput} onChange={(e) => onChange(row.localId, "tagsInput", e.target.value)} />
        </div>
      </div>
      <button type="button" className="row-remove" onClick={() => onRemove(row.localId)} aria-label="Remove row">
        ×
      </button>
    </div>
  );
}

function RecentThumb({ item }) {
  const [failed, setFailed] = useState(false);
  if (!item.image_url || failed) {
    return <div className="recent-thumb recent-thumb-empty" />;
  }
  return (
    <img
      className="recent-thumb"
      src={resolveImageUrl(item.image_url)}
      alt={item.name}
      onError={() => setFailed(true)}
    />
  );
}

function fieldsToForm(item) {
  return {
    name: item.name || "",
    category: item.category || "",
    price: item.price != null ? String(item.price) : "",
    material: item.material || "",
    caption: item.caption || "",
    description: item.description || "",
    tagsInput: Array.isArray(item.tags) ? item.tags.join(", ") : "",
  };
}

function EditItemModal({ itemId, fields, imagePreviewUrl, saving, error, onChange, onImageChosen, onCancel, onSave }) {
  const errors = validateRow(fields);
  return (
    <div className="edit-overlay" onClick={onCancel}>
      <div className="edit-modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="edit-title">Edit {itemId}</h3>
        <div className="row-card edit-row-card">
          <div className="edit-image-col">
            <img className="row-thumb" src={imagePreviewUrl} alt={fields.name || itemId} />
            <label className="dropzone-browse edit-replace-image">
              Replace photo
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  if (e.target.files?.[0]) onImageChosen(e.target.files[0]);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
          <div className="row-fields">
            <div className="row-fields-grid">
              <div className="field">
                <label>Name *</label>
                <input value={fields.name} onChange={(e) => onChange("name", e.target.value)} />
                {errors.name && <span className="field-error">{errors.name}</span>}
              </div>
              <div className="field">
                <label>Category *</label>
                <input list="category-options" value={fields.category} onChange={(e) => onChange("category", e.target.value)} />
                {errors.category && <span className="field-error">{errors.category}</span>}
              </div>
              <div className="field">
                <label>Price *</label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={fields.price}
                  onChange={(e) => onChange("price", e.target.value)}
                />
                {errors.price && <span className="field-error">{errors.price}</span>}
              </div>
              <div className="field">
                <label>Material</label>
                <input value={fields.material} onChange={(e) => onChange("material", e.target.value)} />
              </div>
            </div>
            <div className="field">
              <label>Caption</label>
              <input value={fields.caption} onChange={(e) => onChange("caption", e.target.value)} />
            </div>
            <div className="field">
              <label>Description</label>
              <textarea rows={2} value={fields.description} onChange={(e) => onChange("description", e.target.value)} />
            </div>
            <div className="field">
              <label>Tags (comma-separated)</label>
              <input value={fields.tagsInput} onChange={(e) => onChange("tagsInput", e.target.value)} />
            </div>
          </div>
        </div>
        {error && <p className="inline-error">{error}</p>}
        <div className="edit-actions">
          <button type="button" className="bulk-apply-btn" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
          <button
            type="button"
            className="submit-btn"
            onClick={() => {
              if (Object.keys(errors).length > 0) return;
              onSave();
            }}
            disabled={saving}
          >
            {saving ? "Saving…" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function CatalogPage() {
  const [rows, setRows] = useState([]);
  const [bulkCategory, setBulkCategory] = useState("");
  const [bulkTags, setBulkTags] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const [submitState, setSubmitState] = useState("idle"); // idle | submitting | polling | done | error
  const [submitError, setSubmitError] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);

  const [recentItems, setRecentItems] = useState([]);
  const [recentTotal, setRecentTotal] = useState(0);
  const [recentOffset, setRecentOffset] = useState(0);
  const [recentLoading, setRecentLoading] = useState(false);
  const [recentError, setRecentError] = useState(null);
  const [recentCollapsed, setRecentCollapsed] = useState(false);

  const [editingItem, setEditingItem] = useState(null); // {itemId, imageUrl, fields}
  const [editImageFile, setEditImageFile] = useState(null);
  const [editImagePreview, setEditImagePreview] = useState(null);
  const [editSaving, setEditSaving] = useState(false);
  const [editError, setEditError] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [rowError, setRowError] = useState(null);

  const fileInputRef = useRef(null);
  const pollTimeoutRef = useRef(null);

  useEffect(() => {
    loadRecentItems(recentOffset);
  }, [recentOffset]);

  useEffect(() => () => {
    if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
  }, []);

  async function loadRecentItems(offset) {
    setRecentLoading(true);
    setRecentError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/catalog/items?limit=${RECENT_LIMIT}&offset=${offset}`, {
        headers: { "x-api-key": API_KEY },
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();
      setRecentItems(data.items);
      setRecentTotal(data.total);
    } catch (err) {
      // No demo fallback here on purpose — this is an admin tool, not the
      // chat search page. Silently showing fake catalog data would be
      // actively misleading to someone managing real inventory.
      setRecentError(
        err instanceof TypeError
          ? "Couldn't reach the backend — is it running?"
          : `Couldn't load recently added items: ${err.message}`
      );
    } finally {
      setRecentLoading(false);
    }
  }

  function handleFilesChosen(fileList) {
    const files = Array.from(fileList || []).filter((f) => f.type.startsWith("image/"));
    if (files.length === 0) return;
    const newRows = files.map((file) => ({
      localId: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      file,
      previewUrl: URL.createObjectURL(file),
      name: "",
      caption: "",
      description: "",
      tagsInput: "",
      category: "",
      price: "",
      material: "",
      errors: {},
    }));
    setRows((r) => [...r, ...newRows]);
  }

  function updateRow(localId, field, value) {
    setRows((rows) =>
      rows.map((r) =>
        r.localId === localId ? { ...r, [field]: value, errors: { ...r.errors, [field]: undefined } } : r
      )
    );
  }

  function removeRow(localId) {
    setRows((rows) => {
      const target = rows.find((r) => r.localId === localId);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return rows.filter((r) => r.localId !== localId);
    });
  }

  function applyBulk() {
    if (!bulkCategory.trim() && !bulkTags.trim()) return;
    setRows((rows) =>
      rows.map((r) => ({
        ...r,
        category: bulkCategory.trim() ? bulkCategory.trim() : r.category,
        tagsInput: bulkTags.trim() ? bulkTags.trim() : r.tagsInput,
        errors: bulkCategory.trim() ? { ...r.errors, category: undefined } : r.errors,
      }))
    );
  }

  async function handleSubmit() {
    setSubmitError(null);

    const validated = rows.map((r) => ({ ...r, errors: validateRow(r) }));
    setRows(validated);

    const validRows = validated.filter((r) => Object.keys(r.errors).length === 0);
    if (validRows.length === 0) {
      setSubmitError("Fix the highlighted rows before submitting — no valid rows to add yet.");
      return;
    }

    setSubmitState("submitting");

    const items = validRows.map((r) => ({
      item_id: makeItemId(r.name),
      name: r.name.trim(),
      category: r.category.trim(),
      price: parseFloat(r.price),
      caption: r.caption.trim(),
      description: r.description.trim(),
      tags: r.tagsInput.split(",").map((t) => t.trim()).filter(Boolean),
      ...(r.material.trim() ? { material: r.material.trim() } : {}),
    }));

    const form = new FormData();
    validRows.forEach((r) => form.append("images", r.file));
    form.append("items_json", JSON.stringify(items));

    try {
      const res = await fetch(`${API_BASE}/api/v1/catalog/index`, {
        method: "POST",
        headers: { "x-api-key": API_KEY },
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body?.detail;
        throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `Request failed (${res.status})`);
      }
      const data = await res.json();

      const validIds = new Set(validRows.map((r) => r.localId));
      validRows.forEach((r) => URL.revokeObjectURL(r.previewUrl));
      setRows((rows) => rows.filter((r) => !validIds.has(r.localId)));

      setSubmitState("polling");
      pollJob(data.job_id);
    } catch (err) {
      setSubmitState("error");
      setSubmitError(
        err instanceof TypeError
          ? "Couldn't reach the backend — is it running?"
          : `Upload failed: ${err.message}`
      );
    }
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(`${API_BASE}/api/v1/catalog/jobs/${jobId}`, {
        headers: { "x-api-key": API_KEY },
      });
      if (!res.ok) throw new Error(`Job status request failed (${res.status})`);
      const data = await res.json();
      setJobStatus(data);

      if (data.status === "pending") {
        pollTimeoutRef.current = setTimeout(() => pollJob(jobId), 2000);
      } else {
        setSubmitState("done");
        setRecentOffset(0);
        loadRecentItems(0);
      }
    } catch {
      setSubmitState("error");
      setSubmitError("Lost connection while checking indexing progress.");
    }
  }

  async function openEdit(item) {
    setRowError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/catalog/items/${encodeURIComponent(item.item_id)}`, {
        headers: { "x-api-key": API_KEY },
      });
      if (!res.ok) throw new Error(`Couldn't load item details (${res.status})`);
      const full = await res.json();
      setEditingItem({ itemId: full.item_id, imageUrl: full.image_url, fields: fieldsToForm(full) });
      setEditImageFile(null);
      setEditImagePreview(resolveImageUrl(full.image_url));
      setEditError(null);
    } catch (err) {
      setRowError(err.message);
    }
  }

  function closeEdit() {
    if (editImageFile) URL.revokeObjectURL(editImagePreview);
    setEditingItem(null);
    setEditImageFile(null);
    setEditImagePreview(null);
    setEditError(null);
  }

  function updateEditField(field, value) {
    setEditingItem((cur) => ({ ...cur, fields: { ...cur.fields, [field]: value } }));
  }

  function chooseEditImage(file) {
    setEditImageFile(file);
    setEditImagePreview(URL.createObjectURL(file));
  }

  async function saveEdit() {
    if (!editingItem) return;
    setEditSaving(true);
    setEditError(null);

    const { fields } = editingItem;
    const payload = {
      name: fields.name.trim(),
      category: fields.category.trim(),
      price: parseFloat(fields.price),
      caption: fields.caption.trim(),
      description: fields.description.trim(),
      tags: fields.tagsInput.split(",").map((t) => t.trim()).filter(Boolean),
      ...(fields.material.trim() ? { material: fields.material.trim() } : {}),
    };

    const form = new FormData();
    form.append("fields", JSON.stringify(payload));
    if (editImageFile) form.append("image", editImageFile);

    try {
      const res = await fetch(`${API_BASE}/api/v1/catalog/items/${encodeURIComponent(editingItem.itemId)}`, {
        method: "PATCH",
        headers: { "x-api-key": API_KEY },
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body?.detail;
        throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `Request failed (${res.status})`);
      }
      closeEdit();
      loadRecentItems(recentOffset);
    } catch (err) {
      setEditError(err.message);
    } finally {
      setEditSaving(false);
    }
  }

  async function handleDelete(item) {
    if (!window.confirm(`Delete "${item.name || item.item_id}" from the catalog? This can't be undone.`)) return;
    setDeletingId(item.item_id);
    setRowError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/catalog/items/${encodeURIComponent(item.item_id)}`, {
        method: "DELETE",
        headers: { "x-api-key": API_KEY },
      });
      if (!res.ok && res.status !== 204) throw new Error(`Delete failed (${res.status})`);

      // Step back a page if we just deleted the last item on the current one.
      const nextOffset = recentItems.length === 1 && recentOffset > 0 ? recentOffset - RECENT_LIMIT : recentOffset;
      if (nextOffset !== recentOffset) setRecentOffset(nextOffset);
      else loadRecentItems(recentOffset);
    } catch (err) {
      setRowError(err.message);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="app-shell">
      <style>{`
        .catalog-main {
          width: 100%;
          max-width: 1080px;
          padding: 20px 20px 60px;
          display: flex;
          flex-direction: column;
          gap: 22px;
        }

        .page-subtitle {
          font-size: 13px;
          color: var(--ink-soft);
          margin: 8px 0 0;
        }

        .dropzone {
          border: 1px dashed var(--hairline);
          border-radius: 6px;
          background: var(--surface);
          padding: 34px 20px;
          text-align: center;
          transition: box-shadow 0.15s ease, border-color 0.15s ease;
        }
        .dropzone.dragging {
          border-color: var(--gold-2);
          box-shadow: 0 0 0 1px var(--gold-2);
        }
        .dropzone-text {
          margin: 0;
          font-size: 14px;
          color: var(--ink-soft);
        }
        .dropzone-browse {
          border: none;
          background: none;
          color: var(--gold-1);
          font: inherit;
          cursor: pointer;
          text-decoration: underline;
          padding: 0;
        }

        .bulk-apply-row {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
          font-size: 12.5px;
          color: var(--ink-soft);
        }
        .bulk-apply-row input {
          border: 1px solid var(--hairline);
          border-radius: 4px;
          padding: 7px 10px;
          font: inherit;
          font-size: 13px;
          background: var(--surface);
          color: var(--ink);
          min-width: 160px;
        }
        .bulk-apply-btn {
          border: 1px solid var(--hairline);
          border-radius: 4px;
          background: var(--surface);
          padding: 7px 14px;
          font-size: 12.5px;
          cursor: pointer;
          color: var(--ink);
        }
        .bulk-apply-btn:hover { border-color: var(--gold-2); }

        .rows-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .row-card {
          display: flex;
          gap: 14px;
          border: 1px solid var(--hairline);
          border-radius: 6px;
          background: var(--surface);
          padding: 12px;
          position: relative;
        }
        .row-thumb {
          width: 72px;
          height: 72px;
          object-fit: cover;
          border-radius: 4px;
          border: 1px solid var(--hairline);
          flex-shrink: 0;
        }
        .row-fields {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 8px;
          min-width: 0;
        }
        .row-fields-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 8px;
        }
        .field {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }
        .field label {
          font-size: 11px;
          color: var(--ink-soft);
        }
        .field input, .field textarea {
          border: 1px solid var(--hairline);
          border-radius: 4px;
          padding: 6px 9px;
          font: inherit;
          font-size: 13px;
          background: var(--bg);
          color: var(--ink);
          resize: vertical;
        }
        .field input:focus, .field textarea:focus {
          outline: none;
          border-color: var(--gold-2);
        }
        .field-error {
          font-size: 11px;
          color: #B3492F;
        }
        .row-remove {
          position: absolute;
          top: 8px;
          right: 10px;
          border: none;
          background: none;
          color: var(--ink-soft);
          font-size: 17px;
          line-height: 1;
          cursor: pointer;
        }
        .row-remove:hover { color: var(--ink); }

        .submit-bar {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 10px;
        }
        .submit-btn {
          border: none;
          border-radius: 4px;
          background: var(--ink);
          color: var(--surface);
          padding: 10px 22px;
          font: inherit;
          font-size: 13.5px;
          cursor: pointer;
        }
        .submit-btn:hover { background-image: var(--gradient); }
        .submit-progress {
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 13px;
          color: var(--ink-soft);
        }
        .inline-error {
          font-size: 13px;
          color: #B3492F;
          margin: 0;
        }

        .submit-summary {
          border: 1px solid var(--hairline);
          border-radius: 6px;
          background: var(--surface);
          padding: 14px 16px;
        }
        .summary-success {
          font-family: 'Cormorant Garamond', serif;
          font-size: 17px;
          font-weight: 600;
          margin: 0 0 6px;
        }
        .summary-failed-list {
          margin: 6px 0 10px;
          padding-left: 18px;
          font-size: 12.5px;
          color: var(--ink-soft);
        }

        .recent-section {
          border-top: 1px solid var(--hairline);
          padding-top: 16px;
        }
        .collapsible-toggle {
          border: none;
          background: none;
          font-family: 'Cormorant Garamond', serif;
          font-size: 17px;
          font-weight: 600;
          color: var(--ink);
          cursor: pointer;
          padding: 0;
        }
        .recent-body { margin-top: 12px; }
        .recent-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }
        .recent-table th {
          text-align: left;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--ink-soft);
          border-bottom: 1px solid var(--hairline);
          padding: 6px 8px;
        }
        .recent-table td {
          padding: 7px 8px;
          border-bottom: 1px solid var(--hairline);
        }
        .recent-thumb {
          width: 36px;
          height: 36px;
          object-fit: cover;
          border-radius: 4px;
          border: 1px solid var(--hairline);
          display: block;
        }
        .recent-thumb-empty {
          background: var(--bg);
        }
        .recent-empty {
          text-align: center;
          color: var(--ink-soft);
          padding: 16px 8px;
        }
        .recent-pagination {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 10px;
          font-size: 12.5px;
          color: var(--ink-soft);
        }
        .recent-pagination button {
          border: 1px solid var(--hairline);
          border-radius: 4px;
          background: var(--surface);
          padding: 5px 12px;
          cursor: pointer;
          color: var(--ink);
        }
        .recent-pagination button:disabled {
          opacity: 0.4;
          cursor: default;
        }

        .recent-actions {
          text-align: right;
          white-space: nowrap;
        }
        .row-action-btn {
          border: 1px solid var(--hairline);
          border-radius: 4px;
          background: var(--surface);
          padding: 4px 10px;
          font-size: 12px;
          cursor: pointer;
          color: var(--ink);
          margin-left: 6px;
        }
        .row-action-btn:hover { border-color: var(--gold-2); }
        .row-action-btn:disabled { opacity: 0.5; cursor: default; }
        .row-action-danger { color: #B3492F; }
        .row-action-danger:hover { border-color: #B3492F; }

        .edit-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.45);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          z-index: 100;
        }
        .edit-modal {
          background: var(--surface);
          border-radius: 8px;
          padding: 20px;
          max-width: 720px;
          width: 100%;
          max-height: 90vh;
          overflow-y: auto;
        }
        .edit-title {
          font-family: 'Cormorant Garamond', serif;
          font-size: 19px;
          font-weight: 600;
          margin: 0 0 14px;
          color: var(--ink);
        }
        .edit-row-card { border: none; padding: 0; }
        .edit-image-col {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          flex-shrink: 0;
        }
        .edit-image-col .row-thumb { width: 96px; height: 96px; }
        .edit-replace-image {
          font-size: 11px;
          text-align: center;
        }
        .edit-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
          margin-top: 16px;
        }
      `}</style>

      <TopNav />
      <header className="header">
        <span className="wordmark">Facet</span>
        <div className="facet-line" />
        <p className="page-subtitle">Bulk-add catalog items with photos and details.</p>
      </header>

      <main className="catalog-main">
        <datalist id="category-options">
          {CATEGORY_OPTIONS.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>

        <section
          className={`dropzone ${isDragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFilesChosen(e.dataTransfer.files);
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              handleFilesChosen(e.target.files);
              e.target.value = "";
            }}
          />
          <p className="dropzone-text">
            Drag and drop jewellery photos here, or{" "}
            <button type="button" className="dropzone-browse" onClick={() => fileInputRef.current?.click()}>
              browse files
            </button>
          </p>
        </section>

        {rows.length > 0 && (
          <>
            <div className="bulk-apply-row">
              <span>Apply to all rows:</span>
              <input
                list="category-options"
                placeholder="Category"
                value={bulkCategory}
                onChange={(e) => setBulkCategory(e.target.value)}
              />
              <input
                placeholder="Tags (comma-separated)"
                value={bulkTags}
                onChange={(e) => setBulkTags(e.target.value)}
              />
              <button type="button" className="bulk-apply-btn" onClick={applyBulk}>
                Apply
              </button>
            </div>

            <div className="rows-list">
              {rows.map((row) => (
                <CatalogRow key={row.localId} row={row} onChange={updateRow} onRemove={removeRow} />
              ))}
            </div>

            <div className="submit-bar">
              {submitError && <p className="inline-error">{submitError}</p>}
              {submitState === "submitting" || submitState === "polling" ? (
                <div className="submit-progress">
                  <ShimmerLine />
                  <span>
                    {submitState === "submitting"
                      ? "Uploading..."
                      : `Indexing ${jobStatus?.processed ?? 0} / ${jobStatus?.total ?? rows.length}...`}
                  </span>
                </div>
              ) : (
                <button type="button" className="submit-btn" onClick={handleSubmit}>
                  Add to catalog
                </button>
              )}
            </div>
          </>
        )}

        {submitState === "done" && jobStatus && (
          <div className="submit-summary">
            <p className="summary-success">
              {jobStatus.total - jobStatus.failed_items.length} of {jobStatus.total} item(s) added to the
              catalog.
            </p>
            {jobStatus.failed_items.length > 0 && (
              <ul className="summary-failed-list">
                {jobStatus.failed_items.map((f) => (
                  <li key={f.item_id}>
                    {f.item_id}: {f.error}
                  </li>
                ))}
              </ul>
            )}
            <button
              type="button"
              className="dropzone-browse"
              onClick={() => {
                setSubmitState("idle");
                setJobStatus(null);
              }}
            >
              Dismiss
            </button>
          </div>
        )}

        <section className="recent-section">
          <button type="button" className="collapsible-toggle" onClick={() => setRecentCollapsed((c) => !c)}>
            {recentCollapsed ? "▸" : "▾"} Recently added ({recentTotal})
          </button>
          {!recentCollapsed && (
            <div className="recent-body">
              {recentError && <p className="inline-error">{recentError}</p>}
              {rowError && <p className="inline-error">{rowError}</p>}
              {recentLoading && <ShimmerLine />}
              {!recentLoading && !recentError && (
                <>
                  <table className="recent-table">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Name</th>
                        <th>Category</th>
                        <th>Price</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentItems.map((item) => (
                        <tr key={item.item_id}>
                          <td>
                            <RecentThumb item={item} />
                          </td>
                          <td>{item.name || item.item_id}</td>
                          <td>{item.category || "—"}</td>
                          <td>{item.price != null ? `$${Number(item.price).toLocaleString()}` : "—"}</td>
                          <td className="recent-actions">
                            <button type="button" className="row-action-btn" onClick={() => openEdit(item)}>
                              Edit
                            </button>
                            <button
                              type="button"
                              className="row-action-btn row-action-danger"
                              onClick={() => handleDelete(item)}
                              disabled={deletingId === item.item_id}
                            >
                              {deletingId === item.item_id ? "Deleting…" : "Delete"}
                            </button>
                          </td>
                        </tr>
                      ))}
                      {recentItems.length === 0 && (
                        <tr>
                          <td colSpan={5} className="recent-empty">
                            No catalog items yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                  {recentTotal > 0 && (
                    <div className="recent-pagination">
                      <button
                        type="button"
                        disabled={recentOffset === 0}
                        onClick={() => setRecentOffset((o) => Math.max(0, o - RECENT_LIMIT))}
                      >
                        Prev
                      </button>
                      <span>
                        {recentOffset + 1}–{Math.min(recentOffset + RECENT_LIMIT, recentTotal)} of {recentTotal}
                      </span>
                      <button
                        type="button"
                        disabled={recentOffset + RECENT_LIMIT >= recentTotal}
                        onClick={() => setRecentOffset((o) => o + RECENT_LIMIT)}
                      >
                        Next
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </section>
      </main>

      {editingItem && (
        <EditItemModal
          itemId={editingItem.itemId}
          fields={editingItem.fields}
          imagePreviewUrl={editImagePreview}
          saving={editSaving}
          error={editError}
          onChange={updateEditField}
          onImageChosen={chooseEditImage}
          onCancel={closeEdit}
          onSave={saveEdit}
        />
      )}
    </div>
  );
}
