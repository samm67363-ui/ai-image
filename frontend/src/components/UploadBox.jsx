import React, { useCallback, useRef, useState } from "react";

const MAX_SIZE_MB = 8;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function UploadBox({
  onImageSelected,
  onPredict,
  previewUrl,
  onRemove,
  isLoading,
  error,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndSelect = useCallback(
    (file) => {
      if (!file) return;

      if (!ACCEPTED_TYPES.includes(file.type)) {
        onImageSelected(null, "Unsupported file type. Please upload a JPEG, PNG, WEBP, or BMP image.");
        return;
      }

      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        onImageSelected(null, `File too large. Max size is ${MAX_SIZE_MB} MB.`);
        return;
      }

      onImageSelected(file, null);
    },
    [onImageSelected]
  );

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    validateAndSelect(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    validateAndSelect(file);
  };

  return (
    <div className="upload-section">
      {!previewUrl ? (
        <div
          className={`upload-dropzone ${isDragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(",")}
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          <div className="upload-icon">📤</div>
          <p className="upload-text">Drag &amp; drop an image here</p>
          <p className="upload-subtext">or click to browse (JPEG, PNG, WEBP, BMP — max {MAX_SIZE_MB}MB)</p>
        </div>
      ) : (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-image" />
          <div className="preview-actions">
            <button className="btn btn-primary" onClick={onPredict} disabled={isLoading}>
              {isLoading ? "Predicting..." : "Predict"}
            </button>
            <button className="btn btn-secondary" onClick={onRemove} disabled={isLoading}>
              Remove
            </button>
          </div>
        </div>
      )}

      {error && <p className="error-message">⚠️ {error}</p>}
    </div>
  );
}