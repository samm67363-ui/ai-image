import React from "react";

export default function ResultCard({ result }) {
  if (!result) return null;

  const isMri = result.mode === "mri";

  return (
    <div className="result-card">
      <div className="mode-badge">
        {isMri ? "🧠 Detected: Brain MRI scan" : "🖼️ Detected: General photo"}
      </div>

      <div className="result-header">
        <h2 className="result-class">{result.class}</h2>
        <span className="result-confidence">{result.confidence}%</span>
      </div>

      <div className="top5-list">
        <h3 className="top5-title">
          {isMri ? "All Class Predictions" : "Top 5 Predictions"}
        </h3>
        {result.top5.map((item, index) => (
          <div key={index} className="top5-row">
            <span className="top5-label">{item.class}</span>
            <div className="top5-bar-track">
              <div
                className="top5-bar-fill"
                style={{ width: `${item.confidence}%` }}
              ></div>
            </div>
            <span className="top5-percent">{item.confidence}%</span>
          </div>
        ))}
      </div>

      {isMri && (
        <p className="mri-disclaimer">
          Educational project only, not a medical device. Never use this
          for real diagnosis or treatment decisions -- always consult a
          qualified doctor.
        </p>
      )}
    </div>
  );
}
