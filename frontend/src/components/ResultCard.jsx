import React from "react";

export default function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div className="result-card">
      <div className="result-header">
        <h2 className="result-class">{result.class}</h2>
        <span className="result-confidence">{result.confidence}%</span>
      </div>

      <div className="top5-list">
        <h3 className="top5-title">Top 5 Predictions</h3>
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
    </div>
  );
}