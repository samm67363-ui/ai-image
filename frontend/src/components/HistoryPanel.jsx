import React from "react";

export default function HistoryPanel({ history }) {
  if (history.length === 0) return null;

  return (
    <div className="history-section">
      <h3 className="history-title">Prediction History</h3>
      <div className="history-grid">
        {history.map((item) => (
          <div key={item.id} className="history-card">
            <img src={item.imageUrl} alt={item.className} className="history-image" />
            <div className="history-details">
              <p className="history-class">{item.className}</p>
              <p className="history-confidence">{item.confidence}% confidence</p>
              <p className="history-time">{item.time}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}