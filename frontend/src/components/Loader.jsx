import React from "react";

export default function Loader({ text = "Analyzing image..." }) {
  return (
    <div className="loader-container">
      <div className="loader-spinner"></div>
      <p className="loader-text">{text}</p>
    </div>
  );
}