import React from "react";

export default function Navbar({ theme, onToggleTheme }) {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="navbar-logo">🧠</span>
        <span className="navbar-title">AI Image Classifier</span>
      </div>
      <button
        className="theme-toggle-btn"
        onClick={onToggleTheme}
        aria-label="Toggle dark/light theme"
      >
        {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
      </button>
    </nav>
  );
}