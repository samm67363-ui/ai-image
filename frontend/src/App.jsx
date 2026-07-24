import React, { useEffect, useState } from "react";
import Navbar from "./components/Navbar.jsx";
import Home from "./pages/Home.jsx";

export default function App() {
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    document.body.className = theme === "dark" ? "theme-dark" : "theme-light";
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  return (
    <div className="app-container">
      <Navbar theme={theme} onToggleTheme={toggleTheme} />
      <Home />
    </div>
  );
}