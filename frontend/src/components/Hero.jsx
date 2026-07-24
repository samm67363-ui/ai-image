import React from "react";

export default function Hero() {
  return (
    <section className="hero">
      <h1 className="hero-title">
        See What <span className="gradient-text">AI Sees</span>
      </h1>
      <p className="hero-subtitle">
        Upload any image and get an instant AI-powered prediction, powered by
        a pretrained ResNet18 deep learning model — right in your browser.
      </p>
    </section>
  );
}