import React from "react";

export default function Hero() {
  return (
    <section className="hero">
      <h1 className="hero-title">
        AI <span className="gradient-text">Image Classifier</span>
      </h1>
      <p className="hero-subtitle">
        Upload any image. Everyday photos are identified using a
        general-purpose model; brain MRI scans are automatically detected
        and routed to a model fine-tuned for tumor classification.
      </p>
    </section>
  );
}
