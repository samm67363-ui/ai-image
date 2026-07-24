import React, { useState } from "react";
import Hero from "../components/Hero.jsx";
import UploadBox from "../components/UploadBox.jsx";
import Loader from "../components/Loader.jsx";
import ResultCard from "../components/ResultCard.jsx";
import HistoryPanel from "../components/HistoryPanel.jsx";
import { predictImage } from "../services/api.js";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const handleImageSelected = (file, validationError) => {
    setError(validationError);
    setResult(null);

    if (!file) {
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictImage(selectedFile);
      setResult(data);

      setHistory((prev) => [
        {
          id: Date.now(),
          imageUrl: previewUrl,
          className: data.class,
          confidence: data.confidence,
          time: new Date().toLocaleTimeString(),
        },
        ...prev,
      ]);
    } catch (err) {
      if (err.code === "ECONNABORTED") {
        setError("Request timed out. The backend may be slow or offline.");
      } else if (err.response) {
        setError(err.response.data?.detail || "Prediction failed. Please try again.");
      } else if (err.request) {
        setError("Could not reach the backend server. Please check it's running.");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="home-page">
      <Hero />

      <UploadBox
        onImageSelected={handleImageSelected}
        onPredict={handlePredict}
        previewUrl={previewUrl}
        onRemove={handleRemove}
        isLoading={isLoading}
        error={error}
      />

      {isLoading && <Loader />}

      {result && !isLoading && <ResultCard result={result} />}

      <HistoryPanel history={history} />
    </main>
  );
}