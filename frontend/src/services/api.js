import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await apiClient.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}

export async function checkBackendHealth() {
  try {
    await apiClient.get("/");
    return true;
  } catch {
    return false;
  }
}