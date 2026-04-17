import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000
});

// Prédiction
export const predictRevenue = async (data) => {
  const response = await api.post('/api/predict', data);
  return response.data;
};

// Entraînement
export const trainModel = async (algorithm, params) => {
  const response = await api.post('/api/train', { algorithm, params });
  return response.data;
};

// Résultats
export const getResults = async () => {
  const response = await api.get('/api/results');
  return response.data;
};

// Modèles disponibles
export const getModels = async () => {
  const response = await api.get('/api/models');
  return response.data;
};

// AutoML
export const runAutoML = async () => {
  const response = await api.post('/api/automl');
  return response.data;
};

// Historique
export const getHistory = async () => {
  const response = await api.get('/api/history');
  return response.data;
};

// Statistiques
export const getStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

// Feature importance
export const getFeatureImportance = async () => {
  const response = await api.get('/api/feature-importance');
  return response.data;
};

// Dataset - Films
export const getFilms = async (limit = 20, offset = 0, search = '') => {
  const response = await api.get('/api/dataset/films', { params: { limit, offset, search } });
  return response.data;
};

// Dataset - Statistiques
export const getDatasetStats = async () => {
  const response = await api.get('/api/dataset/stats');
  return response.data;
};

// Créer expérience
export const createExperiment = async (name, algorithm, params) => {
  const response = await api.post('/api/experiments', { name, algorithm, params });
  return response.data;
};

export default api;
