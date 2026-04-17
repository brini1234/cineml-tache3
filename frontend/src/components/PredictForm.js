import React, { useState } from 'react';
import { predictRevenue } from '../services/api';

const PredictForm = () => {
  const [formData, setFormData] = useState({
    budget: 100,
    runtime: 120,
    popularity: 50,
    vote_average: 6.5,
    vote_count: 1000,
    release_year: 2023,
    genre: 'Action'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Horror', 'Sci-Fi'];

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictRevenue(formData);
      if (response.success) {
        setResult(response);
      } else {
        setError(response.error);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatMoney = (value) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    return `$${value.toLocaleString()}`;
  };

  return (
    <div style={{
      background: '#111118',
      border: '1px solid #1e1e2e',
      borderRadius: 12,
      padding: 24
    }}>
      <h3 style={{ marginBottom: 20, color: '#e8c547' }}>🎬 Prédiction Box-Office</h3>
      
      <form onSubmit={handleSubmit}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div>
            <label style={{ fontSize: 11, color: '#6b6b80', display: 'block', marginBottom: 6 }}>
              Budget (M$)
            </label>
            <input
              type="number"
              name="budget"
              value={formData.budget}
              onChange={handleChange}
              style={{
                width: '100%',
                padding: 10,
                background: '#0a0a0f',
                border: '1px solid #1e1e2e',
                borderRadius: 6,
                color: '#fff'
              }}
            />
          </div>
          
          <div>
            <label style={{ fontSize: 11, color: '#6b6b80', display: 'block', marginBottom: 6 }}>
              Durée (minutes)
            </label>
            <input
              type="number"
              name="runtime"
              value={formData.runtime}
              onChange={handleChange}
              style={{
                width: '100%',
                padding: 10,
                background: '#0a0a0f',
                border: '1px solid #1e1e2e',
                borderRadius: 6,
                color: '#fff'
              }}
            />
          </div>
          
          <div>
            <label style={{ fontSize: 11, color: '#6b6b80', display: 'block', marginBottom: 6 }}>
              Popularité
            </label>
            <input
              type="number"
              name="popularity"
              value={formData.popularity}
              onChange={handleChange}
              style={{
                width: '100%',
                padding: 10,
                background: '#0a0a0f',
                border: '1px solid #1e1e2e',
                borderRadius: 6,
                color: '#fff'
              }}
            />
          </div>
          
          <div>
            <label style={{ fontSize: 11, color: '#6b6b80', display: 'block', marginBottom: 6 }}>
              Note moyenne
            </label>
            <input
              type="number"
              step="0.1"
              name="vote_average"
              value={formData.vote_average}
              onChange={handleChange}
              style={{
                width: '100%',
                padding: 10,
                background: '#0a0a0f',
                border: '1px solid #1e1e2e',
                borderRadius: 6,
                color: '#fff'
              }}
            />
          </div>
          
          <div>
            <label style={{ fontSize: 11, color: '#6b6b80', display: 'block', marginBottom: 6 }}>
              Genre
            </label>
            <select
              name="genre"
              value={formData.genre}
              onChange={handleChange}
              style={{
                width: '100%',
                padding: 10,
                background: '#0a0a0f',
                border: '1px solid #1e1e2e',
                borderRadius: 6,
                color: '#fff'
              }}
            >
              {genres.map(g => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>
        </div>
        
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: 20,
            width: '100%',
            padding: 12,
            background: '#e8c547',
            color: '#000',
            border: 'none',
            borderRadius: 8,
            fontWeight: 'bold',
            cursor: 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          {loading ? '⏳ Calcul en cours...' : '🔮 Prédire le revenu'}
        </button>
      </form>
      
      {error && (
        <div style={{ marginTop: 16, padding: 12, background: 'rgba(248,113,113,.1)', borderRadius: 8, color: '#f87171' }}>
          ❌ {error}
        </div>
      )}
      
      {result && (
        <div style={{ marginTop: 16, padding: 16, background: 'rgba(232,197,71,.1)', borderRadius: 8, textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#6b6b80' }}>Revenue estimé</div>
          <div style={{ fontSize: 28, fontWeight: 'bold', color: '#e8c547' }}>
            {formatMoney(result.predicted_revenue)}
          </div>
          <div style={{ fontSize: 11, color: '#6b6b80', marginTop: 8 }}>
            Intervalle de confiance: {formatMoney(result.confidence_interval[0])} - {formatMoney(result.confidence_interval[1])}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictForm;
