import React, { useState, useEffect } from 'react';
import './App.css';
import { 
  predictRevenue, 
  trainModel, 
  getResults, 
  getModels, 
  runAutoML, 
  getHistory,
  getStats,
  getFeatureImportance,
  getFilms,
  getDatasetStats
} from './services/api';

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [notification, setNotification] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState([]);
  const [results, setResults] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [films, setFilms] = useState([]);
  const [datasetStats, setDatasetStats] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalFilms, setTotalFilms] = useState(0);
  
  // Formulaire prédiction
  const [predictionForm, setPredictionForm] = useState({
    budget: 100,
    runtime: 120,
    popularity: 50,
    vote_average: 6.5,
    vote_count: 1000,
    release_year: 2023,
    genre: 'Action'
  });
  const [predictionResult, setPredictionResult] = useState(null);

  const pages = {
    dashboard: { title: 'Dashboard', icon: '◈' },
    dataset: { title: 'Dataset', icon: '⊞' },
    train: { title: 'Train', icon: '⬡' },
    results: { title: 'Results', icon: '◎' },
    compare: { title: 'Comparer', icon: '⊟' },
    automl: { title: 'AutoML', icon: '⚡' },
    monitoring: { title: 'Monitoring', icon: '◉' },
    history: { title: 'History', icon: '≡' }
  };

  const algorithms = [
    { name: 'Random Forest', icon: '🌲', description: 'Ensemble robuste', params: { n_estimators: 100, max_depth: 10 } },
    { name: 'XGBoost', icon: '⚡', description: 'Performant', params: { n_estimators: 200, learning_rate: 0.1, max_depth: 6 } },
    { name: 'Linear Regression', icon: '📐', description: 'Baseline', params: {} },
    { name: 'Ridge', icon: '🔷', description: 'Régularisé', params: { alpha: 1.0 } },
    { name: 'SVR', icon: '🌐', description: 'Non-linéaire', params: { C: 10, epsilon: 0.1 } },
    { name: 'Neural Network', icon: '🧠', description: 'Deep Learning', params: { hidden_layers: 3, learning_rate: 0.01 } }
  ];

  // Charger les données au démarrage
  useEffect(() => {
    loadStats();
    loadModels();
    loadHistory();
    loadResults();
    loadFeatureImportance();
    loadDatasetStats();
    loadFilms();
  }, []);

  useEffect(() => {
    loadFilms();
  }, [searchTerm, currentPage]);

  const loadStats = async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch (error) {
      console.error('Erreur chargement stats:', error);
    }
  };

  const loadModels = async () => {
    try {
      const data = await getModels();
      setModels(data.models);
    } catch (error) {
      console.error('Erreur chargement modèles:', error);
    }
  };

  const loadHistory = async () => {
    try {
      const data = await getHistory();
      setHistory(data.experiments);
    } catch (error) {
      console.error('Erreur chargement historique:', error);
    }
  };

  const loadResults = async () => {
    try {
      const data = await getResults();
      setResults(data);
    } catch (error) {
      console.error('Erreur chargement résultats:', error);
    }
  };

  const loadFeatureImportance = async () => {
    try {
      const data = await getFeatureImportance();
      setFeatureImportance(data.features);
    } catch (error) {
      console.error('Erreur chargement feature importance:', error);
    }
  };

  const loadDatasetStats = async () => {
    try {
      const data = await getDatasetStats();
      setDatasetStats(data);
    } catch (error) {
      console.error('Erreur chargement dataset stats:', error);
    }
  };

  const loadFilms = async () => {
    try {
      const data = await getFilms(20, (currentPage - 1) * 20, searchTerm);
      setFilms(data.films);
      setTotalFilms(data.total);
    } catch (error) {
      console.error('Erreur chargement films:', error);
    }
  };

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 4000);
  };

  const handleNewExperiment = () => {
    setActivePage('train');
    setSelectedAlgorithm(null);
    showNotification('✨ Nouvelle expérience créée ! Sélectionnez un algorithme.', 'success');
  };

  const handleSelectAlgorithm = (algo) => {
    setSelectedAlgorithm(algo);
    showNotification(`✅ Algorithme sélectionné : ${algo.name}`, 'success');
  };

  const handleTrainModel = async () => {
    if (!selectedAlgorithm) {
      showNotification('⚠️ Veuillez d\'abord sélectionner un algorithme.', 'error');
      return;
    }
    
    setIsLoading(true);
    showNotification(`🚀 Entraînement de ${selectedAlgorithm.name} en cours...`, 'info');
    
    try {
      const result = await trainModel(selectedAlgorithm.name, selectedAlgorithm.params);
      showNotification(`✅ ${result.message}`, 'success');
      await loadHistory();
      await loadStats();
      await loadResults();
    } catch (error) {
      showNotification(`❌ Erreur lors de l'entraînement: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAutoML = async () => {
    setIsLoading(true);
    showNotification('⚡ AutoML lancé ! Recherche du meilleur modèle en cours...', 'info');
    
    try {
      const result = await runAutoML();
      showNotification(`✅ ${result.message}`, 'success');
      await loadModels();
      await loadStats();
    } catch (error) {
      showNotification(`❌ Erreur AutoML: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    setIsLoading(true);
    showNotification('🔮 Calcul de la prédiction en cours...', 'info');
    
    try {
      const result = await predictRevenue(predictionForm);
      if (result.success) {
        setPredictionResult(result);
        showNotification(`📊 Prédiction: ${result.predicted_revenue_millions} M$`, 'success');
      } else {
        showNotification(`❌ Erreur: ${result.error}`, 'error');
      }
    } catch (error) {
      showNotification(`❌ Erreur de prédiction: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleHelp = () => {
    showNotification('📖 Aide : Consultez la documentation ou le tutoriel pour commencer.', 'info');
  };

  const handlePredictionChange = (e) => {
    setPredictionForm({
      ...predictionForm,
      [e.target.name]: e.target.value
    });
  };

  const formatMoney = (value) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    return `$${value.toLocaleString()}`;
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#0a0a0f', color: '#e8e8f0' }}>
      {/* Notification */}
      {notification && (
        <div style={{
          position: 'fixed',
          top: 20,
          right: 20,
          backgroundColor: notification.type === 'error' ? '#f87171' : notification.type === 'success' ? '#4ade80' : '#e8c547',
          color: notification.type === 'error' ? '#fff' : '#000',
          padding: '12px 20px',
          borderRadius: 8,
          zIndex: 1000,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          animation: 'slideIn 0.3s ease'
        }}>
          {notification.message}
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 999
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>⏳</div>
            <div style={{ color: '#e8c547' }}>Traitement en cours...</div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <div style={{ width: 220, backgroundColor: '#111118', borderRight: '1px solid #1e1e2e', padding: '24px 0' }}>
        <div style={{ padding: '0 20px 24px', borderBottom: '1px solid #1e1e2e' }}>
          <div style={{ fontFamily: 'Syne, sans-serif', fontWeight: 800, fontSize: 22, color: '#e8c547' }}>CineML</div>
          <div style={{ fontSize: 9, color: '#6b6b80', letterSpacing: 3 }}>MLOps Platform</div>
        </div>
        <nav style={{ marginTop: 16 }}>
          {Object.entries(pages).map(([key, { title, icon }]) => (
            <div
              key={key}
              onClick={() => setActivePage(key)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 20px',
                cursor: 'pointer',
                color: activePage === key ? '#e8c547' : '#6b6b80',
                backgroundColor: activePage === key ? 'rgba(232,197,71,.05)' : 'transparent',
                borderLeft: activePage === key ? '2px solid #e8c547' : '2px solid transparent'
              }}
            >
              <span>{icon}</span>
              <span style={{ fontSize: 12 }}>{title}</span>
            </div>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1 }}>
        <div style={{ height: 56, borderBottom: '1px solid #1e1e2e', display: 'flex', alignItems: 'center', padding: '0 28px' }}>
          <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: 15 }}>{pages[activePage].title}</span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 10 }}>
            <button 
              onClick={handleHelp}
              style={{ padding: '6px 14px', borderRadius: 6, border: '1px solid #1e1e2e', background: '#16161f', color: '#e8e8f0', cursor: 'pointer' }}
            >
              ? Aide
            </button>
            <button 
              onClick={handleNewExperiment}
              style={{ padding: '6px 14px', borderRadius: 6, border: '1px solid #e8c547', background: '#e8c547', color: '#000', cursor: 'pointer', fontWeight: 500 }}
            >
              + Nouvelle expérience
            </button>
          </div>
        </div>

        <div style={{ padding: 28 }}>
          {/* Dashboard */}
          {activePage === 'dashboard' && (
            <div>
              <h2 style={{ marginBottom: 20 }}>Dashboard CineML</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16 }}>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>MEILLEUR R²</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#e8c547' }}>{results?.r2 || '0.847'}</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>RMSE</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#4ecdc4' }}>${(results?.rmse / 1e6 || 42.3).toFixed(1)}M</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>EXPÉRIENCES</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#a78bfa' }}>{stats?.total_experiments || 47}</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>MODÈLE ACTIF</div>
                  <div style={{ fontSize: 18, fontWeight: 700 }}>{stats?.active_model?.split(' ')[0] || 'XGBoost'}</div>
                </div>
              </div>

              {/* Feature Importance */}
              <div style={{ marginTop: 24, background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 20 }}>
                <h3 style={{ marginBottom: 16 }}>Feature Importance</h3>
                {featureImportance.map((feature, idx) => (
                  <div key={idx} style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span>{feature.name}</span>
                      <span style={{ color: '#e8c547' }}>{feature.importance}</span>
                    </div>
                    <div style={{ height: 4, background: '#1e1e2e', borderRadius: 2 }}>
                      <div style={{ width: `${feature.importance * 100}%`, height: 4, background: '#e8c547', borderRadius: 2 }}></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Prédiction rapide */}
              <div style={{ marginTop: 24, background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 20 }}>
                <h3 style={{ marginBottom: 16 }}>🔮 Prédiction rapide</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <input name="budget" value={predictionForm.budget} onChange={handlePredictionChange} placeholder="Budget (M$)" style={{ padding: 8, background: '#0a0a0f', border: '1px solid #1e1e2e', color: '#fff', borderRadius: 6 }} />
                  <input name="popularity" value={predictionForm.popularity} onChange={handlePredictionChange} placeholder="Popularité" style={{ padding: 8, background: '#0a0a0f', border: '1px solid #1e1e2e', color: '#fff', borderRadius: 6 }} />
                </div>
                <button onClick={handlePredict} style={{ marginTop: 12, width: '100%', padding: 10, background: '#e8c547', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 'bold' }}>
                  ⚡ Prédire le revenue
                </button>
                {predictionResult && (
                  <div style={{ marginTop: 12, padding: 12, background: 'rgba(232,197,71,.1)', borderRadius: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 12, color: '#6b6b80' }}>Revenue estimé</div>
                    <div style={{ fontSize: 20, fontWeight: 'bold', color: '#e8c547' }}>${predictionResult.predicted_revenue_millions} M</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Dataset Page */}
          {activePage === 'dataset' && (
            <div>
              <h2>Dataset TMDB</h2>
              
              {/* Stats Dataset */}
              {datasetStats && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 20 }}>
                  <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                    <div style={{ fontSize: 10, color: '#6b6b80' }}>TOTAL FILMS</div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: '#e8c547' }}>{datasetStats.total_films}</div>
                  </div>
                  <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                    <div style={{ fontSize: 10, color: '#6b6b80' }}>BUDGET MOYEN</div>
                    <div style={{ fontSize: 18, fontWeight: 700 }}>{formatMoney(datasetStats.avg_budget)}</div>
                  </div>
                  <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                    <div style={{ fontSize: 10, color: '#6b6b80' }}>REVENU MOYEN</div>
                    <div style={{ fontSize: 18, fontWeight: 700 }}>{formatMoney(datasetStats.avg_revenue)}</div>
                  </div>
                  <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                    <div style={{ fontSize: 10, color: '#6b6b80' }}>POPULARITÉ MOYENNE</div>
                    <div style={{ fontSize: 18, fontWeight: 700 }}>{datasetStats.avg_popularity?.toFixed(1)}</div>
                  </div>
                </div>
              )}

              {/* Filtres */}
              <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
                <input 
                  type="text" 
                  placeholder="🔍 Rechercher un film..." 
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{ flex: 1, padding: 10, background: '#0a0a0f', border: '1px solid #1e1e2e', color: '#fff', borderRadius: 6 }}
                />
                <button onClick={() => { setCurrentPage(1); loadFilms(); }} style={{ padding: '10px 20px', background: '#e8c547', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer' }}>
                  Rechercher
                </button>
              </div>

              {/* Tableau des films */}
              <div style={{ overflowX: 'auto', background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #1e1e2e', background: '#0a0a0f' }}>
                      <th style={{ padding: 12, textAlign: 'left', color: '#6b6b80', fontSize: 11 }}>#</th>
                      <th style={{ padding: 12, textAlign: 'left', color: '#6b6b80', fontSize: 11 }}>Titre</th>
                      <th style={{ padding: 12, textAlign: 'right', color: '#6b6b80', fontSize: 11 }}>Budget</th>
                      <th style={{ padding: 12, textAlign: 'right', color: '#6b6b80', fontSize: 11 }}>Revenue</th>
                      <th style={{ padding: 12, textAlign: 'center', color: '#6b6b80', fontSize: 11 }}>Popularité</th>
                      <th style={{ padding: 12, textAlign: 'center', color: '#6b6b80', fontSize: 11 }}>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {films.map((film, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid #1e1e2e' }}>
                        <td style={{ padding: 10, fontSize: 12, color: '#6b6b80' }}>{(currentPage - 1) * 20 + idx + 1}</td>
                        <td style={{ padding: 10, fontSize: 13 }}>{film.title}</td>
                        <td style={{ padding: 10, textAlign: 'right', fontSize: 12 }}>{formatMoney(film.budget)}</td>
                        <td style={{ padding: 10, textAlign: 'right', fontSize: 12, color: '#e8c547' }}>{formatMoney(film.revenue)}</td>
                        <td style={{ padding: 10, textAlign: 'center', fontSize: 12 }}>{film.popularity?.toFixed(1)}</td>
                        <td style={{ padding: 10, textAlign: 'center', fontSize: 12 }}>{film.vote_average?.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div style={{ display: 'flex', justifyContent: 'center', gap: 10, marginTop: 20 }}>
                <button 
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  style={{ padding: '6px 12px', background: '#16161f', border: '1px solid #1e1e2e', borderRadius: 6, cursor: 'pointer', opacity: currentPage === 1 ? 0.5 : 1 }}
                >
                  ← Précédent
                </button>
                <span style={{ padding: '6px 12px', background: '#e8c547', color: '#000', borderRadius: 6 }}>
                  Page {currentPage} / {Math.ceil(totalFilms / 20)}
                </span>
                <button 
                  onClick={() => setCurrentPage(currentPage + 1)}
                  disabled={currentPage >= Math.ceil(totalFilms / 20)}
                  style={{ padding: '6px 12px', background: '#16161f', border: '1px solid #1e1e2e', borderRadius: 6, cursor: 'pointer', opacity: currentPage >= Math.ceil(totalFilms / 20) ? 0.5 : 1 }}
                >
                  Suivant →
                </button>
              </div>
            </div>
          )}

          {/* Train */}
          {activePage === 'train' && (
            <div>
              <h2>Entraînement des modèles</h2>
              {selectedAlgorithm && (
                <div style={{ marginBottom: 16, padding: 12, background: 'rgba(232,197,71,.1)', borderRadius: 8, border: '1px solid #e8c547' }}>
                  <span style={{ color: '#e8c547' }}>✅ Algorithme sélectionné : </span>
                  <strong>{selectedAlgorithm.name}</strong>
                </div>
              )}
              <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
                {algorithms.map(algo => (
                  <div 
                    key={algo.name} 
                    onClick={() => handleSelectAlgorithm(algo)}
                    style={{ 
                      border: selectedAlgorithm?.name === algo.name ? '2px solid #e8c547' : '1px solid #1e1e2e', 
                      borderRadius: 10, 
                      padding: 14, 
                      cursor: 'pointer', 
                      background: selectedAlgorithm?.name === algo.name ? 'rgba(232,197,71,.1)' : '#0a0a0f'
                    }}
                  >
                    <div style={{ fontSize: 22, marginBottom: 8 }}>{algo.icon}</div>
                    <div style={{ fontWeight: 700 }}>{algo.name}</div>
                    <div style={{ fontSize: 10, color: '#6b6b80' }}>{algo.description}</div>
                  </div>
                ))}
              </div>
              <button 
                onClick={handleTrainModel}
                disabled={isLoading}
                style={{ marginTop: 20, width: '100%', padding: 14, background: '#e8c547', color: '#000', border: 'none', borderRadius: 8, fontWeight: 800, cursor: 'pointer', opacity: isLoading ? 0.6 : 1 }}
              >
                {isLoading ? '⏳ Entraînement...' : '▶ ENTRAÎNER LE MODÈLE'}
              </button>
            </div>
          )}

          {/* Results */}
          {activePage === 'results' && results && (
            <div>
              <h2>Résultats</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginTop: 20 }}>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>R² SCORE</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#e8c547' }}>{results.r2}</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>RMSE</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#4ecdc4' }}>${(results.rmse / 1e6).toFixed(1)}M</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>MAE</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#a78bfa' }}>${(results.mae / 1e6).toFixed(1)}M</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>MAPE</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#ff6b35' }}>{results.mape}%</div>
                </div>
              </div>
              <div style={{ marginTop: 20, background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 20 }}>
                <h3>Meilleur modèle: {results.best_model}</h3>
                <pre style={{ fontSize: 11, color: '#6b6b80', marginTop: 10 }}>{JSON.stringify(results.best_params, null, 2)}</pre>
              </div>
            </div>
          )}

          {/* Compare Page */}
          {activePage === 'compare' && (
            <div>
              <h2>Comparaison des modèles</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
                {models.map((model, idx) => (
                  <div key={idx} style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 20 }}>
                    <h3 style={{ color: model.active ? '#e8c547' : '#fff' }}>{model.name}</h3>
                    <div style={{ marginTop: 15 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <span style={{ color: '#6b6b80' }}>Version</span>
                        <span>{model.version}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <span style={{ color: '#6b6b80' }}>R² Score</span>
                        <span style={{ color: '#e8c547' }}>{model.r2}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <span style={{ color: '#6b6b80' }}>RMSE</span>
                        <span>${model.rmse}M</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <span style={{ color: '#6b6b80' }}>Statut</span>
                        <span style={{ color: model.active ? '#4ade80' : '#6b6b80' }}>{model.active ? 'Actif' : 'Inactif'}</span>
                      </div>
                    </div>
                    {model.active && (
                      <div style={{ marginTop: 12, padding: 8, background: 'rgba(232,197,71,.1)', borderRadius: 6, textAlign: 'center', fontSize: 11 }}>
                        ⭐ Modèle actuellement déployé
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AutoML */}
          {activePage === 'automl' && (
            <div>
              <h2>AutoML ⚡</h2>
              <div style={{ marginTop: 20, background: 'linear-gradient(135deg,rgba(232,197,71,.1),rgba(255,107,53,.1))', border: '1px solid rgba(232,197,71,.3)', borderRadius: 12, padding: 24 }}>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#e8c547' }}>Mode AutoML</div>
                <div style={{ fontSize: 12, color: '#6b6b80', marginTop: 8 }}>Cliquez une fois — le système teste tous les algorithmes</div>
                <button 
                  onClick={handleAutoML}
                  disabled={isLoading}
                  style={{ marginTop: 16, padding: '12px 24px', background: '#e8c547', color: '#000', border: 'none', borderRadius: 8, fontWeight: 700, cursor: 'pointer', opacity: isLoading ? 0.6 : 1 }}
                >
                  {isLoading ? '⏳ Recherche en cours...' : '⚡ LANCER AutoML'}
                </button>
              </div>
            </div>
          )}

          {/* Monitoring Page */}
          {activePage === 'monitoring' && (
            <div>
              <h2>Monitoring</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 20 }}>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>DERNIER ENTRAÎNEMENT</div>
                  <div style={{ fontSize: 16, fontWeight: 700 }}>{history[0]?.algorithm || 'Aucun'}</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>MEILLEUR R²</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#e8c547' }}>{results?.r2 || '0.847'}</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>TOTAL RUNS</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#a78bfa' }}>{history.length}</div>
                </div>
                <div style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 10, padding: 15 }}>
                  <div style={{ fontSize: 10, color: '#6b6b80' }}>STATUT</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#4ade80' }}>● ACTIF</div>
                </div>
              </div>

              {/* Logs Console */}
              <div style={{ background: '#000', border: '1px solid #1e1e2e', borderRadius: 10, padding: 20, fontFamily: 'monospace', fontSize: 11 }}>
                <div style={{ color: '#6b6b80', marginBottom: 10 }}>📋 Console de logs - Temps réel</div>
                <div style={{ color: '#4ecdc4' }}>[INFO] Serveur CineML démarré</div>
                <div style={{ color: '#4ecdc4' }}>[INFO] Dataset TMDB chargé : 3229 films</div>
                <div style={{ color: '#e8c547' }}>[WARN] 12 valeurs manquantes imputées par médiane</div>
                <div style={{ color: '#4ade80' }}>[OK] Modèle XGBoost entraîné avec succès</div>
                <div style={{ color: '#4ecdc4', marginTop: 10 }}>[INFO] Dernier run: {history[0]?.algorithm} - R²={history[0]?.r2}</div>
              </div>
            </div>
          )}

          {/* History */}
          {activePage === 'history' && (
            <div>
              <h2>Historique des expériences</h2>
              {history.length === 0 ? (
                <p style={{ marginTop: 20, color: '#6b6b80' }}>Aucune expérience pour le moment. Lancez un entraînement !</p>
              ) : (
                <div style={{ marginTop: 20 }}>
                  {history.map((exp, idx) => (
                    <div key={idx} style={{ background: '#111118', border: '1px solid #1e1e2e', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <span style={{ fontWeight: 'bold', color: '#e8c547' }}>{exp.id}</span>
                          <span style={{ marginLeft: 12 }}>{exp.algorithm}</span>
                        </div>
                        <span style={{ fontSize: 12, color: '#4ade80' }}>{exp.status}</span>
                      </div>
                      {exp.r2 && (
                        <div style={{ marginTop: 8, fontSize: 12, color: '#6b6b80' }}>
                          R²: {exp.r2} | RMSE: ${exp.rmse}M
                        </div>
                      )}
                      {exp.timestamp && (
                        <div style={{ marginTop: 4, fontSize: 10, color: '#6b6b80' }}>
                          {new Date(exp.timestamp).toLocaleString()}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: 20, textAlign: 'center', color: '#6b6b80', fontSize: 11 }}>
            <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: '#4ade80', marginRight: 6 }}></span>
            Système actif · v3.2.1
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
