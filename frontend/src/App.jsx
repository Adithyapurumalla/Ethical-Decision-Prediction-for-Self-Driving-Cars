import React, { useState, useEffect } from 'react';
import {
  Car, Shield, Cpu, BarChart2, Upload, History, Info,
  AlertTriangle, CheckCircle, Zap, Download, RefreshCw, Sun, Moon,
  CloudRain, CloudFog, Snowflake, AlertOctagon, UserCheck, ShieldAlert
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('studio');

  // Interactive Telemetry Parameters
  const [params, setParams] = useState({
    speed_mph: 45,
    pedestrian_count: 3,
    passenger_count: 1,
    obstacle_type: 'pedestrian_group',
    pedestrian_jaywalking: 0,
    weather_condition: 'clear',
    brake_status: 'failed',
    road_type: 'city_street',
    time_of_day: 'day',
    ethical_score: 0.85
  });

  // Prediction State
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);

  // History & Metrics State
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [report, setReport] = useState('');
  const [edaImages, setEdaImages] = useState([]);
  const [selectedEda, setSelectedEda] = useState('');

  // Auto-run inference whenever any parameter changes
  useEffect(() => {
    fetchPrediction();
  }, [params]);

  useEffect(() => {
    fetchMetricsAndReport();
    fetchEdaImages();
  }, []);

  const fetchPrediction = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (res.ok) {
        const data = await res.json();
        setPredictionData(data);
        setHistory(prev => [
          { ...params, Predicted_Decision: data.prediction, Confidence: data.confidence },
          ...prev.slice(0, 49)
        ]);
      }
    } catch (err) {
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchMetricsAndReport = async () => {
    try {
      const mRes = await fetch(`${API_BASE}/metrics`);
      if (mRes.ok) {
        const mData = await mRes.json();
        setMetrics(mData.metrics || []);
      }
      const rRes = await fetch(`${API_BASE}/report`);
      if (rRes.ok) {
        const rData = await rRes.json();
        setReport(rData.report || '');
      }
    } catch (err) {
      console.error('Fetch Metrics Error:', err);
    }
  };

  const fetchEdaImages = async () => {
    try {
      const res = await fetch(`${API_BASE}/eda-images`);
      if (res.ok) {
        const data = await res.json();
        setEdaImages(data.images || []);
        if (data.images && data.images.length > 0) {
          setSelectedEda(data.images[0]);
        }
      }
    } catch (err) {
      console.error('Fetch EDA Error:', err);
    }
  };

  const applyPreset = (preset) => {
    if (preset === 'school') {
      setParams({
        speed_mph: 30, pedestrian_count: 5, passenger_count: 1,
        obstacle_type: 'pedestrian_group', brake_status: 'failed',
        weather_condition: 'clear', road_type: 'school_zone',
        time_of_day: 'day', pedestrian_jaywalking: 0, ethical_score: 0.95
      });
    } else if (preset === 'barricade') {
      setParams({
        speed_mph: 60, pedestrian_count: 4, passenger_count: 2,
        obstacle_type: 'barricade', brake_status: 'failed',
        weather_condition: 'rain', road_type: 'highway',
        time_of_day: 'night', pedestrian_jaywalking: 0, ethical_score: 0.70
      });
    } else if (preset === 'animal') {
      setParams({
        speed_mph: 45, pedestrian_count: 1, passenger_count: 3,
        obstacle_type: 'animal', brake_status: 'functional',
        weather_condition: 'clear', road_type: 'city_street',
        time_of_day: 'dusk', pedestrian_jaywalking: 0, ethical_score: 0.90
      });
    } else if (preset === 'jaywalker') {
      setParams({
        speed_mph: 35, pedestrian_count: 1, passenger_count: 2,
        obstacle_type: 'jaywalker', brake_status: 'functional',
        weather_condition: 'fog', road_type: 'city_street',
        time_of_day: 'night', pedestrian_jaywalking: 1, ethical_score: 0.65
      });
    }
  };

  const updateParam = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Render Visual SVG Scenario Road Graphic
  const renderRoadSVG = () => {
    const decision = predictionData?.prediction || 'brake_hard';
    let pathD = "M 200 330 L 200 180";
    let color = "#10b981";
    let actionText = "MAINTAIN COURSE";

    if (decision === 'brake_hard') {
      pathD = "M 200 330 L 200 250";
      color = "#ef4444";
      actionText = "BRAKE HARD 🛑";
    } else if (decision === 'swerve_left') {
      pathD = "M 200 330 Q 200 230 100 140";
      color = "#f59e0b";
      actionText = "SWERVE LEFT ↙️";
    } else if (decision === 'swerve_right') {
      pathD = "M 200 330 Q 200 230 300 140";
      color = "#3b82f6";
      actionText = "SWERVE RIGHT ↘️";
    }

    const obsIcon = params.obstacle_type.includes('pedestrian') || params.obstacle_type === 'jaywalker' ? '🚶‍♂️🚶‍♀️' :
      params.obstacle_type === 'barricade' ? '🧱' : params.obstacle_type === 'animal' ? '🦌' : '🚘';

    return (
      <svg viewBox="0 0 400 380" style={{ width: '100%', maxHeight: '310px', background: '#090d16', borderRadius: '14px', border: '1px solid #1e293b' }}>
        <rect x="60" y="0" width="280" height="380" fill="#111827" />
        <line x1="200" y1="0" x2="200" y2="380" stroke="#334155" strokeWidth="4" strokeDasharray="15,15" />
        <line x1="60" y1="0" x2="60" y2="380" stroke="#f59e0b" strokeWidth="4" />
        <line x1="340" y1="0" x2="340" y2="380" stroke="#f59e0b" strokeWidth="4" />

        <path d={pathD} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" markerEnd="url(#arrow)" />

        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
          </marker>
        </defs>

        <g transform="translate(160, 70)">
          <rect x="-10" y="-10" width="100" height="50" rx="8" fill="rgba(239, 68, 68, 0.2)" stroke="#ef4444" strokeWidth="2" />
          <text x="40" y="22" fontSize="22" textAnchor="middle">{obsIcon}</text>
          <text x="40" y="55" fontSize="11" fill="#f87171" fontWeight="bold" textAnchor="middle">
            {params.obstacle_type.toUpperCase().replace('_', ' ')} (x{params.pedestrian_count})
          </text>
        </g>

        <g transform="translate(165, 270)">
          <rect x="0" y="0" width="70" height="90" rx="12" fill="#0284c7" stroke="#38bdf8" strokeWidth="3" />
          <rect x="8" y="15" width="54" height="25" rx="4" fill="#0f172a" />
          <text x="35" y="32" fontSize="14" fill="#38bdf8" textAnchor="middle">👤 x{params.passenger_count}</text>
          <circle cx="12" cy="5" r="4" fill="#fef08a" />
          <circle cx="58" cy="5" r="4" fill="#fef08a" />
          <text x="35" y="70" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">{params.speed_mph} MPH</text>
        </g>

        <rect x="20" y="20" width="160" height="35" rx="6" fill="rgba(15, 23, 42, 0.9)" stroke={color} strokeWidth="1.5" />
        <text x="100" y="42" fontSize="12" fill={color} fontWeight="bold" textAnchor="middle">{actionText}</text>
      </svg>
    );
  };

  return (
    <div className="container">
      {/* Header Banner */}
      <header className="hero-header">
        <div>
          <h1 className="hero-title">🚗 Ethical Decision Prediction for Self-Driving Cars</h1>
          <p className="hero-subtitle">High-Tech Cockpit & Real-Time AI Moral Dilemma Simulator</p>
        </div>
        <div className="status-badge">
          <div className="status-dot"></div>
          Live Inference Engine Active
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs-nav">
        <button className={`tab-btn ${activeTab === 'studio' ? 'active' : ''}`} onClick={() => setActiveTab('studio')}>
          <Zap size={18} /> Telemetry Cockpit
        </button>
        <button className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`} onClick={() => setActiveTab('batch')}>
          <Upload size={18} /> Batch Prediction
        </button>
        <button className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
          <BarChart2 size={18} /> Visual Analytics
        </button>
        <button className={`tab-btn ${activeTab === 'performance' ? 'active' : ''}`} onClick={() => setActiveTab('performance')}>
          <Cpu size={18} /> Model Benchmarks
        </button>
        <button className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
          <History size={18} /> Prediction History
        </button>
      </nav>

      {/* TAB 1: TELEMETRY COCKPIT */}
      {activeTab === 'studio' && (
        <div>
          <div className="studio-grid">
            {/* Interactive Control Panel */}
            <div className="glass-card">
              <h3>⚙️ Telemetry & Scenario Cockpit</h3>

              {/* Vehicle Speed Slider */}
              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <label>Vehicle Speed</label>
                  <strong style={{ color: 'var(--accent-cyan)' }}>{params.speed_mph} MPH</strong>
                </div>
                <input
                  type="range" min="10" max="90" step="5"
                  value={params.speed_mph}
                  onChange={e => updateParam('speed_mph', parseInt(e.target.value))}
                />
              </div>

              {/* Brake System Status Toggle */}
              <div className="form-group">
                <label>Mechanical Brake Status</label>
                <div className="toggle-switch-container">
                  <div
                    className={`toggle-switch-btn ${params.brake_status === 'functional' ? 'active-green' : ''}`}
                    onClick={() => updateParam('brake_status', 'functional')}
                  >
                    🟢 Functional
                  </div>
                  <div
                    className={`toggle-switch-btn ${params.brake_status === 'failed' ? 'active-red' : ''}`}
                    onClick={() => updateParam('brake_status', 'failed')}
                  >
                    🔴 Brakes Failed
                  </div>
                </div>
              </div>

              {/* Obstacle Type Option Chips */}
              <div className="form-group">
                <label>Obstacle / Target Category</label>
                <div className="chip-grid">
                  {[
                    { id: 'pedestrian_group', label: '🚸 Pedestrians' },
                    { id: 'barricade', label: '🧱 Barricade' },
                    { id: 'animal', label: '🦌 Animal' },
                    { id: 'vehicle', label: '🚘 Vehicle' },
                    { id: 'jaywalker', label: '🚶 Jaywalker' }
                  ].map(item => (
                    <div
                      key={item.id}
                      className={`option-chip ${params.obstacle_type === item.id ? 'active' : ''}`}
                      onClick={() => updateParam('obstacle_type', item.id)}
                    >
                      {item.label}
                    </div>
                  ))}
                </div>
              </div>

              {/* Counts Steppers */}
              <div className="flex-row" style={{ margin: '20px 0' }}>
                <div className="form-group">
                  <label>Pedestrian Count</label>
                  <div className="counter-stepper">
                    <button className="counter-btn" onClick={() => updateParam('pedestrian_count', Math.max(1, params.pedestrian_count - 1))}>-</button>
                    <div className="counter-value">{params.pedestrian_count}</div>
                    <button className="counter-btn" onClick={() => updateParam('pedestrian_count', Math.min(10, params.pedestrian_count + 1))}>+</button>
                  </div>
                </div>
                <div className="form-group">
                  <label>Car Occupants</label>
                  <div className="counter-stepper">
                    <button className="counter-btn" onClick={() => updateParam('passenger_count', Math.max(1, params.passenger_count - 1))}>-</button>
                    <div className="counter-value">{params.passenger_count}</div>
                    <button className="counter-btn" onClick={() => updateParam('passenger_count', Math.min(6, params.passenger_count + 1))}>+</button>
                  </div>
                </div>
              </div>

              {/* Road Setting Chips */}
              <div className="form-group">
                <label>Road Setting</label>
                <div className="chip-grid">
                  {[
                    { id: 'city_street', label: '🏙️ City Street' },
                    { id: 'highway', label: '🛣️ Highway' },
                    { id: 'school_zone', label: '🏫 School Zone' },
                    { id: 'intersection', label: '🚦 Intersection' }
                  ].map(item => (
                    <div
                      key={item.id}
                      className={`option-chip ${params.road_type === item.id ? 'active' : ''}`}
                      onClick={() => updateParam('road_type', item.id)}
                    >
                      {item.label}
                    </div>
                  ))}
                </div>
              </div>

              {/* Weather Condition Chips */}
              <div className="form-group">
                <label>Weather Condition</label>
                <div className="chip-grid">
                  {[
                    { id: 'clear', label: '☀️ Clear' },
                    { id: 'rain', label: '🌧️ Rain' },
                    { id: 'fog', label: '🌫️ Fog' },
                    { id: 'snow', label: '❄️ Snow' }
                  ].map(item => (
                    <div
                      key={item.id}
                      className={`option-chip ${params.weather_condition === item.id ? 'active' : ''}`}
                      onClick={() => updateParam('weather_condition', item.id)}
                    >
                      {item.label}
                    </div>
                  ))}
                </div>
              </div>

              {/* Crossing Signal Toggle */}
              <div className="form-group">
                <label>Pedestrian Signal Status</label>
                <div className="toggle-switch-container">
                  <div
                    className={`toggle-switch-btn ${params.pedestrian_jaywalking === 0 ? 'active-green' : ''}`}
                    onClick={() => updateParam('pedestrian_jaywalking', 0)}
                  >
                    🟢 Legal Crossing (0)
                  </div>
                  <div
                    className={`toggle-switch-btn ${params.pedestrian_jaywalking === 1 ? 'active-red' : ''}`}
                    onClick={() => updateParam('pedestrian_jaywalking', 1)}
                  >
                    🔴 Jaywalking (1)
                  </div>
                </div>
              </div>
            </div>

            {/* Simulation Output Card */}
            <div className="glass-card">
              <h3>🎯 Live AI Prediction & Trajectory</h3>

              {predictionData ? (
                <div>
                  <div style={{ textAlign: 'center' }}>
                    <div className={`badge-action badge-${predictionData.prediction}`}>
                      {predictionData.prediction.toUpperCase().replace('_', ' ')}
                    </div>
                  </div>

                  <div className="proba-item" style={{ marginTop: '10px' }}>
                    <div className="proba-label">
                      <span>Model Confidence</span>
                      <span>{predictionData.confidence}%</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div className="progress-bar-fill" style={{ width: `${predictionData.confidence}%` }}></div>
                    </div>
                  </div>

                  <div style={{ margin: '16px 0' }}>
                    {renderRoadSVG()}
                  </div>

                  <h4>Class Probabilities Breakdown</h4>
                  {predictionData.probabilities.map((item, idx) => (
                    <div key={idx} className="proba-item">
                      <div className="proba-label">
                        <span>{item.label}</span>
                        <span>{item.probability}%</span>
                      </div>
                      <div className="progress-bar-bg">
                        <div className="progress-bar-fill" style={{ width: `${item.probability}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>Calculating real-time inference...</p>
              )}
            </div>
          </div>

          {/* Decision Rationale */}
          {predictionData && (
            <div className="glass-card">
              <h3>💡 AI Decision Rationale & Explanation</h3>
              <div className="explanation-box">
                {predictionData.explanation}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: BATCH SCENARIO UPLOAD */}
      {activeTab === 'batch' && (
        <div className="glass-card">
          <h3>📁 Batch Scenario Prediction</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '16px' }}>
            Batch inference feature is active via REST API endpoint <code>POST /api/batch-predict</code>.
          </p>
          <button className="btn-primary" onClick={() => alert('Batch file processing API is active on endpoint /api/batch-predict.')}>
            <Upload size={18} style={{ display: 'inline', marginRight: '8px' }} /> Upload CSV
          </button>
        </div>
      )}

      {/* TAB 3: VISUAL ANALYTICS */}
      {activeTab === 'analytics' && (
        <div className="glass-card">
          <h3>📊 EDA Gallery & Visual Reports</h3>
          {edaImages.length > 0 ? (
            <div>
              <div className="form-group" style={{ maxWidth: '400px', marginBottom: '20px' }}>
                <label>Select Visualization Figure</label>
                <select className="form-control" value={selectedEda} onChange={e => setSelectedEda(e.target.value)}>
                  {edaImages.map((img, i) => (
                    <option key={i} value={img}>{img}</option>
                  ))}
                </select>
              </div>
              {selectedEda && (
                <div style={{ textAlign: 'center' }}>
                  <img
                    src={`http://localhost:8000/outputs/eda/${selectedEda}`}
                    alt={selectedEda}
                    style={{ maxWidth: '100%', borderRadius: '12px', border: '1px solid var(--card-border)' }}
                  />
                </div>
              )}
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>No EDA images found. Ensure `eda.py` has been executed.</p>
          )}
        </div>
      )}

      {/* TAB 4: MODEL PERFORMANCE */}
      {activeTab === 'performance' && (
        <div className="glass-card">
          <h3>⚡ Model Benchmark & Metrics Table</h3>
          {metrics.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>5-Fold CV F1</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((row, idx) => (
                  <tr key={idx}>
                    <td><strong>{row.Model}</strong></td>
                    <td>{row.Accuracy}</td>
                    <td>{row.Precision}</td>
                    <td>{row.Recall}</td>
                    <td>{row['F1 Score']}</td>
                    <td>{row['5-Fold CV F1']}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Metrics loading or unavailable.</p>
          )}

          {report && (
            <div style={{ marginTop: '24px' }}>
              <h4>Detailed Training Report (model_report.txt)</h4>
              <pre style={{ background: '#0f172a', padding: '16px', borderRadius: '8px', overflowX: 'auto', color: '#38bdf8', fontSize: '0.85rem' }}>
                {report}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: PREDICTION HISTORY */}
      {activeTab === 'history' && (
        <div className="glass-card">
          <h3>📜 Session Prediction Log</h3>
          {history.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Speed</th>
                  <th>Brakes</th>
                  <th>Obstacle</th>
                  <th>Pedestrians</th>
                  <th>Occupants</th>
                  <th>Predicted Action</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row, idx) => (
                  <tr key={idx}>
                    <td>{idx + 1}</td>
                    <td>{row.speed_mph} MPH</td>
                    <td>{row.brake_status}</td>
                    <td>{row.obstacle_type}</td>
                    <td>{row.pedestrian_count}</td>
                    <td>{row.passenger_count}</td>
                    <td><strong style={{ color: 'var(--accent-cyan)' }}>{row.Predicted_Decision}</strong></td>
                    <td>{row.Confidence}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>No predictions recorded yet.</p>
          )}
        </div>
      )}
    </div>
  );
}
