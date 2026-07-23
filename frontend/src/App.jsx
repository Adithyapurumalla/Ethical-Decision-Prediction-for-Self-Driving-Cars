import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';
import {
  LayoutDashboard, Camera, Cpu, BarChart3, Sliders, Activity,
  History, Settings, Shield, Gauge, MapPin, Users, Sun, CloudRain,
  AlertTriangle, CheckCircle2, Zap, Download, Search, ChevronLeft,
  ChevronRight, RefreshCw, Radio, HardDrive, Server, Layers, Crosshair
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

// Sample Recharts Data
const trafficData = [
  { time: '12:00', density: 35, pedestrians: 2, latency: 16 },
  { time: '12:05', density: 45, pedestrians: 5, latency: 18 },
  { time: '12:10', density: 60, pedestrians: 8, latency: 19 },
  { time: '12:15', density: 50, pedestrians: 4, latency: 17 },
  { time: '12:20', density: 75, pedestrians: 9, latency: 22 },
  { time: '12:25', density: 40, pedestrians: 3, latency: 18 },
];

const decisionDistData = [
  { name: 'Maintain Course', value: 48, color: '#10B981' },
  { name: 'Brake Hard', value: 32, color: '#EF4444' },
  { name: 'Swerve Left', value: 12, color: '#F59E0B' },
  { name: 'Swerve Right', value: 8, color: '#3B82F6' },
];

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');

  // Real-Time Telemetry State
  const [telemetry, setTelemetry] = useState({
    speed_mph: 45,
    pedestrian_count: 3,
    passenger_count: 1,
    obstacle_type: 'pedestrian_group',
    pedestrian_jaywalking: 0,
    weather_condition: 'clear',
    brake_status: 'functional',
    road_type: 'city_street',
    time_of_day: 'day',
    ethical_score: 0.85,
    traffic_signal: 'GREEN',
    road_condition: 'Dry Asphalt',
    traffic_density: 'Moderate (45%)',
    gps: '37.7749° N, 122.4194° W'
  });

  // Prediction Output State
  const [prediction, setPrediction] = useState({
    decision: 'maintain_course',
    confidence: 98.7,
    risk_score: 12,
    inference_time: 18,
    model_name: 'XGBoost + SHAP v3.2',
    explanation: 'The pedestrian is outside the vehicle trajectory. Current lane is clear. No collision risk detected.'
  });

  // System Uptime Clock
  const [timeStr, setTimeStr] = useState('');
  const [uptime, setUptime] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString() + ' | ' + now.toLocaleDateString());
      setUptime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Auto-fetch prediction on telemetry change
  useEffect(() => {
    fetchPrediction();
  }, [telemetry]);

  const fetchPrediction = async () => {
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetry)
      });
      if (res.ok) {
        const data = await res.json();
        setPrediction(prev => ({
          ...prev,
          decision: data.prediction,
          confidence: data.confidence,
          explanation: data.explanation
        }));
      }
    } catch (e) {
      console.log('API Offline, using simulated prediction');
    }
  };

  // SHAP Feature Importance Mock/Dynamic Values
  const shapFeatures = [
    { name: 'Pedestrian Count', importance: 32, value: telemetry.pedestrian_count },
    { name: 'Brake Status', importance: 24, value: telemetry.brake_status },
    { name: 'Weather Condition', importance: 18, value: telemetry.weather_condition },
    { name: 'Traffic Signal', importance: 15, value: telemetry.traffic_signal },
    { name: 'Vehicle Speed', importance: 11, value: `${telemetry.speed_mph} MPH` },
    { name: 'Road Type', importance: 7, value: telemetry.road_type }
  ];

  const sidebarItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'camera', label: 'Live Camera', icon: Camera },
    { id: 'decision', label: 'Decision Engine', icon: Cpu },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'simulation', label: 'Scenario Simulation', icon: Sliders },
    { id: 'performance', label: 'Model Performance', icon: Activity },
    { id: 'history', label: 'Decision History', icon: History },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-[#0B1220] text-slate-100 flex flex-col font-sans antialiased">
      {/* ========================================================================= */}
      {/* ENTERPRISE TOP HEADER */}
      {/* ========================================================================= */}
      <header className="h-16 border-b border-slate-800/80 bg-[#0B1220]/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
        {/* Left Title & Logo */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
              Autonomous Ethics Decision Intelligence
              <span className="text-[10px] font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-full">PROD-v3.2</span>
            </h1>
            <p className="text-xs text-slate-400">Real-Time Ethical Decision Support for Autonomous Vehicles</p>
          </div>
        </div>

        {/* Right Status Chips & Live Clock */}
        <div className="hidden lg:flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-xs">
            <span className="inline-flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-1 rounded-full font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> AI Model Online
            </span>
            <span className="inline-flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-1 rounded-full font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> Camera Connected
            </span>
            <span className="inline-flex items-center gap-1.5 bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2.5 py-1 rounded-full font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span> GPS Active
            </span>
          </div>

          <div className="h-6 w-[1px] bg-slate-800"></div>

          <div className="text-right">
            <div className="text-xs font-mono text-slate-300 font-semibold">{timeStr}</div>
            <div className="text-[10px] text-slate-500 font-mono">Uptime: {Math.floor(uptime/60)}m {uptime%60}s | 99.98%</div>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* MAIN CONTAINER (SIDEBAR + CONTENT) */}
      {/* ========================================================================= */}
      <div className="flex flex-1 overflow-hidden">
        {/* COLLAPSIBLE SIDEBAR */}
        <motion.aside
          animate={{ width: sidebarCollapsed ? 70 : 240 }}
          className="bg-[#0D1527] border-r border-slate-800/80 flex flex-col justify-between p-3 transition-all duration-300 relative"
        >
          <div className="space-y-1">
            {sidebarItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {!sidebarCollapsed && <span>{item.label}</span>}
                </button>
              );
            })}
          </div>

          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-center p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/60 transition-all border border-slate-800"
          >
            {sidebarCollapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
        </motion.aside>

        {/* MAIN DASHBOARD CONTENT (12-COLUMN RESPONSIVE GRID) */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* ===================================================================== */}
          {/* TOP ROW: 4 KPI CARDS */}
          {/* ===================================================================== */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-4 flex items-center justify-between glass-card-hover">
              <div>
                <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Vehicle Speed</span>
                <div className="text-2xl font-bold text-white mt-1 font-mono">{telemetry.speed_mph} <span className="text-xs font-sans text-slate-400">MPH</span></div>
                <span className="text-[10px] text-emerald-400 font-medium">↑ +2.4 mph normal cruise</span>
              </div>
              <div className="p-3 bg-blue-500/10 text-blue-400 rounded-xl border border-blue-500/20">
                <Gauge className="w-6 h-6" />
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="glass-panel p-4 flex items-center justify-between glass-card-hover">
              <div>
                <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Road Type</span>
                <div className="text-lg font-bold text-white mt-1 capitalize">{telemetry.road_type.replace('_', ' ')}</div>
                <span className="text-[10px] text-blue-400 font-medium">Zone 4B urban sector</span>
              </div>
              <div className="p-3 bg-indigo-500/10 text-indigo-400 rounded-xl border border-indigo-500/20">
                <MapPin className="w-6 h-6" />
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-panel p-4 flex items-center justify-between glass-card-hover">
              <div>
                <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Pedestrians Detected</span>
                <div className="text-2xl font-bold text-white mt-1 font-mono">{telemetry.pedestrian_count} <span className="text-xs font-sans text-slate-400">Persons</span></div>
                <span className="text-[10px] text-amber-400 font-medium">● Outside trajectory zone</span>
              </div>
              <div className="p-3 bg-amber-500/10 text-amber-400 rounded-xl border border-amber-500/20">
                <Users className="w-6 h-6" />
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="glass-panel p-4 flex items-center justify-between glass-card-hover">
              <div>
                <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Weather Condition</span>
                <div className="text-lg font-bold text-white mt-1 capitalize">{telemetry.weather_condition}</div>
                <span className="text-[10px] text-emerald-400 font-medium">Visibility 100% | 72°F</span>
              </div>
              <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
                <Sun className="w-6 h-6" />
              </div>
            </motion.div>
          </div>

          {/* ===================================================================== */}
          {/* MIDDLE GRID: LIVE CAMERA FEED HUD + AI DECISION PANEL */}
          {/* ===================================================================== */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

            {/* LIVE CAMERA FEED (8 COLS) */}
            <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="lg:col-span-8 glass-panel p-4 flex flex-col justify-between relative overflow-hidden min-h-[380px]">
              {/* HUD Header Bar */}
              <div className="flex items-center justify-between z-10 bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800 backdrop-blur-md">
                <div className="flex items-center space-x-3 text-xs">
                  <span className="flex items-center gap-1.5 text-emerald-400 font-mono font-bold">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span> 60.0 FPS
                  </span>
                  <span className="text-slate-500">|</span>
                  <span className="text-slate-300 font-mono">Model: XGBoost + SHAP</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-blue-400 font-mono">Latency: 18 ms</span>
                </div>
                <div className="text-xs text-slate-400 font-mono">CONFIDENCE: <strong className="text-white">98.7%</strong></div>
              </div>

              {/* Simulated Camera Video Stream HUD */}
              <div className="relative my-4 flex-1 rounded-xl bg-slate-950/90 border border-slate-800 flex items-center justify-center overflow-hidden">
                {/* Simulated Road Lines & Bounding Boxes */}
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-950/10 to-slate-950"></div>
                
                {/* Lane Detection Overlay Lines */}
                <svg className="absolute inset-0 w-full h-full stroke-blue-500/40 stroke-2" fill="none">
                  <line x1="20%" y1="100%" x2="42%" y2="40%" strokeDasharray="6,6" />
                  <line x1="80%" y1="100%" x2="58%" y2="40%" strokeDasharray="6,6" />
                </svg>

                {/* Bounding Box 1: Pedestrian */}
                <div className="absolute top-1/3 left-1/4 w-16 h-28 border-2 border-amber-400 bg-amber-500/10 rounded flex flex-col justify-between p-1 text-[9px] font-mono font-bold text-amber-400 shadow-lg shadow-amber-500/20">
                  <span>[PEDESTRIAN 0.96]</span>
                  <span>DIST: 24m</span>
                </div>

                {/* Bounding Box 2: Vehicle */}
                <div className="absolute top-1/4 right-1/3 w-32 h-20 border-2 border-emerald-400 bg-emerald-500/10 rounded flex flex-col justify-between p-1 text-[9px] font-mono font-bold text-emerald-400 shadow-lg shadow-emerald-500/20">
                  <span>[VEHICLE 0.99]</span>
                  <span>DIST: 48m</span>
                </div>

                {/* Center Reticle Crosshair */}
                <div className="absolute text-blue-500/60 pointer-events-none">
                  <Crosshair className="w-12 h-12 stroke-1 animate-pulse" />
                </div>

                {/* HUD Overlay Stats */}
                <div className="absolute bottom-4 left-4 bg-slate-900/90 border border-slate-800 p-2.5 rounded-xl text-xs space-y-1 font-mono">
                  <div className="text-emerald-400">✓ Lane Keep Assist: ACTIVE</div>
                  <div className="text-blue-400">✓ Emergency Brake: STANDBY</div>
                  <div className="text-slate-300">Object Trackers: 4 Active</div>
                </div>

                {/* Bottom Right Recording Badge */}
                <div className="absolute bottom-4 right-4 flex items-center space-x-2 bg-red-950/80 border border-red-500/30 px-3 py-1 rounded-full text-xs text-red-400 font-mono">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>
                  <span>REC 1080p60</span>
                </div>
              </div>
            </motion.div>

            {/* LIVE DECISION PANEL (4 COLS) */}
            <motion.div initial={{ opacity: 0, x: 15 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-4 glass-panel p-5 flex flex-col justify-between space-y-4">
              <div>
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider flex items-center gap-2">
                    <Cpu className="w-4 h-4" /> AI Decision Panel
                  </h3>
                  <span className="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full font-mono">REAL-TIME</span>
                </div>

                {/* Decision Badge */}
                <div className="my-4 text-center">
                  <span className="text-xs text-slate-400 uppercase font-semibold">Current AI Action</span>
                  <div className="mt-2 inline-block bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-extrabold text-xl px-6 py-2.5 rounded-full shadow-lg shadow-emerald-600/30 tracking-wider uppercase border border-emerald-400/30">
                    {telemetry.brake_status === 'failed' ? 'SWERVE LEFT ↙️' : 'MAINTAIN COURSE ⬆️'}
                  </div>
                </div>

                {/* Confidence & Risk Metric Cards */}
                <div className="grid grid-cols-2 gap-3 my-4">
                  <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl text-center">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Confidence</span>
                    <div className="text-xl font-extrabold text-blue-400 font-mono mt-1">98.7%</div>
                  </div>
                  <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl text-center">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Risk Score</span>
                    <div className="text-xl font-extrabold text-emerald-400 font-mono mt-1">12% <span className="text-[9px] font-sans text-emerald-500">LOW</span></div>
                  </div>
                </div>

                {/* Ethical Reasoning Callout */}
                <div className="bg-blue-950/20 border-l-4 border-blue-500 p-3.5 rounded-r-xl text-xs text-slate-300 leading-relaxed font-sans">
                  <strong className="text-blue-400 block mb-1">Ethical Reasoning:</strong>
                  "{prediction.explanation}"
                </div>
              </div>

              {/* Execution Metadata */}
              <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                <span>Inference: <strong className="text-white">18 ms</strong></span>
                <span>Model: <strong className="text-white">XGBoost + SHAP</strong></span>
              </div>
            </motion.div>
          </div>

          {/* ===================================================================== */}
          {/* REAL-TIME TELEMETRY & SHAP EXPLAINABLE AI */}
          {/* ===================================================================== */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

            {/* REAL-TIME TELEMETRY CARDS (6 COLS) */}
            <div className="lg:col-span-6 glass-panel p-5 space-y-4">
              <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider flex items-center gap-2 border-b border-slate-800 pb-3">
                <Activity className="w-4 h-4" /> Live Telemetry Feed
              </h3>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 font-semibold uppercase block">Brake System</span>
                  <span className={`font-bold ${telemetry.brake_status === 'functional' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {telemetry.brake_status.toUpperCase()}
                  </span>
                </div>

                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 font-semibold uppercase block">Traffic Signal</span>
                  <span className="font-bold text-emerald-400">{telemetry.traffic_signal}</span>
                </div>

                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 font-semibold uppercase block">Traffic Density</span>
                  <span className="font-bold text-blue-400">{telemetry.traffic_density}</span>
                </div>

                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 font-semibold uppercase block">GPS Location</span>
                  <span className="font-bold text-slate-300 font-mono text-[10px]">{telemetry.gps}</span>
                </div>
              </div>
            </div>

            {/* EXPLAINABLE AI SHAP IMPORTANCE (6 COLS) */}
            <div className="lg:col-span-6 glass-panel p-5 space-y-3">
              <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="flex items-center gap-2"><Layers className="w-4 h-4" /> Why did the AI choose this action?</span>
                <span className="text-[10px] font-mono text-slate-400">SHAP Explainer</span>
              </h3>

              <div className="space-y-2.5 pt-1">
                {shapFeatures.map((feat, idx) => (
                  <div key={idx} className="space-y-1 text-xs">
                    <div className="flex justify-between font-semibold">
                      <span className="text-slate-300">{feat.name} ({feat.value})</span>
                      <span className="text-blue-400 font-mono">+{feat.importance}%</span>
                    </div>
                    <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${feat.importance * 2.5}%` }}
                        transition={{ duration: 0.6, delay: idx * 0.05 }}
                        className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ===================================================================== */}
          {/* RECHARTS LIVE ANALYTICS */}
          {/* ===================================================================== */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 glass-panel p-5">
              <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider mb-4 flex items-center justify-between">
                <span>Traffic Density & Pedestrian Trend</span>
                <span className="text-xs text-slate-500 font-normal">Real-Time Telemetry Stream</span>
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trafficData}>
                    <defs>
                      <linearGradient id="colorDensity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="time" stroke="#64748B" fontSize={11} />
                    <YAxis stroke="#64748B" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                    <Area type="monotone" dataKey="density" stroke="#3B82F6" fillOpacity={1} fill="url(#colorDensity)" name="Traffic Density (%)" />
                    <Line type="monotone" dataKey="pedestrians" stroke="#F59E0B" strokeWidth={2} name="Pedestrians" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="lg:col-span-4 glass-panel p-5">
              <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider mb-4">Decision Class Distribution</h3>
              <div className="h-64 flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={decisionDistData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                      {decisionDistData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* ===================================================================== */}
          {/* MODEL HEALTH MONITORING PANEL */}
          {/* ===================================================================== */}
          <div className="glass-panel p-5">
            <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <HardDrive className="w-4 h-4" /> Production Hardware & Model Health
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 text-xs font-mono">
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">GPU Usage</span>
                <strong className="text-blue-400 text-sm">42%</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">CPU Usage</span>
                <strong className="text-emerald-400 text-sm">18%</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">RAM Memory</span>
                <strong className="text-purple-400 text-sm">3.2 GB</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">FPS Rate</span>
                <strong className="text-emerald-400 text-sm">60.0</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">Avg Latency</span>
                <strong className="text-blue-400 text-sm">18 ms</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">Predictions</span>
                <strong className="text-white text-sm">12,480</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">Avg Confidence</span>
                <strong className="text-emerald-400 text-sm">96.4%</strong>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <span className="text-[10px] text-slate-500 font-sans block">Model Version</span>
                <strong className="text-slate-300 text-sm">v3.2.0-prod</strong>
              </div>
            </div>
          </div>

        </main>
      </div>
    </div>
  );
}
