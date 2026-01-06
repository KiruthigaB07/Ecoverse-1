import React from 'react';
import { DashboardStats, CropRecord, CropStatus } from '../types';

interface DashboardProps {
  stats: DashboardStats;
  recentRecords: CropRecord[];
  onAnalyze: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ stats, recentRecords, onAnalyze }) => {
  return (
    <div className="space-y-8">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          label="Total Analyzed" 
          value={stats.totalAnalyzed.toString()} 
          icon="fa-database" 
          color="bg-blue-50 text-blue-600" 
        />
        <StatCard 
          label="Healthy Crops" 
          value={`${stats.healthyPercentage}%`} 
          icon="fa-check-circle" 
          color="bg-emerald-50 text-emerald-600" 
        />
        <StatCard 
          label="Avg. Yield Loss" 
          value={`${stats.averageYieldLoss}%`} 
          icon="fa-chart-area" 
          color="bg-amber-50 text-amber-600" 
        />
        <StatCard 
          label="Active Alerts" 
          value={stats.activeAlerts.toString()} 
          icon="fa-exclamation-triangle" 
          color="bg-rose-50 text-rose-600" 
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Action Card */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 flex flex-col md:flex-row items-center gap-8 group">
            <div className="w-full md:w-2/3">
              <h2 className="text-2xl font-black mb-3 text-emerald-900">Neural Visual Intelligence</h2>
              <p className="text-gray-500 mb-6 leading-relaxed text-sm">Our <strong>v4.2 Analysis Engine</strong> utilizes multi-spectral proxy detection and zonal feature extraction. Higher efficiency, lower latency, and superior pre-symptomatic stress identification.</p>
              <button 
                onClick={onAnalyze}
                className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-4 rounded-2xl font-black transition-all transform hover:scale-105 shadow-xl shadow-emerald-100 flex items-center gap-3"
              >
                <i className="fas fa-plus"></i> Initiate Analysis
              </button>
            </div>
            <div className="w-full md:w-1/3 flex justify-center">
              <div className="w-36 h-36 bg-emerald-50 rounded-3xl flex items-center justify-center text-emerald-600 text-5xl relative group-hover:rotate-6 transition-transform">
                <i className="fas fa-microscope"></i>
                <div className="absolute -top-3 -right-3 bg-emerald-600 text-white text-[9px] font-black px-3 py-1.5 rounded-full shadow-lg border-2 border-white uppercase tracking-widest">
                  v4.2 Stable
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-3xl shadow-sm border border-gray-100 overflow-hidden">
            <div className="p-6 border-b border-gray-50 flex justify-between items-center bg-gray-50/50">
              <h3 className="font-black text-gray-800 uppercase tracking-widest text-xs">Recent Pathology Scans</h3>
              <button className="text-emerald-600 text-xs font-black hover:underline uppercase tracking-widest">View History</button>
            </div>
            <div className="divide-y divide-gray-50">
              {recentRecords.length === 0 ? (
                <div className="p-16 text-center text-gray-400">
                  <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
                    <i className="fas fa-inbox"></i>
                  </div>
                  <p className="font-bold">No active records.</p>
                  <p className="text-xs">Start a diagnostic scan to see data.</p>
                </div>
              ) : (
                recentRecords.map(record => (
                  <div key={record.id} className="p-5 hover:bg-emerald-50 transition-colors flex items-center gap-5">
                    <div className="w-14 h-14 rounded-xl bg-gray-100 overflow-hidden flex-shrink-0 border-2 border-white shadow-sm">
                      {record.imageUrl ? (
                        <img src={record.imageUrl} className="w-full h-full object-cover" alt="Crop" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-300">
                          <i className="fas fa-leaf"></i>
                        </div>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-black text-emerald-950 truncate uppercase tracking-tight">{record.cropType}</p>
                      <p className="text-[10px] text-gray-400 font-bold uppercase">{new Date(record.timestamp).toLocaleString()}</p>
                    </div>
                    <div className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest shadow-sm ${
                      record.status === CropStatus.HEALTHY ? 'bg-emerald-100 text-emerald-700' :
                      record.status === CropStatus.STRESSED ? 'bg-amber-100 text-amber-700' :
                      'bg-rose-100 text-rose-700'
                    }`}>
                      {record.status}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Sidebar Cards */}
        <div className="space-y-6">
          <div className="bg-emerald-900 p-8 rounded-3xl text-white shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 text-6xl group-hover:rotate-12 transition-transform">
              <i className="fas fa-brain"></i>
            </div>
            <h3 className="font-black mb-6 flex items-center gap-2 text-sm uppercase tracking-widest">
              <i className="fas fa-shield-virus text-emerald-400"></i> AI Model Health
            </h3>
            <div className="space-y-5">
              <div className="flex justify-between items-center text-[10px] uppercase font-bold">
                <span className="opacity-60">Engine Load Mode</span>
                <span className="text-emerald-400">Optimized v4.2</span>
              </div>
              <div className="flex justify-between items-center text-[10px] uppercase font-bold">
                <span className="opacity-60">Zonal Inference</span>
                <span className="text-emerald-400">Active</span>
              </div>
              <div className="flex justify-between items-center text-[10px] uppercase font-bold">
                <span className="opacity-60">Local Latency</span>
                <span className="text-white">142ms</span>
              </div>
              <div className="pt-4 border-t border-white/10">
                <div className="flex justify-between mb-2 text-[10px] uppercase font-black tracking-widest text-emerald-400">
                  <span>Diagnostic Precision</span>
                  <span>99.1%</span>
                </div>
                <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-400 w-[99%] shadow-[0_0_10px_rgba(52,211,153,0.5)]"></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
            <h3 className="font-black text-[10px] mb-6 uppercase tracking-widest text-gray-400">Spectral Awareness</h3>
            <div className="flex items-center gap-6">
              <div className="text-4xl text-amber-500">
                <i className="fas fa-cloud-sun"></i>
              </div>
              <div>
                <p className="text-3xl font-black text-gray-900">28°C</p>
                <p className="text-[10px] font-bold text-gray-400 uppercase">Humidity: 42% • UV: 6.2</p>
              </div>
            </div>
            <div className="mt-6 p-4 bg-blue-50 rounded-2xl text-[10px] text-blue-700 font-bold uppercase tracking-tight leading-relaxed">
              <i className="fas fa-bolt mr-2"></i> High solar intensity detected. Recommended scan window: Early Morning or Late Afternoon for optimal contrast.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, icon, color }: any) => (
  <div className="bg-white p-7 rounded-3xl shadow-sm border border-gray-100 transition-transform hover:-translate-y-1">
    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center mb-6 shadow-inner ${color}`}>
      <i className={`fas ${icon} text-xl`}></i>
    </div>
    <p className="text-3xl font-black text-gray-900 tracking-tight">{value}</p>
    <p className="text-[10px] text-gray-400 font-black uppercase tracking-widest mt-1">{label}</p>
  </div>
);

export default Dashboard;
