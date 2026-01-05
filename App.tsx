
import React, { useState, useEffect, useCallback } from 'react';
import { dbService } from './db';
import { CropRecord, DashboardStats, CropStatus, UserSettings } from './types';
import { syncOfflineRecords, SyncProgress } from './geminiService';
import Dashboard from './components/Dashboard';
import Analyzer from './components/Analyzer';
import History from './components/History';
import Analytics from './components/Analytics';
import Insurance from './components/Insurance';
import Sidebar from './components/Sidebar';
import Settings from './components/Settings';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'analyze' | 'history' | 'analytics' | 'insurance' | 'settings'>('dashboard');
  const [records, setRecords] = useState<CropRecord[]>([]);
  const [settings, setSettings] = useState<UserSettings>(dbService.getSettings());
  const [stats, setStats] = useState<DashboardStats>({
    totalAnalyzed: 0,
    healthyPercentage: 0,
    averageYieldLoss: 0,
    activeAlerts: 0
  });
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [syncState, setSyncState] = useState<SyncProgress>({ current: 0, total: 0, status: 'idle' });

  const loadData = useCallback(() => {
    const allRecords = dbService.getRecords();
    setRecords(allRecords);
    
    if (allRecords.length > 0) {
      const healthy = allRecords.filter(r => r.status === CropStatus.HEALTHY).length;
      const avgLoss = allRecords.reduce((acc, r) => acc + r.analysis.expectedLoss, 0) / allRecords.length;
      const alerts = allRecords.filter(r => r.analysis.expectedLoss >= settings.insuranceThreshold).length;

      setStats({
        totalAnalyzed: allRecords.length,
        healthyPercentage: Math.round((healthy / allRecords.length) * 100),
        averageYieldLoss: Math.round(avgLoss),
        activeAlerts: alerts
      });
    }
  }, [settings.insuranceThreshold]);

  const processBackgroundSync = useCallback(async () => {
    if (!navigator.onLine || syncState.status === 'syncing') return;
    
    const pending = dbService.getPendingRecords();
    if (pending.length === 0) return;

    await syncOfflineRecords((progress) => {
      setSyncState(progress);
      if (progress.status === 'completed') {
        loadData();
        setTimeout(() => setSyncState(prev => ({ ...prev, status: 'idle' })), 3000);
      }
    });
  }, [syncState.status, loadData]);

  useEffect(() => {
    const handleStatus = () => {
      const online = navigator.onLine;
      setIsOnline(online);
      if (online && settings.autoSync) {
        processBackgroundSync();
      }
    };

    window.addEventListener('online', handleStatus);
    window.addEventListener('offline', handleStatus);
    
    loadData();

    if (navigator.onLine && settings.autoSync) {
      processBackgroundSync();
    }

    return () => {
      window.removeEventListener('online', handleStatus);
      window.removeEventListener('offline', handleStatus);
    };
  }, [loadData, processBackgroundSync, settings.autoSync]);

  const handleNewRecord = (record: CropRecord) => {
    const recordToSave = {
      ...record,
      isPendingSync: !isOnline,
      syncAttempts: 0
    };
    dbService.saveRecord(recordToSave);
    loadData();
    setActiveTab('dashboard');
  };

  const handleSettingsUpdate = (newSettings: UserSettings) => {
    dbService.saveSettings(newSettings);
    setSettings(newSettings);
    loadData();
  };

  return (
    <div className="flex min-h-screen bg-gray-50 text-gray-900">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="flex-1 p-4 md:p-8 overflow-y-auto">
        <header className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold text-emerald-800">AgroGuard AI</h1>
            <p className="text-gray-500">Intelligent Crop Health & Yield Management</p>
          </div>
          
          <div className="flex items-center gap-3">
            {syncState.status === 'syncing' ? (
              <span className="flex items-center gap-2 px-3 py-1 bg-emerald-600 text-white text-sm font-medium rounded-full border border-emerald-700 shadow-sm animate-pulse">
                <i className="fas fa-sync-alt fa-spin"></i> {syncState.currentCrop || 'Optimizing'} {syncState.current}/{syncState.total}
              </span>
            ) : syncState.status === 'completed' ? (
              <span className="flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-medium rounded-full border border-emerald-200">
                <i className="fas fa-check-circle"></i> Sync Complete
              </span>
            ) : !isOnline ? (
              <span className="flex items-center gap-2 px-3 py-1 bg-amber-100 text-amber-700 text-sm font-medium rounded-full border border-amber-200">
                <i className="fas fa-wifi-slash"></i> Zero-Internet Mode
              </span>
            ) : (
              <button 
                onClick={processBackgroundSync}
                className="flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-medium rounded-full border border-emerald-200 hover:bg-emerald-200 transition-colors"
              >
                <i className="fas fa-sync-alt"></i> Sync Cloud
              </button>
            )}
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-xl shadow-sm border border-gray-100">
              <div className={`w-3 h-3 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-amber-500'} animate-pulse`}></div>
              <span className="text-sm font-medium">{isOnline ? 'System Active' : 'Offline Ready'}</span>
            </div>
          </div>
        </header>

        {activeTab === 'dashboard' && <Dashboard stats={stats} recentRecords={records.slice(0, 5)} onAnalyze={() => setActiveTab('analyze')} />}
        {activeTab === 'analyze' && <Analyzer onResult={handleNewRecord} isOnline={isOnline} sensitivity={settings.stressSensitivity} stressThreshold={settings.stressThreshold} />}
        {activeTab === 'history' && <History records={records} onUpdate={loadData} />}
        {activeTab === 'analytics' && <Analytics records={records} />}
        {activeTab === 'insurance' && <Insurance records={records} threshold={settings.insuranceThreshold} />}
        {activeTab === 'settings' && <Settings settings={settings} onUpdate={handleSettingsUpdate} />}
      </main>
    </div>
  );
};

export default App;
