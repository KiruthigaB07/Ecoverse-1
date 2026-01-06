
import React from 'react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: any) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const menuItems = [
    { id: 'dashboard', icon: 'fa-th-large', label: 'Dashboard' },
    { id: 'analyze', icon: 'fa-microscope', label: 'New Analysis' },
    { id: 'history', icon: 'fa-history', label: 'Crop History' },
    { id: 'analytics', icon: 'fa-chart-line', label: 'Analytics' },
    { id: 'insurance', icon: 'fa-shield-halved', label: 'Insurance' },
    { id: 'settings', icon: 'fa-cog', label: 'Settings' },
  ];

  return (
    <aside className="w-20 md:w-64 bg-emerald-900 text-white flex flex-col transition-all duration-300">
      <div className="p-6 text-center md:text-left">
        <div className="flex items-center gap-3 justify-center md:justify-start">
          <div className="bg-white/10 p-2 rounded-lg">
            <i className="fas fa-leaf text-2xl text-emerald-400"></i>
          </div>
          <span className="hidden md:block font-bold text-xl tracking-tight">AgroGuard</span>
        </div>
      </div>

      <nav className="flex-1 mt-6">
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`w-full flex items-center gap-4 px-6 py-4 transition-colors ${
              activeTab === item.id 
              ? 'bg-emerald-800 text-white border-r-4 border-emerald-400' 
              : 'text-emerald-300 hover:bg-emerald-800/50 hover:text-white'
            }`}
          >
            <i className={`fas ${item.icon} text-lg w-6`}></i>
            <span className="hidden md:block font-medium">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="p-6 mt-auto">
        <div className="hidden md:block bg-emerald-800/50 p-4 rounded-xl border border-emerald-700/50">
          <p className="text-xs text-emerald-400 font-semibold uppercase mb-1">Current Field</p>
          <p className="text-sm font-medium">North Sector - Corn</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
