import React from 'react';
import { UserSettings, StressSensitivity } from '../types';

interface SettingsProps {
  settings: UserSettings;
  onUpdate: (settings: UserSettings) => void;
}

const Settings: React.FC<SettingsProps> = ({ settings, onUpdate }) => {
  const sensitivities: StressSensitivity[] = ['Low', 'Standard', 'High', 'Aggressive'];

  const SENSITIVITY_INSIGHTS: Record<StressSensitivity, { 
    label: string; 
    desc: string; 
    icon: string; 
    colorClass: string; 
    bgClass: string; 
    reactivity: string;
    thresholdImpact: string;
    weightMultiplier: string;
  }> = {
    Low: {
      label: "Conservative Detection",
      desc: "Minimizes false positives. The AI will only flag stress when morphological symptoms are distinct and unambiguous. Best for stable environments.",
      icon: "fa-shield",
      colorClass: "text-blue-600",
      bgClass: "bg-blue-50",
      reactivity: "25%",
      thresholdImpact: "Requires a very low threshold (10-15%) for early alerts.",
      weightMultiplier: "0.8x"
    },
    Standard: {
      label: "Balanced Monitoring",
      desc: "The default profile optimized for most field conditions. Provides a reliable balance between early warning and diagnostic accuracy.",
      icon: "fa-balance-scale",
      colorClass: "text-emerald-600",
      bgClass: "bg-emerald-50",
      reactivity: "50%",
      thresholdImpact: "Standard 20-25% threshold is highly recommended.",
      weightMultiplier: "1.0x"
    },
    High: {
      label: "Proactive Early-Warning",
      desc: "Increased weight on subtle spectral shifts. Flags physiological stress patterns before they become clear visible necrotic lesions.",
      icon: "fa-eye",
      colorClass: "text-amber-600",
      bgClass: "bg-amber-50",
      reactivity: "75%",
      thresholdImpact: "Amplifies detected stress. Recommended threshold: 30-35%.",
      weightMultiplier: "1.2x"
    },
    Aggressive: {
      label: "Hyper-Sensitive Audit",
      desc: "Maximum reactivity to any physiological deviation. Highly useful during active regional outbreaks or for high-value specialty crops.",
      icon: "fa-bolt",
      colorClass: "text-rose-600",
      bgClass: "bg-rose-50",
      reactivity: "100%",
      thresholdImpact: "Significant boost to probability. Pair with 40%+ thresholds.",
      weightMultiplier: "1.4x"
    }
  };

  const handleChange = (key: keyof UserSettings, value: any) => {
    onUpdate({ ...settings, [key]: value });
  };

  const currentInsight = SENSITIVITY_INSIGHTS[settings.stressSensitivity];

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-3 text-emerald-800">
          <i className="fas fa-sliders-h"></i> System Configuration
        </h2>

        <div className="space-y-10">
          {/* Stress Detection Module */}
          <section className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <i className="fas fa-microscope text-emerald-600"></i> Symptomless Stress Detection
                </h3>
                <p className="text-sm text-gray-500 max-w-2xl leading-relaxed">
                  Configure sensitivity and early-warning thresholds for pre-symptomatic physiological stress.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <label className="text-xs font-black uppercase tracking-widest text-gray-400">Analysis Sensitivity Profile</label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {sensitivities.map((s) => (
                  <button
                    key={s}
                    onClick={() => handleChange('stressSensitivity', s)}
                    className={`px-4 py-3 rounded-xl font-bold text-sm transition-all border-2 ${
                      settings.stressSensitivity === s
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-md shadow-emerald-100'
                        : 'bg-gray-50 text-gray-500 border-gray-100 hover:border-emerald-200 hover:text-emerald-600'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>

              {/* Dynamic Insight Box */}
              <div className={`mt-4 p-5 rounded-2xl border transition-all duration-300 ${currentInsight.bgClass} ${currentInsight.colorClass.replace('text-', 'border-').replace('600', '100')}`}>
                <div className="flex gap-4">
                  <div className={`w-12 h-12 rounded-xl bg-white flex items-center justify-center text-xl shadow-sm ${currentInsight.colorClass}`}>
                    <i className={`fas ${currentInsight.icon}`}></i>
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <div className="flex items-center gap-2">
                        <h4 className="font-bold text-sm">{currentInsight.label}</h4>
                        <span className="text-[9px] font-black bg-white/60 px-2 py-0.5 rounded-full border border-current opacity-70">
                          {currentInsight.weightMultiplier} Weighting
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-black uppercase tracking-tight opacity-60">AI Reactivity</span>
                        <div className="w-20 h-1.5 bg-black/5 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-500 ${currentInsight.colorClass.replace('text-', 'bg-')}`} 
                            style={{ width: currentInsight.reactivity }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    <p className="text-xs opacity-80 leading-relaxed font-medium mb-3">
                      {currentInsight.desc}
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="flex items-center gap-2 px-3 py-2 bg-white/40 rounded-lg border border-white/50">
                        <i className={`fas fa-info-circle text-[10px] ${currentInsight.colorClass}`}></i>
                        <p className="text-[9px] font-bold uppercase tracking-tight">
                          <span className="opacity-60">Impact:</span> {currentInsight.thresholdImpact}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-2 bg-white/40 rounded-lg border border-white/50">
                        <i className={`fas fa-microchip text-[10px] ${currentInsight.colorClass}`}></i>
                        <p className="text-[9px] font-bold uppercase tracking-tight">
                          <span className="opacity-60">Strategy:</span> {settings.stressSensitivity === 'Aggressive' ? 'Outbreak Mode' : 'Monitoring Mode'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4 pt-4 border-t border-gray-50">
              <div className="flex justify-between items-center">
                <label className="text-xs font-black uppercase tracking-widest text-gray-400">Stress Probability Threshold</label>
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded ${currentInsight.bgClass} ${currentInsight.colorClass}`}>
                    {settings.stressSensitivity} Sensitivity Active
                  </span>
                  <span className="bg-emerald-100 text-emerald-700 px-3 py-1 rounded-lg font-black text-sm">{settings.stressThreshold}%</span>
                </div>
              </div>
              <div className="relative pt-2">
                <input
                  type="range"
                  min="5"
                  max="80"
                  step="5"
                  value={settings.stressThreshold}
                  onChange={(e) => handleChange('stressThreshold', parseInt(e.target.value))}
                  className="w-full accent-emerald-600 h-2 bg-gray-100 rounded-lg appearance-none cursor-pointer"
                />
                <div 
                  className="absolute -top-1 w-0.5 h-4 bg-emerald-400 opacity-50" 
                  style={{ left: `${(settings.stressThreshold - 5) / 75 * 100}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-[10px] text-gray-400 font-bold uppercase tracking-tighter">
                <div className="flex flex-col">
                  <span>Highly Reactive</span>
                  <span className="text-[8px] opacity-60">(Alerts frequently)</span>
                </div>
                <div className="flex flex-col text-right">
                  <span>Highly Verifiable</span>
                  <span className="text-[8px] opacity-60">(Alerts on certainty)</span>
                </div>
              </div>
              <div className="p-4 bg-emerald-50/50 rounded-2xl border border-emerald-100/50">
                <p className="text-[11px] text-emerald-800 leading-relaxed">
                  <span className="font-bold mr-1">Dynamic Guide:</span> 
                  The AI uses a <span className="font-bold underline">{currentInsight.weightMultiplier}</span> weight multiplier for visual markers. 
                  Currently, detection must reach a <span className="font-bold">{settings.stressThreshold}%</span> confidence level to be classified as 'Stressed'.
                </p>
              </div>
            </div>
          </section>

          {/* Insurance Trigger Module */}
          <section className="space-y-4 pt-6 border-t border-gray-100">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <i className="fas fa-shield-halved text-emerald-600"></i> Insurance Claim Threshold
            </h3>
            <p className="text-sm text-gray-500 leading-relaxed">
              Set the predicted yield loss percentage that triggers an automatic insurance alert.
            </p>
            <div className="flex items-center gap-6">
              <input
                type="range"
                min="10"
                max="50"
                step="5"
                value={settings.insuranceThreshold}
                onChange={(e) => handleChange('insuranceThreshold', parseInt(e.target.value))}
                className="flex-1 accent-emerald-600 h-2 bg-gray-100 rounded-lg appearance-none cursor-pointer"
              />
              <div className="w-16 h-12 bg-emerald-50 text-emerald-700 font-black flex items-center justify-center rounded-xl border border-emerald-100">
                {settings.insuranceThreshold}%
              </div>
            </div>
          </section>

          {/* Synchronization */}
          <section className="flex items-center justify-between p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div>
              <h3 className="font-bold text-emerald-900">Auto-Cloud Sync</h3>
              <p className="text-xs text-gray-500">Automatically re-process offline records when internet is restored.</p>
            </div>
            <button
              onClick={() => handleChange('autoSync', !settings.autoSync)}
              className={`w-14 h-8 rounded-full p-1 transition-colors relative ${
                settings.autoSync ? 'bg-emerald-500' : 'bg-gray-300'
              }`}
            >
              <div
                className={`w-6 h-6 bg-white rounded-full transition-transform transform ${
                  settings.autoSync ? 'translate-x-6' : 'translate-x-0'
                } shadow-sm`}
              ></div>
            </button>
          </section>
        </div>
      </div>
      
      <div className="p-8 bg-emerald-800 text-white rounded-3xl flex items-center justify-between">
        <div>
          <h3 className="font-bold text-xl mb-1">Field-Ready Configuration</h3>
          <p className="text-emerald-100 text-sm opacity-80">Thresholds are applied immediately to all future visual audits.</p>
        </div>
        <div className="text-4xl opacity-20">
          <i className="fas fa-leaf"></i>
        </div>
      </div>
    </div>
  );
};

export default Settings;
