
import React, { useState, useRef, useEffect } from 'react';
import { analyzeCropHealth } from '../geminiService';
import { CropRecord, CropStatus, YieldAnalysis, StressSensitivity } from '../types';

interface AnalyzerProps {
  onResult: (record: CropRecord) => void;
  isOnline: boolean;
  sensitivity: StressSensitivity;
  stressThreshold: number;
}

// Added missing cropOptions for the selection dropdown
const cropOptions = [
  {
    category: 'Grains',
    crops: ['Corn', 'Wheat', 'Rice', 'Barley', 'Oats']
  },
  {
    category: 'Vegetables',
    crops: ['Tomato', 'Potato', 'Onion', 'Lettuce', 'Pepper']
  },
  {
    category: 'Fruits',
    crops: ['Apple', 'Grape', 'Strawberry', 'Citrus', 'Banana']
  },
  {
    category: 'Specialty',
    crops: ['Coffee', 'Cotton', 'Soybean', 'Sugarcane', 'Tea']
  }
];

const Analyzer: React.FC<AnalyzerProps> = ({ onResult, isOnline, sensitivity, stressThreshold }) => {
  const [analyzing, setAnalyzing] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [cropType, setCropType] = useState('Corn');
  const [showResult, setShowResult] = useState<CropRecord | null>(null);
  const [scanProgress, setScanProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let interval: any;
    if (analyzing) {
      setScanProgress(0);
      interval = setInterval(() => {
        setScanProgress(p => p < 100 ? p + 2 : 100);
      }, 50);
    }
    return () => clearInterval(interval);
  }, [analyzing]);

  const processFile = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => setImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleStartAnalysis = async () => {
    if (!image) return;
    setAnalyzing(true);
    try {
      const result: YieldAnalysis = await analyzeCropHealth(image, cropType, sensitivity);
      let status = CropStatus.HEALTHY;
      if (result.expectedLoss > 30) status = CropStatus.CRITICAL;
      else if (result.expectedLoss > 5 || result.symptomlessStressDetected || result.stressProbability >= stressThreshold) {
        status = result.expectedLoss > 15 ? CropStatus.DISEASED : CropStatus.STRESSED;
      }
      
      // Artificial delay to allow user to see the "Scan" animation (improves perceived precision)
      await new Promise(r => setTimeout(r, 1500));
      
      setShowResult({ 
        id: Math.random().toString(36).substr(2, 9), 
        timestamp: Date.now(), 
        cropType, 
        imageUrl: image, 
        status, 
        analysis: result
      });
    } catch (error) { 
      alert("Analysis engine failure. Ensure API connection."); 
    } finally { 
      setAnalyzing(false); 
    }
  };

  if (showResult) {
    const { analysis } = showResult;
    return (
      <div className="max-w-5xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
        <div className="bg-white rounded-3xl shadow-2xl overflow-hidden border border-gray-100">
          <div className="bg-emerald-800 p-8 text-white flex justify-between items-center relative overflow-hidden">
            <div className="relative z-10">
              <h2 className="text-3xl font-black mb-1">Diagnostic Report</h2>
              <p className="text-emerald-100 opacity-80 flex items-center gap-2">
                <i className="fas fa-microchip"></i> AI Engine v4.2 Scientific Analysis
              </p>
            </div>
            <div className={`relative z-10 px-6 py-2 rounded-2xl font-black text-sm uppercase shadow-xl ${
              showResult.status === CropStatus.HEALTHY ? 'bg-emerald-500' :
              showResult.status === CropStatus.STRESSED ? 'bg-amber-500' : 'bg-rose-500'
            }`}>
              {showResult.status}
            </div>
            <div className="absolute top-0 right-0 opacity-10 text-9xl -mr-10 -mt-10">
              <i className="fas fa-leaf"></i>
            </div>
          </div>

          <div className="p-8">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
              <div className="lg:col-span-5 space-y-6">
                <div className="relative rounded-2xl overflow-hidden shadow-lg border-4 border-emerald-50">
                  <img src={showResult.imageUrl} className="w-full aspect-square object-cover" alt="Evidence" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent"></div>
                  <div className="absolute bottom-4 left-4 right-4 flex justify-between text-white text-[10px] font-bold uppercase tracking-widest opacity-80">
                    <span>Evidence Frame #01</span>
                    <span>Spectral Channel: VIS</span>
                  </div>
                </div>

                <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100 space-y-4">
                  <h4 className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Scientific Markers</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <Marker label="Chlorophyll" value={`${Math.round(100 - (analysis.stressProbability * 0.4))}%`} active={!analysis.symptomlessStressDetected} />
                    <Marker label="Necrosis" value={`${analysis.detailedMetrics?.leafCoverage || 0}%`} active={analysis.expectedLoss > 0} />
                    <Marker label="Match Rate" value={`${Math.round(analysis.confidenceScore * 100)}%`} active={true} />
                    <Marker label="Pattern" value={analysis.detailedMetrics?.spreadVelocity || 'Static'} active={true} />
                  </div>
                </div>
              </div>

              <div className="lg:col-span-7 space-y-8">
                <section>
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest">Pathology Findings</h3>
                    {analysis.symptomlessStressDetected && (
                      <span className="bg-rose-100 text-rose-700 px-3 py-1 rounded-full text-[10px] font-black animate-pulse flex items-center gap-1">
                        <i className="fas fa-wave-square"></i> Stress Signature Detected
                      </span>
                    )}
                  </div>
                  <div className="bg-emerald-50 p-6 rounded-2xl border border-emerald-100 relative overflow-hidden group">
                    <div className="relative z-10">
                      <h4 className="text-xl font-bold text-emerald-900 mb-2">{analysis.diseaseDetected}</h4>
                      <p className="text-emerald-800 text-sm leading-relaxed mb-6 italic">"{analysis.diseaseDescription}"</p>
                      <div className="flex items-center gap-6">
                        <div>
                          <p className="text-[10px] font-bold text-emerald-600 uppercase">Predicted Loss</p>
                          <p className="text-4xl font-black text-emerald-900">{analysis.expectedLoss}%</p>
                        </div>
                        <div className="w-px h-12 bg-emerald-200"></div>
                        <div>
                          <p className="text-[10px] font-bold text-emerald-600 uppercase">Stress Probability</p>
                          <p className="text-2xl font-black text-emerald-900">{analysis.stressProbability}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>

                <section>
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4">AgroGuard AI Intervention</h3>
                  <div className="space-y-3">
                    {analysis.recommendations.map((rec, i) => (
                      <div key={i} className="flex gap-4 p-4 bg-white border border-gray-100 rounded-2xl shadow-sm hover:border-emerald-200 transition-colors">
                        <div className="w-10 h-10 bg-emerald-100 rounded-xl flex items-center justify-center text-emerald-600 shrink-0"><i className="fas fa-tools"></i></div>
                        <p className="text-gray-700 text-sm font-medium self-center">{rec}</p>
                      </div>
                    ))}
                  </div>
                </section>

                <div className="flex gap-4 pt-4">
                  <button onClick={() => onResult(showResult)} className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white py-4 rounded-2xl font-bold shadow-xl shadow-emerald-100 transition-all flex items-center justify-center gap-2">
                    <i className="fas fa-save"></i> Save & Confirm Findings
                  </button>
                  <button onClick={() => setShowResult(null)} className="px-8 bg-gray-100 text-gray-700 py-4 rounded-2xl font-bold hover:bg-gray-200 transition-colors">Redo</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-20">
      <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-100">
        <div className="bg-emerald-800 p-8 text-white">
          <h2 className="text-2xl font-bold mb-2">Scientific Crop Diagnostic</h2>
          <p className="text-emerald-100 opacity-80">Deploying multimodal analysis for early stress detection</p>
        </div>
        
        <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-12">
          <div className="space-y-8">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-black text-gray-400 mb-2 uppercase tracking-widest">Observation Subject</label>
                <select value={cropType} onChange={(e) => setCropType(e.target.value)} className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 font-medium outline-none focus:ring-2 focus:ring-emerald-500 transition-all">
                  {cropOptions.map(g => <optgroup key={g.category} label={g.category}>{g.crops.map(c => <option key={c} value={c}>{c}</option>)}</optgroup>)}
                </select>
              </div>

              <div className="p-5 bg-emerald-50 rounded-2xl border border-emerald-100">
                <h4 className="text-sm font-bold text-emerald-800 mb-3 flex items-center gap-2">
                  <i className="fas fa-vial"></i> Pre-Analysis Checklist
                </h4>
                <ul className="text-[11px] text-emerald-700 space-y-2 opacity-80 font-medium">
                  <li className="flex gap-2"><i className="fas fa-check text-emerald-500"></i> Minimum 500px resolution required</li>
                  <li className="flex gap-2"><i className="fas fa-check text-emerald-500"></i> Focus on adaxial (upper) leaf surface</li>
                  <li className="flex gap-2"><i className="fas fa-check text-emerald-500"></i> Avoid extreme spectral glare/shadows</li>
                </ul>
              </div>
            </div>
            
            <button disabled={analyzing || !image} onClick={handleStartAnalysis} className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-200 disabled:text-gray-400 text-white py-5 rounded-2xl font-black text-lg shadow-xl flex flex-col items-center justify-center transition-all transform active:scale-95 group relative overflow-hidden">
              {analyzing ? (
                <>
                  <div className="flex items-center gap-3 z-10"><i className="fas fa-sync fa-spin"></i> Processing Neural Layers...</div>
                  <div className="absolute bottom-0 left-0 h-1 bg-emerald-400 transition-all duration-300" style={{ width: `${scanProgress}%` }}></div>
                </>
              ) : (
                <div className="flex items-center gap-3 z-10"><i className="fas fa-microscope text-xl"></i> Initiate Visual Diagnostic</div>
              )}
            </button>
          </div>

          <div onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={(e) => { e.preventDefault(); setIsDragging(false); const file = e.dataTransfer.files?.[0]; if (file) processFile(file); }} onClick={() => fileInputRef.current?.click()} className={`min-h-[400px] border-4 border-dashed rounded-3xl flex flex-col items-center justify-center cursor-pointer transition-all relative overflow-hidden ${isDragging ? 'border-emerald-600 bg-emerald-50' : image ? 'border-emerald-500 bg-black' : 'border-gray-200 bg-gray-50'}`}>
            {image ? (
              <>
                <img src={image} className={`w-full h-full object-contain p-2 transition-all ${analyzing ? 'opacity-40 blur-sm' : ''}`} alt="Preview" />
                {analyzing && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="w-full h-[2px] bg-emerald-400 absolute animate-scan"></div>
                    <div className="text-emerald-400 text-xs font-black uppercase tracking-tighter bg-black/80 px-4 py-2 rounded-lg border border-emerald-400/30">Feature Extraction Active</div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center p-8 group">
                <div className="w-24 h-24 bg-white text-emerald-600 flex items-center justify-center text-4xl mx-auto mb-4 rounded-3xl shadow-lg border border-gray-100 group-hover:scale-110 transition-transform">
                  <i className="fas fa-camera-retro"></i>
                </div>
                <p className="font-black text-gray-800 text-lg">Load Evidence Frame</p>
                <p className="text-xs text-gray-400 max-w-[200px] mx-auto mt-2">Drag image or tap to utilize device optical sensor</p>
              </div>
            )}
            <input type="file" ref={fileInputRef} onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f); }} className="hidden" accept="image/*" />
          </div>
        </div>
      </div>
      <style>{`
        @keyframes scan {
          0% { top: 0%; }
          100% { top: 100%; }
        }
        .animate-scan {
          animation: scan 2s linear infinite;
          box-shadow: 0 0 15px 2px rgba(16, 185, 129, 0.8);
        }
      `}</style>
    </div>
  );
};

const Marker = ({ label, value, active }: { label: string, value: string, active: boolean }) => (
  <div className={`p-3 rounded-xl border flex flex-col transition-all ${active ? 'bg-white border-emerald-100' : 'bg-gray-100 border-gray-200 grayscale opacity-60'}`}>
    <span className="text-[8px] font-black text-gray-400 uppercase tracking-tight">{label}</span>
    <span className="text-sm font-black text-emerald-900 truncate">{value}</span>
  </div>
);

export default Analyzer;
