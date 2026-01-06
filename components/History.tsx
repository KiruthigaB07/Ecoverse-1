import React, { useState } from 'react';
import { CropRecord, CropStatus } from '../types';
import { dbService } from '../db';

interface HistoryProps {
  records: CropRecord[];
  onUpdate: () => void;
}

const History: React.FC<HistoryProps> = ({ records, onUpdate }) => {
  const [selectedRecord, setSelectedRecord] = useState<CropRecord | null>(null);

  const handleDelete = (id: string) => {
    if (confirm('Delete this record?')) {
      dbService.deleteRecord(id);
      onUpdate();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-emerald-900">Crop Health Timeline</h2>
        <p className="text-sm text-gray-500 font-medium">{records.length} Records</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {records.map((record) => (
          <div key={record.id} className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-all relative group">
            <div className="relative h-48 bg-gray-100">
              {record.imageUrl && <img src={record.imageUrl} className="w-full h-full object-cover" alt="Crop" />}
              <div className={`absolute top-4 right-4 px-3 py-1 rounded-full text-[10px] font-black shadow-sm ${
                record.status === CropStatus.HEALTHY ? 'bg-emerald-500 text-white' :
                record.status === CropStatus.STRESSED ? 'bg-amber-500 text-white' : 'bg-rose-500 text-white'
              }`}>
                {record.status}
              </div>
            </div>
            <div className="p-5">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-bold text-lg text-emerald-950">{record.cropType}</h3>
                  <p className="text-xs text-gray-500">{new Date(record.timestamp).toLocaleDateString()}</p>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => setSelectedRecord(record)} className="w-8 h-8 flex items-center justify-center text-emerald-600 hover:bg-emerald-50 rounded-lg"><i className="fas fa-eye"></i></button>
                  <button onClick={() => handleDelete(record.id)} className="w-8 h-8 flex items-center justify-center text-rose-600 hover:bg-rose-50 rounded-lg"><i className="fas fa-trash"></i></button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-gray-50 p-2 rounded-lg flex items-center justify-between">
                  <span className="text-[9px] font-bold text-gray-400 uppercase">Yield Loss</span>
                  <span className="text-xs font-black text-rose-600">{record.analysis.expectedLoss}%</span>
                </div>
                <div className="bg-gray-50 p-2 rounded-lg flex items-center justify-between">
                  <span className="text-[9px] font-bold text-gray-400 uppercase">Stress Prob</span>
                  <span className="text-xs font-black text-amber-600">{record.analysis.stressProbability}%</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {selectedRecord && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white w-full max-w-4xl rounded-3xl overflow-hidden shadow-2xl animate-in zoom-in duration-300 max-h-[90vh] flex flex-col">
            <div className="bg-emerald-800 p-6 text-white flex justify-between items-center shrink-0">
              <h2 className="text-xl font-bold">Diagnostic Report: {selectedRecord.cropType}</h2>
              <button onClick={() => setSelectedRecord(null)} className="w-10 h-10 rounded-full hover:bg-white/10 flex items-center justify-center"><i className="fas fa-times"></i></button>
            </div>
            <div className="overflow-y-auto p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="rounded-2xl overflow-hidden shadow-lg border border-gray-100">
                    {selectedRecord.imageUrl && <img src={selectedRecord.imageUrl} className="w-full aspect-square object-cover" alt="Detail" />}
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <h4 className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-3">AI Diagnostic Summary</h4>
                    <div className="bg-gray-50 p-5 rounded-2xl space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">Risk Level</span>
                        <span className={`px-3 py-1 rounded-full text-[10px] font-black ${
                          selectedRecord.analysis.riskLevel === 'High' ? 'bg-rose-500 text-white' : 'bg-emerald-500 text-white'
                        }`}>{selectedRecord.analysis.riskLevel}</span>
                      </div>
                      <p className="text-sm text-gray-800 leading-relaxed font-medium">{selectedRecord.analysis.diseaseDescription}</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-3">Intervention Protocol</h4>
                    <div className="space-y-2">
                      {selectedRecord.analysis.recommendations.map((rec, i) => (
                        <div key={i} className="bg-white border border-gray-100 p-3 rounded-xl flex gap-3 text-sm text-gray-600 shadow-sm">
                          <i className="fas fa-check-circle text-emerald-500 mt-1"></i>
                          <span className="font-medium">{rec}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                     <div className="bg-rose-50 p-4 rounded-2xl border border-rose-100 text-center">
                        <p className="text-[9px] font-black text-rose-400 uppercase">Yield Loss</p>
                        <p className="text-2xl font-black text-rose-700">{selectedRecord.analysis.expectedLoss}%</p>
                     </div>
                     <div className="bg-amber-50 p-4 rounded-2xl border border-amber-100 text-center">
                        <p className="text-[9px] font-black text-amber-400 uppercase">Stress Prob</p>
                        <p className="text-2xl font-black text-amber-700">{selectedRecord.analysis.stressProbability}%</p>
                     </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;