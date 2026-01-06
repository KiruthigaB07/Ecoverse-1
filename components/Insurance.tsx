
import React, { useMemo } from 'react';
import { CropRecord } from '../types';

interface InsuranceProps {
  records: CropRecord[];
  threshold: number;
}

const Insurance: React.FC<InsuranceProps> = ({ records, threshold }) => {
  const insuranceAlerts = useMemo(() => {
    return records.filter(r => r.analysis.expectedLoss >= threshold);
  }, [records, threshold]);

  const generateReport = (record: CropRecord) => {
    const reportText = `
      AGROGUARD AI - INSURANCE CLAIM TRIGGER
      -------------------------------------
      Claim Ref: CLM-${record.id.toUpperCase()}
      Date: ${new Date(record.timestamp).toLocaleString()}
      Crop Type: ${record.cropType}
      
      ANALYSIS FINDINGS:
      Predicted Yield Loss: ${record.analysis.expectedLoss}%
      Detected Issue: ${record.analysis.diseaseDetected || 'Symptomless Stress Pattern'}
      Confidence Score: ${Math.round(record.analysis.confidenceScore * 100)}%
      Trigger Threshold: ${threshold}%
      
      VISUAL EVIDENCE:
      The AI system identified high-risk visual patterns including significant leaf lesions
      and discoloration that correlate with a potential yield loss exceeding ${record.analysis.expectedLoss}%.
      
      RECOMMENDATION: This record exceeds the ${threshold}% yield loss threshold. 
      Suggested for immediate visual insurance assessment and payout processing.
    `;
    
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `Insurance_Report_${record.id}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold">Yield Insurance Trigger AI</h2>
          <p className="text-gray-500">Automated visual alerts for threshold-exceeding yield losses.</p>
        </div>
        <div className="bg-rose-50 border border-rose-100 px-4 py-2 rounded-xl text-rose-700 text-sm font-bold flex items-center gap-2">
          <i className="fas fa-bell"></i>
          Threshold: {threshold}% Loss
        </div>
      </div>

      {insuranceAlerts.length === 0 ? (
        <div className="p-12 bg-white rounded-3xl border border-gray-100 text-center shadow-sm">
          <div className="w-20 h-20 bg-emerald-50 text-emerald-500 rounded-full flex items-center justify-center mx-auto mb-4 text-3xl">
            <i className="fas fa-shield-check"></i>
          </div>
          <h3 className="text-xl font-bold mb-2">Safe Harvest Forecast</h3>
          <p className="text-gray-400">No crops currently meet the yield insurance claim threshold of {threshold}% based on visual analysis.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {insuranceAlerts.map(alert => (
            <div key={alert.id} className="bg-white p-6 rounded-3xl border border-rose-100 shadow-sm relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-24 h-24 bg-rose-50 -mr-8 -mt-8 rounded-full transition-transform group-hover:scale-110"></div>
              <div className="relative z-10">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <span className="px-2 py-1 bg-rose-100 text-rose-700 text-[10px] font-bold rounded uppercase">High Risk Trigger</span>
                    <h3 className="text-lg font-bold mt-2">{alert.cropType} Claim Candidate</h3>
                  </div>
                  <i className="fas fa-exclamation-circle text-rose-500 text-xl"></i>
                </div>
                
                <div className="flex items-center gap-6 mb-6">
                  <div>
                    <p className="text-xs text-gray-400 font-bold uppercase">Predicted Loss</p>
                    <p className="text-3xl font-black text-rose-600">{alert.analysis.expectedLoss}%</p>
                  </div>
                  <div className="h-10 w-px bg-gray-100"></div>
                  <div>
                    <p className="text-xs text-gray-400 font-bold uppercase">Confidence</p>
                    <p className="text-xl font-bold">{Math.round(alert.analysis.confidenceScore * 100)}%</p>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button 
                    onClick={() => generateReport(alert)}
                    className="flex-1 bg-gray-900 text-white py-3 rounded-xl font-bold text-sm hover:bg-black transition-colors flex items-center justify-center gap-2"
                  >
                    <i className="fas fa-file-pdf"></i> Generate Report
                  </button>
                  <button className="px-4 py-3 bg-white border border-gray-200 text-gray-700 rounded-xl font-bold text-sm hover:bg-gray-50 transition-colors">
                    Notify Agent
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Insurance Info Card */}
      <div className="bg-emerald-50 p-8 rounded-3xl border border-emerald-100 flex flex-col md:flex-row items-center gap-8">
        <div className="flex-1">
          <h3 className="text-xl font-bold text-emerald-800 mb-2">Visual Claim Verification</h3>
          <p className="text-emerald-700 leading-relaxed text-sm">
            AgroGuard's insurance trigger uses multispectral visual analysis to validate claims. By comparing current crop imagery with historical disease databases, we provide objective visual evidence to accelerate your insurance payout process.
          </p>
        </div>
        <div className="w-full md:w-1/3 grid grid-cols-2 gap-4">
          <div className="bg-white p-3 rounded-2xl text-center">
            <p className="text-[10px] font-bold text-gray-400 uppercase">Verification</p>
            <p className="text-lg font-bold">Instant</p>
          </div>
          <div className="bg-white p-3 rounded-2xl text-center">
            <p className="text-[10px] font-bold text-gray-400 uppercase">Payout Rate</p>
            <p className="text-lg font-bold">98%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Insurance;
