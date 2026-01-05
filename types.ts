
export enum CropStatus {
  HEALTHY = 'Healthy',
  STRESSED = 'Stressed',
  DISEASED = 'Diseased',
  CRITICAL = 'Critical'
}

export type StressSensitivity = 'Low' | 'Standard' | 'High' | 'Aggressive';

export interface UserSettings {
  stressSensitivity: StressSensitivity;
  stressThreshold: number; // 0-100 threshold for symptomless stress probability
  insuranceThreshold: number;
  autoSync: boolean;
}

export interface DetailedMetrics {
  leafCoverage: number; // % of plant affected
  spreadVelocity: 'Static' | 'Slow' | 'Moderate' | 'Aggressive';
  climateRiskFactor: number; // 0-1 based on humidity/temp correlation
}

export interface YieldAnalysis {
  expectedLoss: number;
  confidenceScore: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  recommendations: string[];
  diseaseDetected?: string;
  diseaseDescription?: string;
  similarityScore?: number;
  symptomlessStressDetected: boolean;
  stressProbability: number; // Probability of early stress (0-100)
  treatmentUrgency: 'Immediate' | 'Within 48h' | 'Monitoring';
  detailedMetrics?: DetailedMetrics;
}

export interface CropRecord {
  id: string;
  timestamp: number;
  cropType: string;
  imageUrl?: string;
  status: CropStatus;
  analysis: YieldAnalysis;
  feedback?: string;
  isPendingSync?: boolean; 
  syncAttempts?: number;
  lastSyncError?: string;
}

export interface DashboardStats {
  totalAnalyzed: number;
  healthyPercentage: number;
  averageYieldLoss: number;
  activeAlerts: number;
}
