import { GoogleGenAI, Type } from "@google/genai";
import { YieldAnalysis, StressSensitivity, CropRecord, CropStatus } from "./types";
import { dbService } from "./db";

// Resolve API key safely for both dev/build and browser runtime.
// Prefer `import.meta.env.VITE_API_KEY` (Vite), fallback to process.env when available.
const API_KEY: string | undefined = (typeof process !== 'undefined' && (process as any).env && (process as any).env.API_KEY)
  || (typeof import.meta !== 'undefined' && (import.meta as any).env && (import.meta as any).env.VITE_API_KEY)
  || undefined;

const ai = new GoogleGenAI({ apiKey: API_KEY });

const MAX_SYNC_ATTEMPTS = 3;
let isSyncActive = false;

/**
 * Enhanced Scientific Schema for Gemini v4.2
 */
const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    expectedLoss: { type: Type.NUMBER, description: "Predicted yield loss percentage (0-100)." },
    confidenceScore: { type: Type.NUMBER, description: "Statistical confidence in diagnostic markers." },
    riskLevel: { type: Type.STRING, description: "Low, Medium, or High" },
    recommendations: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "Actionable agronomical steps."
    },
    diseaseDetected: { type: Type.STRING, description: "Scientific or common name of pathogen." },
    diseaseDescription: { type: Type.STRING, description: "Technical reasoning focusing on visual morphology." },
    similarityScore: { type: Type.NUMBER, description: "Pattern match against historical Disease Twin database." },
    symptomlessStressDetected: { type: Type.BOOLEAN, description: "Detection of pre-necrotic physiological shifts." },
    stressProbability: { type: Type.NUMBER, description: "Calculated stress probability (0-100)." },
    treatmentUrgency: { type: Type.STRING },
    detailedMetrics: {
      type: Type.OBJECT,
      properties: {
        leafCoverage: { type: Type.NUMBER, description: "Percent of surface area showing active symptoms." },
        spreadVelocity: { type: Type.STRING, description: "Static, Slow, Moderate, or Aggressive." },
        climateRiskFactor: { type: Type.NUMBER, description: "Sensitivity to current environmental variance." }
      },
      required: ["leafCoverage", "spreadVelocity", "climateRiskFactor"]
    }
  },
  required: ["expectedLoss", "confidenceScore", "riskLevel", "recommendations", "symptomlessStressDetected", "stressProbability", "treatmentUrgency", "diseaseDetected", "diseaseDescription"]
};

interface LocalVisualFeatures {
  greenness: number;
  variance: number;
  necroticDensity: number;
  redness: number;
  edgeDensity: number;
  zonalIntegrity: number; // Ratio of healthy center vs periphery
}

interface DiseaseProfile {
  name: string;
  weights: Partial<LocalVisualFeatures>;
  description: string;
  recommendations: string[];
  baseLossModifier: number;
}

/**
 * Knowledge Base for Local Inference Model
 */
const DISEASE_PROFILES: DiseaseProfile[] = [
  {
    name: "Leaf Rust (Puccinia spp.)",
    weights: { redness: 0.8, edgeDensity: 0.6, variance: 0.4 },
    description: "Detected characteristic reddish-orange fungal pustules causing high localized spectral variance.",
    recommendations: ["Apply triazole fungicide", "Reduce overhead irrigation", "Monitor spread to adjacent rows"],
    baseLossModifier: 1.2
  },
  {
    name: "Late Blight (Phytophthora infestans)",
    weights: { necroticDensity: 0.9, greenness: -0.8, variance: 0.7 },
    description: "Widespread necrotic lesions with high variance. Typical of aggressive oomycete infection.",
    recommendations: ["Immediate copper-based spray", "Remove heavily infected plant matter", "Check for stem cankers"],
    baseLossModifier: 2.0
  },
  {
    name: "Powdery Mildew",
    weights: { variance: 0.8, edgeDensity: 0.5, greenness: -0.2 },
    description: "Surface-level white mycelial growth creating high textural complexity and edge counts.",
    recommendations: ["Increase airflow in canopy", "Apply sulfur-based powders", "Check relative humidity levels"],
    baseLossModifier: 0.8
  },
  {
    name: "Nitrogen/Iron Chlorosis",
    weights: { greenness: -0.9, zonalIntegrity: -0.7, necroticDensity: -0.5 },
    description: "Broad-spectrum chlorophyll loss starting from leaf margins. Pattern suggests physiological deficiency.",
    recommendations: ["Check soil pH (likely >7.0)", "Apply chelated iron/nitrogen", "Review drainage efficiency"],
    baseLossModifier: 0.6
  },
  {
    name: "Bacterial Leaf Spot",
    weights: { edgeDensity: 0.9, necroticDensity: 0.5, variance: 0.5 },
    description: "Small, angular necrotic spots with sharp boundaries (high edge density).",
    recommendations: ["Avoid handling wet foliage", "Apply fixed copper bactericide", "Sanitize equipment between sectors"],
    baseLossModifier: 1.1
  }
];

const extractVisualFeatures = async (imageData: string): Promise<LocalVisualFeatures> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return resolve({ greenness: 0.5, variance: 0.1, necroticDensity: 0, redness: 0.1, edgeDensity: 0.1, zonalIntegrity: 0.8 });
      
      const size = 128;
      canvas.width = size;
      canvas.height = size;
      ctx.drawImage(img, 0, 0, size, size);
      
      const pixels = ctx.getImageData(0, 0, size, size).data;
      let green = 0, dark = 0, red = 0, edges = 0, totalVar = 0;
      let centerGreen = 0, edgeGreen = 0;
      const pixelCount = size * size;
      const centerLimit = size * 0.25;
      const centerMax = size * 0.75;
      
      for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
        const pixelIdx = i / 4;
        const x = pixelIdx % size;
        const y = Math.floor(pixelIdx / size);
        const isCenter = x > centerLimit && x < centerMax && y > centerLimit && y < centerMax;

        const isGreen = g > r * 1.1 && g > b * 1.1;
        if (isGreen) {
          green++;
          if (isCenter) centerGreen++; else edgeGreen++;
        }
        if (r > g * 1.3 && r > b * 1.2) red++; 
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if (lum < 70) dark++;
        
        totalVar += Math.abs(r - g) + Math.abs(g - b);
      }

      for (let i = 0; i < pixels.length - 8; i += 8) {
        if (Math.abs(pixels[i] - pixels[i+4]) > 40) edges++;
      }

      resolve({
        greenness: green / pixelCount,
        variance: Math.min(1, totalVar / (pixelCount * 200)),
        necroticDensity: dark / pixelCount,
        redness: red / pixelCount,
        edgeDensity: Math.min(1, edges / (pixelCount * 0.4)),
        zonalIntegrity: edgeGreen > 0 ? centerGreen / edgeGreen : 1
      });
    };
    img.src = imageData;
  });
};

const runLocalInference = async (
  imageData: string | null, 
  cropType: string, 
  sensitivity: StressSensitivity
): Promise<YieldAnalysis> => {
  const visual = imageData 
    ? await extractVisualFeatures(imageData) 
    : { greenness: 0.8, variance: 0.1, necroticDensity: 0.05, redness: 0.02, edgeDensity: 0.1, zonalIntegrity: 0.9 };
  
  const sMod = sensitivity === 'Aggressive' ? 1.4 : (sensitivity === 'High' ? 1.2 : 1.0);
  
  // Calculate Profile Scores
  let bestProfile: DiseaseProfile | null = null;
  let maxScore = -1;

  for (const profile of DISEASE_PROFILES) {
    let score = 0;
    let weightCount = 0;
    
    for (const [key, weight] of Object.entries(profile.weights)) {
      const featVal = visual[key as keyof LocalVisualFeatures];
      // Positive weight: value should be high. Negative weight: value should be low (1 - value).
      score += weight > 0 ? featVal * weight : (1 - featVal) * Math.abs(weight);
      weightCount += Math.abs(weight);
    }
    
    const finalScore = score / weightCount;
    if (finalScore > maxScore) {
      maxScore = finalScore;
      bestProfile = profile;
    }
  }

  // Base Stress Heuristic for general severity
  const stressHeuristic = ((1 - visual.greenness) * 0.4 + visual.necroticDensity * 0.4 + (1 - visual.zonalIntegrity) * 0.2) * sMod;
  const isSymptomless = stressHeuristic > 0.25 && visual.necroticDensity < 0.1 && visual.edgeDensity < 0.2;
  const confidence = Math.min(0.92, 0.5 + (maxScore * 0.4));

  // Determine disease based on score threshold
  const matchedDisease = maxScore > 0.45 && bestProfile ? bestProfile : null;

  return {
    expectedLoss: Math.round(stressHeuristic * 35 * (matchedDisease ? matchedDisease.baseLossModifier : 1)),
    confidenceScore: confidence,
    riskLevel: stressHeuristic > 0.5 ? "High" : (stressHeuristic > 0.2 ? "Medium" : "Low"),
    recommendations: matchedDisease ? matchedDisease.recommendations : (isSymptomless 
      ? ["Apply micronutrient foliar spray", "Verify soil moisture", "Conduct sap test"]
      : ["Isolate sector", "Apply organic fungicide", "Monitor spread"]),
    diseaseDetected: matchedDisease ? matchedDisease.name : (isSymptomless ? "Physiological Stress" : "Unknown Pathogen"),
    diseaseDescription: matchedDisease 
      ? `Local Diagnostic Engine Match: ${bestProfile?.description} (Match Confidence: ${Math.round(maxScore * 100)}%)` 
      : `Heuristic Analysis: Zonal integrity at ${Math.round(visual.zonalIntegrity * 100)}%. ${isSymptomless ? 'Detected spectral shifts indicating latent stress.' : 'Detected atypical morphological lesions.'}`,
    similarityScore: maxScore,
    symptomlessStressDetected: isSymptomless,
    stressProbability: Math.round(stressHeuristic * 100),
    treatmentUrgency: stressHeuristic > 0.4 ? 'Immediate' : (stressHeuristic > 0.15 ? 'Within 48h' : 'Monitoring'),
    detailedMetrics: { 
      leafCoverage: Math.round(visual.necroticDensity * 160), 
      spreadVelocity: visual.edgeDensity > 0.35 ? 'Aggressive' : (visual.edgeDensity > 0.15 ? 'Moderate' : 'Slow'), 
      climateRiskFactor: 0.5 
    }
  };
};

export const analyzeCropHealth = async (
  imageData: string | null, 
  cropType: string, 
  sensitivity: StressSensitivity = 'Standard'
): Promise<YieldAnalysis> => {
  if (!navigator.onLine || !API_KEY) return runLocalInference(imageData, cropType, sensitivity);
  
  const prompt = `Act as a Senior Plant Pathologist and Computer Vision expert. 
  Target: ${cropType}. Sensitivity Profile: ${sensitivity}.
  
  CORE MISSION: 
  1. Perform "Disease Twin" mapping: Identify if patterns match known pathogens (e.g. Early Blight, Rust).
  2. Detect "Symptomless Stress": Look for subtle chlorosis, wilting, or spectral shifts.
  3. Predict Yield Impact: Estimate loss based on visible foliar damage vs expected harvest stages.
  
  Output strictly in JSON format according to the provided schema.`;

  try {
    const parts: any[] = [{ text: prompt }];
    if (imageData) parts.push({ inlineData: { mimeType: "image/jpeg", data: imageData.split(",")[1] } });
    
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [{ parts }],
      config: { responseMimeType: "application/json", responseSchema: ANALYSIS_SCHEMA, temperature: 0.1 }
    });
    
    return JSON.parse(response.text || "{}");
  } catch (error) {
    console.error("Cloud Analytics Engine Offline/Failed:", error);
    return runLocalInference(imageData, cropType, sensitivity);
  }
};

export interface SyncProgress {
  current: number;
  total: number;
  status: 'idle' | 'syncing' | 'completed' | 'failed';
  currentCrop?: string;
}

export const syncOfflineRecords = async (onProgress?: (p: SyncProgress) => void) => {
  if (isSyncActive || !navigator.onLine || !API_KEY) return;
  isSyncActive = true;
  const pending = dbService.getPendingRecords();
  if (pending.length === 0) { isSyncActive = false; return; }

  const settings = dbService.getSettings();
  try {
    for (let i = 0; i < pending.length; i++) {
      const record = pending[i];
      if ((record.syncAttempts || 0) >= MAX_SYNC_ATTEMPTS) continue;
      if (onProgress) onProgress({ current: i + 1, total: pending.length, status: 'syncing', currentCrop: record.cropType });

      try {
        const cloudResult = await analyzeCropHealth(record.imageUrl || null, record.cropType, settings.stressSensitivity);
        let status = CropStatus.HEALTHY;
        if (cloudResult.expectedLoss > settings.insuranceThreshold) status = CropStatus.CRITICAL;
        else if (cloudResult.expectedLoss > 5 || cloudResult.symptomlessStressDetected) {
          status = cloudResult.expectedLoss > 15 ? CropStatus.DISEASED : CropStatus.STRESSED;
        }

        dbService.updateRecord(record.id, { 
          analysis: cloudResult, 
          status, 
          isPendingSync: false,
          syncAttempts: (record.syncAttempts || 0) + 1 
        });
      } catch (err) {
        dbService.updateRecord(record.id, { syncAttempts: (record.syncAttempts || 0) + 1 });
      }
    }
    if (onProgress) onProgress({ current: pending.length, total: pending.length, status: 'completed' });
  } finally {
    isSyncActive = false;
  }
};
