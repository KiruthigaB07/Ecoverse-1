
import { CropRecord, UserSettings } from './types';

const DB_KEY = 'agroguard_records';
const SETTINGS_KEY = 'agroguard_settings';

const DEFAULT_SETTINGS: UserSettings = {
  stressSensitivity: 'Standard',
  stressThreshold: 20, // 20% probability triggers a stress warning
  insuranceThreshold: 25,
  autoSync: true
};

export const dbService = {
  getRecords: (): CropRecord[] => {
    const data = localStorage.getItem(DB_KEY);
    return data ? JSON.parse(data) : [];
  },

  getPendingRecords: (): CropRecord[] => {
    return dbService.getRecords().filter(r => r.isPendingSync);
  },

  saveRecord: (record: CropRecord) => {
    const records = dbService.getRecords();
    records.unshift(record);
    localStorage.setItem(DB_KEY, JSON.stringify(records));
  },

  updateRecord: (id: string, updatedData: Partial<CropRecord>) => {
    const records = dbService.getRecords().map(r => 
      r.id === id ? { ...r, ...updatedData } : r
    );
    localStorage.setItem(DB_KEY, JSON.stringify(records));
  },

  deleteRecord: (id: string) => {
    const records = dbService.getRecords().filter(r => r.id !== id);
    localStorage.setItem(DB_KEY, JSON.stringify(records));
  },

  updateFeedback: (id: string, feedback: string) => {
    const records = dbService.getRecords().map(r => 
      r.id === id ? { ...r, feedback } : r
    );
    localStorage.setItem(DB_KEY, JSON.stringify(records));
  },

  getSettings: (): UserSettings => {
    const data = localStorage.getItem(SETTINGS_KEY);
    return data ? JSON.parse(data) : DEFAULT_SETTINGS;
  },

  saveSettings: (settings: UserSettings) => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }
};
