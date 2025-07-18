import { initializeApp, cert, App } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { getStorage } from 'firebase-admin/storage';
import dotenv from 'dotenv';

dotenv.config();

const app: App = initializeApp({
  credential: cert(JSON.parse(process.env.FIREBASE_CREDENTIAL as string)),
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
});

export const db = getFirestore(app);
export const storage = getStorage(app);
