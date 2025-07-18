import { initializeApp, cert, App } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { getStorage } from 'firebase-admin/storage';
import dotenv from 'dotenv';

dotenv.config();

const firebaseConfig = JSON.parse(
  Buffer.from(process.env.FIREBASE_CREDENTIAL!, 'base64').toString('utf8')
);

const app: App = initializeApp({
  credential: cert(firebaseConfig),
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
});

export const db = getFirestore(app);
export const storage = getStorage(app);
