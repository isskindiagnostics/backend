'use strict';
var __importDefault =
  (this && this.__importDefault) ||
  function (mod) {
    return mod && mod.__esModule ? mod : { default: mod };
  };
Object.defineProperty(exports, '__esModule', { value: true });
const bullmq_1 = require('bullmq');
const queue_1 = require('./queue');
const axios_1 = __importDefault(require('axios'));
const app_1 = require('firebase-admin/app');
const firestore_1 = require('firebase-admin/firestore');
const dotenv_1 = __importDefault(require('dotenv'));
dotenv_1.default.config();
// Inicializa Firebase
(0, app_1.initializeApp)({
  credential: (0, app_1.cert)(JSON.parse(process.env.FIREBASE_CREDENTIAL)),
});
const db = (0, firestore_1.getFirestore)();
// Cria o worker que escuta jobs na fila
const worker = new bullmq_1.Worker(
  'image-analysis',
  async (job) => {
    const { jobId, userId, image } = job.data;
    const jobRef = db
      .collection('users')
      .doc(userId)
      .collection('jobs')
      .doc(jobId);
    // Marca como "processing"
    await jobRef.update({ status: 'processing' });
    try {
      // Envia imagem para FastAPI (modelo YOLO)
      const response = await axios_1.default.post(
        'http://python-backend:8000/predict/',
        Buffer.from(image, 'base64'),
        {
          headers: { 'Content-Type': 'application/octet-stream' },
        }
      );
      // Salva o resultado no Firebase
      await jobRef.update({
        status: 'done',
        result: response.data,
        completedAt: new Date(),
      });
    } catch (error) {
      let errorMsg = 'Unknown error';
      if (error instanceof Error) {
        errorMsg = error.message;
      } else if (typeof error === 'string') {
        errorMsg = error;
      }
      await jobRef.update({
        status: 'error',
        error: errorMsg,
        failedAt: new Date(),
      });
    }
  },
  { connection: queue_1.connection, concurrency: 4 }
);
console.log('Worker is running...');
