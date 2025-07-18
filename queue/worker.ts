import { Worker } from 'bullmq';
import FormData from 'form-data';
import { connection } from './queue';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { db, storage } from './firebase/config';

const bucket = storage.bucket();

async function getPythonRAM(): Promise<number> {
  try {
    const response = await axios.get('http://python-backend:8000/ram');
    return response.data.ram; // em MB
  } catch (err) {
    console.warn('Unable to obtain RAM from the backend:', err);
    return 0;
  }
}

// Cria o worker que escuta jobs na fila
const worker = new Worker(
  'image-analysis',
  async (job) => {
    const { jobId, userId, image, mimeType } = job.data;

    const jobRef = db
      .collection('users')
      .doc(userId)
      .collection('jobs')
      .doc(jobId);

    // Marca como "processing"
    await jobRef.update({ status: 'processing' });

    try {
      const buffer = Buffer.from(image, 'base64');

      // Define extensão com base no mimeType (jpeg/jpg -> jpg, png -> png)
      const ext = mimeType === 'image/png' ? 'png' : 'jpg';
      const fileName = `users/${userId}/jobs/${jobId}/image.${ext}`;
      const file = bucket.file(fileName);

      // Salva imagem no Firebase Storage
      const token = uuidv4();
      await file.save(buffer, {
        contentType: mimeType,
        metadata: {
          metadata: {
            firebaseStorageDownloadTokens: token,
          },
        },
      });

      // Gera URL pública
      const imageUrl = `https://firebasestorage.googleapis.com/v0/b/${bucket.name}/o/${encodeURIComponent(fileName)}?alt=media&token=${token}`;

      // Cria FormData e adiciona a imagem como campo `file`
      const form = new FormData();
      form.append('file', buffer, {
        filename: `image.${ext}`,
        contentType: mimeType,
      });

      // Envia imagem para FastAPI (modelo YOLO)
      const response = await axios.post(
        'http://python-backend:8000/predict/',
        form,
        {
          headers: form.getHeaders(),
        }
      );

      const MAX_RAM_MB = 1500;
      let currentRAM = await getPythonRAM();

      if (currentRAM > MAX_RAM_MB) {
        console.log(`[THROTTLE] High RAM (${currentRAM} MB). Waiting...`);
        await new Promise((resolve) => setTimeout(resolve, 5000));
        throw new Error('Very high RAM - automatic retry will be attempted');
      }

      // Salva o resultado no Firebase
      await jobRef.update({
        status: 'done',
        result: response.data,
        completedAt: new Date(),
        imageUrl,
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
  { connection, concurrency: 6 }
);

console.log('Worker is running...');
