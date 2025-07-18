import express from 'express';
import multer from 'multer';
import { analysisQueue } from '../queue';
import { v4 as uuidv4 } from 'uuid';
import dotenv from 'dotenv';
import { db } from '../firebase/config';

dotenv.config();
console.log('FIREBASE_STORAGE_BUCKET:', process.env.FIREBASE_STORAGE_BUCKET);

// Inicia o middleware para upload de imagem
const upload = multer({ storage: multer.memoryStorage() });

const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];

const app = express();
const PORT = 3001;

// Rota POST para enviar imagem
app.post('/analyze', upload.single('image'), async (req, res) => {
  const userId = req.body.userId;
  if (!req.file || !userId)
    return res.status(400).send('Missing image or userId');

  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    return res
      .status(400)
      .send('Invalid image type. Only JPEG and PNG are allowed.');
  }

  const jobId = uuidv4();

  // Salva job na subcoleção de usuários: users/{userId}/jobs/{jobId}
  await db.collection('users').doc(userId).collection('jobs').doc(jobId).set({
    status: 'pending',
    createdAt: new Date(),
  });

  // Adiciona job na fila do Redis
  await analysisQueue.add(
    'analyze-image',
    {
      jobId,
      userId,
      image: req.file.buffer.toString('base64'),
      mimeType: req.file.mimetype,
    },
    {
      attempts: 5,
      backoff: {
        type: 'exponential',
        delay: 5000,
      },
    }
  );

  // Retorna o ID do job para o frontend acompanhar o status
  res.json({ jobId });
});

app.listen(PORT, () => console.log(`API running on port ${PORT}`));
