'use strict';
var __importDefault =
  (this && this.__importDefault) ||
  function (mod) {
    return mod && mod.__esModule ? mod : { default: mod };
  };
Object.defineProperty(exports, '__esModule', { value: true });
const express_1 = __importDefault(require('express'));
const multer_1 = __importDefault(require('multer'));
const queue_1 = require('../queue');
const uuid_1 = require('uuid');
const app_1 = require('firebase-admin/app');
const firestore_1 = require('firebase-admin/firestore');
const dotenv_1 = __importDefault(require('dotenv'));
dotenv_1.default.config();
// Inicia o middleware para upload de imagem
const upload = (0, multer_1.default)({
  storage: multer_1.default.memoryStorage(),
});
const app = (0, express_1.default)();
const PORT = 3001;
// Inicializa Firebase Admin SDK com a variável de ambiente
(0, app_1.initializeApp)({
  credential: (0, app_1.cert)(JSON.parse(process.env.FIREBASE_CREDENTIAL)),
});
const db = (0, firestore_1.getFirestore)();
// Rota POST para enviar imagem
app.post('/analyze', upload.single('image'), async (req, res) => {
  const userId = req.body.userId;
  if (!req.file || !userId)
    return res.status(400).send('Missing image or userId');
  const jobId = (0, uuid_1.v4)();
  // Salva job na subcoleção de usuários: users/{userId}/jobs/{jobId}
  await db.collection('users').doc(userId).collection('jobs').doc(jobId).set({
    status: 'pending',
    createdAt: new Date(),
  });
  // Adiciona job na fila do Redis
  await queue_1.analysisQueue.add('analyze-image', {
    jobId,
    userId,
    image: req.file.buffer.toString('base64'),
  });
  // Retorna o ID do job para o frontend acompanhar o status
  res.json({ jobId });
});
app.listen(PORT, () => console.log(`API running on port ${PORT}`));
