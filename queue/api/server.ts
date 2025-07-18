import express from 'express'; // Importa express (framework para servidor web).
import multer from 'multer'; // Importa multer para processar uploads de arquivos (imagens).
import { analysisQueue } from '../queue'; // Importa a fila analysisQueue.
import { v4 as uuidv4 } from 'uuid'; // Importa uuidv4 para criar ids únicos para os jobs.
import dotenv from 'dotenv'; // Importa as configurações do Firebase.
import { db } from '../firebase/config'; // Carrega variáveis de ambiente e mostra no console o nome do bucket do Firebase.

dotenv.config();

// Configura o multer para armazenar imagens na memória (não salva no disco).
const upload = multer({ storage: multer.memoryStorage() });

// Define tipos permitidos de imagens (jpeg e png).
const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];

// Cria o servidor express e define a porta 80.
const app = express();
const PORT = 80;

// Cria a rota POST /analyze para receber uma imagem.
// Usa multer para pegar o arquivo enviado no campo image.
app.post('/analyze', upload.single('image'), async (req, res) => {
  const userId = req.body.userId;

  // Verifica se veio a imagem e o id do usuário, se não, responde com erro 400.
  if (!req.file || !userId)
    return res.status(400).send('Missing image or userId');

  // Verifica se o tipo da imagem é permitido, se não, responde com erro.
  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    return res
      .status(400)
      .send('Invalid image type. Only JPEG and PNG are allowed.');
  }

  // Gera um ID único para o job.
  const jobId = uuidv4();

  // Salva o job no Firebase com status pending.
  await db.collection('users').doc(userId).collection('jobs').doc(jobId).set({
    status: 'pending',
    createdAt: new Date(),
  });

  // Adiciona o job na fila do Redis com as informações: jobId, userId, imagem (convertida para base64), e mimeType.
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

  // Retorna para o frontend o ID do job, para ele poder acompanhar o status.
  res.json({ jobId });
});

// Inicia o servidor na porta configurada e mostra no console.
app.listen(PORT, () => console.log(`API running on port ${PORT}`));
