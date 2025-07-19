import { Worker } from 'bullmq'; // classe para criar um “worker” que vai processar jobs da fila.
import FormData from 'form-data'; // FormData: para enviar arquivos via HTTP POST.
import { connection } from './queue'; // connection: configuração do Redis para conectar a fila.
import axios from 'axios'; // axios: biblioteca para fazer requisições HTTP.
import { v4 as uuidv4 } from 'uuid'; // uuidv4: para gerar IDs únicos.
import { db, storage } from './firebase/config'; // db, storage: acesso ao Firebase Firestore e Storage.

const bucket = storage.bucket(); // pega o bucket do Firebase Storage para salvar imagens.

// Função para perguntar ao backend Python quanta RAM ele está usando, via requisição HTTP GET.
// Se der erro, retorna 0 e avisa no console.
async function getPythonRAM(): Promise<number> {
  try {
    const response = await axios.get('http://python-backend:8000/ram');
    return response.data.ram; // em MB
  } catch (err) {
    console.warn('Unable to obtain RAM from the backend:', err);
    return 0;
  }
}

// Cria o worker que fica escutando a fila image-analysis no Redis.
// Quando chega um job (trabalho), ele recebe os dados (job.data): id do job, id do usuário, imagem e tipo da imagem.
const worker = new Worker(
  'image-analysis',
  async (job) => {
    const { jobId, userId, image, mimeType } = job.data;

    // Cria uma referência ao documento do job no Firebase, para atualizar o status.
    const jobRef = db
      .collection('users')
      .doc(userId)
      .collection('jobs')
      .doc(jobId);

    // Atualiza o status para processing (está processando).
    await jobRef.update({ status: 'processing' });

    try {
      // Converte a imagem, que veio codificada em base64, para um Buffer (formato de arquivo em memória).
      const buffer = Buffer.from(image, 'base64');

      const ext = mimeType === 'image/png' ? 'png' : 'jpg'; // Decide a extensão da imagem (png ou jpg) para salvar o arquivo.
      const fileName = `users/${userId}/jobs/${jobId}/image.${ext}`; // Define o caminho do arquivo no Firebase Storage.
      const file = bucket.file(fileName); // Cria um token único para permitir acesso público à imagem.

      // Salva o arquivo no Firebase Storage com o conteúdo, tipo e token.
      const token = uuidv4();
      await file.save(buffer, {
        contentType: mimeType,
        metadata: {
          metadata: {
            firebaseStorageDownloadTokens: token,
          },
        },
      });

      // Monta a URL pública para a imagem salva.
      const imageUrl = `https://firebasestorage.googleapis.com/v0/b/${bucket.name}/o/${encodeURIComponent(fileName)}?alt=media&token=${token}`;

      // Cria um FormData para enviar a imagem para o backend Python.
      const form = new FormData();

      // Adiciona o arquivo no campo file da requisição.
      form.append('file', buffer, {
        filename: `image.${ext}`,
        contentType: mimeType,
      });

      // Envia a imagem para o FastAPI na rota /predict/ para rodar a análise.
      const response = await axios.post(
        'http://python-backend:8000/predict/',
        form,
        {
          headers: form.getHeaders(),
        }
      );

      const MAX_RAM_MB = 1500;
      let currentRAM = await getPythonRAM(); // Consulta a RAM usada pelo backend Python (função que vimos antes).

      // Se a RAM ultrapassar 1500 MB (limite seguro), ele espera 5 segundos e lança um erro para que o BullMQ tente novamente depois (retry automático).
      if (currentRAM > MAX_RAM_MB) {
        console.log(`[THROTTLE] High RAM (${currentRAM} MB). Waiting...`);
        await new Promise((resolve) => setTimeout(resolve, 5000));
        throw new Error('Very high RAM - automatic retry will be attempted');
      }

      // Se tudo deu certo, atualiza o job no Firebase com o resultado da predição, a URL da imagem e marca como done.
      await jobRef.update({
        status: 'done',
        result: response.data,
        completedAt: new Date(),
        imageUrl,
      });
    } catch (error) {
      // Se deu erro, captura a mensagem de erro e salva no Firebase com status error.
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
  { connection, concurrency: 3 } // O worker é configurado para rodar até 3 jobs em paralelo.
);
// Imprime no console que o worker está rodando.
console.log('Worker is running...');
