import { Queue } from 'bullmq'; // Importa o Queue para criar filas de trabalho.
import dotenv from 'dotenv'; // Importa dotenv para carregar variáveis de ambiente do arquivo .env.

dotenv.config();

// Define a conexão com o Redis: o host vem do .env ou assume redis (nome do container Redis no Docker).
export const connection = {
  host: process.env.REDIS_HOST || 'redis',
  port: 6379,
};

// Cria e exporta a fila chamada image-analysis usando essa conexão.
export const analysisQueue = new Queue('image-analysis', {
  connection,
  defaultJobOptions: {
    removeOnComplete: {
      age: 3600, // remove jobs 1 hour depois de ser completado
      count: 1000, // ou mantem apenas os ultimos 1000
    },
    removeOnFail: {
      age: 86400, // remove jobs que falharam depois de 24h
      count: 500, // ou mantem apenas os ultimos 500
    },
  },
});
