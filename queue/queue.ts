import { Queue } from 'bullmq';
import dotenv from 'dotenv';

dotenv.config();

// Conexão com Redis (usado pela fila BullMQ)
export const connection = {
  host: process.env.REDIS_HOST || 'redis', // nome do serviço no docker-compose
  port: 6379,
};

// Define a fila BullMQ chamada "image-analysis"
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
