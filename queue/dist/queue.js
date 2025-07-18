'use strict';
var __importDefault =
  (this && this.__importDefault) ||
  function (mod) {
    return mod && mod.__esModule ? mod : { default: mod };
  };
Object.defineProperty(exports, '__esModule', { value: true });
exports.analysisQueue = exports.connection = void 0;
const bullmq_1 = require('bullmq');
const dotenv_1 = __importDefault(require('dotenv'));
dotenv_1.default.config();
// Conexão com Redis (usado pela fila BullMQ)
exports.connection = {
  host: process.env.REDIS_HOST || 'redis', // nome do serviço no docker-compose
  port: 6379,
};
// Define a fila BullMQ chamada "image-analysis"
exports.analysisQueue = new bullmq_1.Queue('image-analysis', {
  connection: exports.connection,
});
