# queue/Dockerfile
FROM node:18

WORKDIR /app

# Copy only package.json and lock file first (for layer caching)
COPY package*.json ./

RUN npm install

# Now copy the rest of the code
COPY . .

# Build TypeScript ahead of time
RUN npm run build

# This will be overridden by docker-compose for API vs Worker
CMD ["node", "dist/api/server.js"]
