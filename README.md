# 🧠 Isskin Backend

This repository contains the complete backend of the image analysis application for dermatological detection of skin cancer. We use FastAPI with YOLOv11 models for inference, Next.js (API Routes) with BullMQ for queue orchestration, Redis as a broker, Firebase as a database, and Docker Compose for local orchestration and deployment on Azure App Service.

## 📦 Tech Stack

The backend is composed of different technologies working together across layers. Here's a breakdown of what each component does:

| Layer       | Technology                            | Description                                                               |
| ----------- | ------------------------------------- | ------------------------------------------------------------------------- |
| Inference   | **Python 3.10, FastAPI, YOLOv11**     | Processes skin images using trained machine learning models               |
| Queue       | **BullMQ (Node.js), Redis**           | Manages asynchronous job queues so image processing doesn't block the API |
| API Gateway | **Next.js API Routes**                | Exposes endpoints for uploading images and managing jobs                  |
| Database    | **Firebase Firestore**                | Stores job metadata, such as status and prediction results                |
| Storage     | **Firebase Storage**                  | Saves the original user-uploaded images                                   |
| Deployment  | **Docker Compose, Azure App Service** | Containerizes and orchestrates the entire backend infrastructure          |
| Monitoring  | **RAM Route (/ram)**                  | Allows basic resource monitoring (especially memory) to enable throttling |

## 🚀 How to run locally

You’ll need Docker installed on your machine. We recommend Postman or a similar tool for sending test requests.

Once that is sorted, you will need to:

1. Clone the projecty:

```bash
git clone https://isskin@dev.azure.com/isskin/backend/_git/backend
cd isskin-backend
```

2. Copy the example `.env` file and fill in your credentials:

```env
FRONTEND_URLS=
REDIS_HOST=
FIREBASE_STORAGE_BUCKET=
FIREBASE_CREDENTIAL=
```

To generate Firebase credentials, simply navigate to the Service Accounts section under your project settings and download the JSON credentials file. Since the deployment require the credentials in Base64 format, you can convert the downloaded JSON file by running the following command in your terminal:

```bash
base64 -i yourfile.json
```

This command will output the Base64-encoded version of your Firebase credentials, which you can then paste on your `.env`.

Also, your `FIREBASE_STORAGE_BUCKET` should not include the `gs://` prefix.

3. Start the Services:

```bash
docker compose up --build
```

4. Access the Node.js API from your frontend or tools at:

```bash
http://localhost:3001/analyze
```

Since the Node.js API is the main entrypoint the frontend talks to, it listens on port 3001.

4. Use Postman or another tool to send POST requests to the API and monitor the response flow. If you prefer, you can also test it with curl:

```bash
curl -X POST http://localhost:3001/analyze -F "image=@/path/to/image.jpg"
```

## 📸 Application Flow

This is how your image travels through the system, step by step:

#### User Upload

The frontend sends an image to the Next.js API Route.

#### Queueing

The API doesn't handle the image directly. Instead, it creates a job and pushes it to a BullMQ queue, which is stored in Redis.

#### Worker Processing

A Node.js worker listens for new jobs in the queue. When it finds one:

It first checks the available memory on the Python server (to avoid overloading).

If memory is fine, it sends the image to FastAPI.

#### Inference (FastAPI)

FastAPI receives the image and runs YOLOv11 to perform predictions.

#### Saving Results

Once the analysis is done:

The image is stored in Firebase Storage.

The results and status (done, error, etc.) are saved in Firestore.

#### Frontend Update

The frontend fetches the job status from Firestore using the jobId originally returned, and displays the result when it's ready.

# 📁 Folder Structure

```bash
app/                      # Python service (FastAPI + YOLOv11)
├── models/               # Trained YOLOv11 models
│   ├── isskin-binary-11l-v1.pt
│   └── isskin-dx-11l-v1.pt
├── main.py               # FastAPI app
└── Dockerfile            # Dockerfile for Python service

queue/                    # Node.js queue manager and API
├── api/                  # Next.js API Routes (image upload, job creation)
│   └── server.ts
├── worker.ts             # BullMQ worker that runs YOLO jobs
├── queue.ts              # Queue creation and job config
├── Dockerfile            # Dockerfile for Node.js service
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
└── .prettierrc           # Linter config

docker-compose.yml        # Docker Compose config (all services)
.env                      # Environment variable config
```

## 🧹 Job Cleanup Strategy

Redis stores all jobs in memory. To prevent memory bloat, we’ve configured automatic cleanup via BullMQ:

```ts
defaultJobOptions: {
  removeOnComplete: {
    age: 3600,     // Remove completed jobs after 1 hour
    count: 1000    // Or keep only the latest 1000
  },
  removeOnFail: {
    age: 86400,    // Remove failed jobs after 24 hours
    count: 500     // Or keep only the latest 500
  }
}
```

This keeps the system stable and memory-efficient, especially when dealing with large job volumes.
