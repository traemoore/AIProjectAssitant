# AI Code Assistant

## Backend (Python)
- The backend is built using Flask and uses the GPT-2 model from the HuggingFace Transformers library to generate text based on user input.
- It exposes an API endpoint `/generate` to handle POST requests with a prompt and return generated text.

### Backend Docker
- A Dockerfile is provided to containerize the Python backend application.

### Kubernetes Deployment
- A Kubernetes deployment and service YAML configuration is provided for deploying the backend.

## Frontend (React Native)
- The frontend is a simple React Native app that sends prompts to the backend and displays the AI-generated responses.

### Frontend Docker
- A Dockerfile is provided to containerize the React Native app.

## How to Run

### Backend:
1. Navigate to the backend directory.
2. Build the Docker image:
   ```
   docker build -t ai-backend .
   ```
3. Run the Docker container:
   ```
   docker run -p 5000:5000 ai-backend
   ```

### Frontend:
1. Navigate to the frontend directory.
2. Build the Docker image:
   ```
   docker build -t ai-frontend .
   ```
3. Run the Docker container:
   ```
   docker run -p 8081:8081 ai-frontend
   ```

## Kubernetes Deployment
1. Deploy the backend on Kubernetes:
   ```
   kubectl apply -f k8s_deployment.yaml
   kubectl apply -f k8s_service.yaml
   ```
2. Access the service via the load balancer IP.
