# shakespearen-rag

A semantic search tool for Shakespeare's works using ChromaDB and local embeddings, with FastAPI backend and Next.js frontend.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the vector database (run once):
   ```bash
   python script.py
   ```
   This will download Shakespeare's complete works, chunk the text, generate embeddings, and store them in ChromaDB.

3. Set up environment variables:
   Copy `.env.example` to `.env` and add your Cerebras API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your actual API key:
   ```
   CEREBRAS_API_KEY=your_api_key_here
   ```

## Running the Application

### Option 1: Docker (Recommended)

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```
   This will automatically:
   - Build the FastAPI backend and Next.js frontend
   - Initialize the ChromaDB database (if not present)
   - Start both services with proper networking
   - Mount the database as a persistent volume

2. **Access the application:**
   - Web UI: http://localhost:3000
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

3. **Stop the services:**
   ```bash
   docker-compose down
   ```

#### Docker Development Mode (Faster for Development)
For faster development with hot reloading:
```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml up --build
```

This enables:
- Hot reloading for both backend and frontend
- Faster rebuilds during development
- Source code mounting instead of copying

### Option 2: Local Development

#### Option 2.1: CLI Interface (Original)
```bash
python query.py
```
Enter your search queries interactively. Type 'exit' or 'quit' to stop.

#### Option 2.2: Web Interface (Local)

1. **Start the FastAPI backend:**
   ```bash
   python app.py
   ```
   The API will be available at http://localhost:8000

2. **Start the Next.js frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
   The web interface will be available at http://localhost:3000

3. Open your browser and navigate to http://localhost:3000 to start querying!

## Components

- **script.py**: Builds the vector database from Shakespeare's texts
- **query.py**: Interactive CLI search interface
- **app.py**: FastAPI backend API for web interface
- **frontend/**: Next.js web application with modern UI
- **Dockerfile.backend**: Docker configuration for the Python backend
- **Dockerfile.frontend**: Docker configuration for the Next.js frontend
- **docker-compose.yml**: Orchestrates the multi-container application

## Technology Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: ChromaDB
- **Generation**: Cerebras LLM (llama-3.3-70b)
- **Backend**: FastAPI (Python)
- **Frontend**: Next.js (React + TypeScript + Tailwind CSS)
- **Containerization**: Docker & Docker Compose

## Docker Performance Optimizations

The Docker setup includes several optimizations for faster builds and startup:

- **Multi-stage builds**: Separate build and runtime stages for smaller images
- **Layer caching**: Optimized COPY commands and dependency installation order
- **Build context optimization**: Comprehensive `.dockerignore` files
- **Health checks**: Proper startup sequencing with health checks
- **Development mode**: Hot reloading with `docker-compose.override.yml`
- **Alpine Linux**: Small production images (Node.js runner uses Alpine)
- **Build optimizations**: `--prefer-offline`, `--no-audit` for faster npm installs

The system allows natural language searches through Shakespeare's works using retrieval-augmented generation (RAG).
