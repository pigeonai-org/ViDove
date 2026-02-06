# ViDove Web Frontend with LLM Integration

A modern, fully-typed web frontend for ViDove translation configuration using conversational AI. Built with **TypeScript React** frontend and **FastAPI** backend with comprehensive type safety.

## 🚀 Features

- **💬 Conversational Configuration**: Chat with an AI assistant to configure translation tasks
- **🔧 Real-time Configuration**: Live display of current settings with type safety
- **📁 File Upload**: Support for video, audio, and SRT files with typed interfaces  
- **📋 Task Management**: Monitor translation task progress
- **🎨 Modern UI**: Responsive design with professional styling
- **📝 Full TypeScript**: End-to-end type safety for better development experience
- **🔍 Type Checking**: Integrated mypy for backend and TypeScript for frontend

## 🏗️ Architecture

### Frontend (TypeScript React)
- **Type-safe API client** with comprehensive interfaces
- **Pydantic-style models** for frontend data validation
- **Real-time chat interface** with proper typing
- **File upload handling** with typed responses
- **Responsive design** optimized for various screen sizes

### Backend (FastAPI with Type Annotations)
- **Pydantic models** for all data structures
- **Type-safe endpoints** with response models
- **LLM integration** using OpenAI GPT-4o
- **Session management** with typed configurations
- **File handling** with type validation

## 🛠️ Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### 1. Environment Setup
```bash
# Clone the repository
cd demo

# Create environment file
cp env.example .env

# Add your OpenAI API key to .env
OPENAI_API_KEY=your_api_key_here
```

### 2. Development Mode
```bash
# Start all services in development mode
docker-compose up --build

# Frontend will be available at http://localhost:3000
# Backend API at http://localhost:8000
```

### 3. Production Mode
```bash
# Build and start production containers
docker-compose -f docker-compose.yml up --build -d
```

## 🔧 Development

### Frontend Development (TypeScript)

```bash
cd frontend

# Install dependencies with TypeScript support
npm install

# Start development server with hot reload
npm start

# Type checking
npm run type-check

# Build production bundle
npm run build
```

### Key Frontend Type Definitions
- `SessionConfig`: Complete translation configuration interface
- `ChatMessage`: Type-safe message structure  
- `ApiService`: Fully typed API client with error handling
- `TaskStatus`: Typed task state management

### Backend Development (Python with Types)

```bash
cd backend

# Install dependencies including type checking tools
pip install -r requirements.txt

# Run development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Type checking with mypy
mypy main.py

# API documentation available at http://localhost:8000/docs
```

### Key Backend Type Features
- **Pydantic models** for all request/response validation
- **Literal types** for enum-like values (input types, status values)
- **Generic type hints** for collections and optionals
- **Type-safe configuration** schema with validation

## 📊 API Documentation

### Type-Safe Endpoints

#### Chat & Configuration
- `POST /api/chat/start` → `StartSessionResponse`
- `POST /api/chat/{session_id}/message` → `SendMessageResponse`  
- `GET /api/chat/{session_id}/config` → `ConfigResponse`

#### Task Management
- `POST /api/tasks/create` → `CreateTaskResponse`
- `GET /api/tasks/{task_id}/status` → `TaskStatus`
- `GET /api/tasks` → `List[TaskInfo]`

#### File Operations
- `POST /api/upload` → `UploadFileResponse`

### Configuration Schema Types

All configuration options are fully typed with validation:

```typescript
interface SessionConfig {
  source_lang: "EN" | "ZH" | "ES" | "FR" | "DE" | "RU" | "JA" | "AR" | "KR";
  target_lang: "EN" | "ZH" | "ES" | "FR" | "DE" | "RU" | "JA" | "AR" | "KR";
  domain: "General" | "SC2";
  'translation.model': "gpt-3.5-turbo" | "gpt-4" | "gpt-4o" | "Assistant" | "Multiagent";
  // ... other typed configuration options with ViDove-specific values
}
```

## 🎯 Type Safety Benefits

### Frontend (TypeScript)
- **Compile-time error catching** for API calls and data handling
- **IntelliSense support** with auto-completion
- **Refactoring safety** with IDE support
- **Interface contracts** between components

### Backend (Python Type Hints + Pydantic)
- **Runtime validation** of request/response data
- **Automatic API documentation** generation
- **IDE support** with type hints
- **Error prevention** through static analysis

## 🧪 Testing

### Frontend Type Checking
```bash
cd frontend
npm run type-check  # TypeScript compilation check
npm test           # Jest tests with type support
```

### Backend Type Checking  
```bash
cd backend
mypy main.py       # Static type analysis
python -m pytest  # Runtime tests (if implemented)
```

## 🔐 Security & Types

- **Input validation** through Pydantic models
- **Type-safe environment** variable handling
- **CORS configuration** with typed origins
- **File upload validation** with size and type checks

## 📦 Deployment

The application uses **multi-stage Docker builds** optimized for TypeScript compilation:

### Frontend Build Process
1. **TypeScript compilation** and type checking
2. **React optimization** and bundling  
3. **Nginx static serving** in production

### Backend Build Process
1. **Python dependency** installation
2. **Type checking** with mypy (optional)
3. **FastAPI production** server with Uvicorn

## 🛠️ Configuration

### Environment Variables (Typed)
```bash
# Backend
OPENAI_API_KEY=your_api_key          # Required for LLM integration

# Frontend  
REACT_APP_API_URL=http://localhost:8000  # Backend API endpoint
GENERATE_SOURCEMAP=false                  # Production optimization
```

## 🤝 Contributing

1. **Type safety first**: All new code must include proper type annotations
2. **Frontend**: Use TypeScript interfaces and avoid `any` types  
3. **Backend**: Include Pydantic models and type hints
4. **Testing**: Include type checking in CI/CD pipeline
5. **Documentation**: Update type interfaces when adding features

## 📚 Integration with ViDove

This web frontend generates **type-safe configuration objects** that can be directly consumed by the main ViDove translation pipeline:

```python
# Generated configuration is fully typed and validated
config: SessionConfig = session.current_config
task = Task.fromYoutubeLink(url, task_id, task_dir, config.dict())
```

The conversational interface makes ViDove accessible to users without technical expertise while maintaining the full power and type safety of the underlying system.
