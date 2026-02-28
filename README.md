# MindWeek Backend

MindWeek is an intelligent task management system that helps users organize their thoughts and tasks into structured weekly plans using AI-powered processing.

## Features

- **Brain Dump Interface**: Submit unstructured thoughts and ideas that get automatically parsed into organized tasks
- **AI-Powered Task Extraction**: Uses Google Gemini to extract structured tasks from free-form text
- **Smart Task Categorization**: Automatically categorizes tasks and assigns priorities
- **Vector-Based Task Retrieval**: Semantic search and retrieval of relevant tasks using embeddings
- **Intelligent Weekly Planning**: Generates personalized weekly plans based on user tasks and preferences
- **Flexible Planning Modes**: Choose between planning all tasks or focusing on the most relevant ones
- **SQLite Database**: Local storage for user data and tasks
- **Chroma Vector Store**: Efficient similarity search for task retrieval

## Tech Stack

- **Python 3.8+**
- **FastAPI**: Modern, fast web framework for API development
- **SQLModel**: SQL toolkit and ORM combining SQLAlchemy and Pydantic
- **Google Gemini**: AI models for task extraction and planning
- **ChromaDB**: Vector database for semantic search
- **SQLite**: Local relational database storage
- **python-dotenv**: Environment variable management

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd mindweek-backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your Google Gemini API key to the `.env` file

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key (required) | N/A |
| `GEMINI_MODEL` | The Gemini model to use for task extraction and planning | `gemini-2.5-flash` |
| `GEMINI_EMBED_MODEL` | The embedding model for vector representations | `gemini-embedding-001` |

## Usage

1. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### POST `/brain-dump`

Submit a brain dump containing unstructured thoughts. The system will:
- Store the brain dump in the database
- Extract structured tasks using AI
- Categorize tasks with priority and estimated time
- Store tasks in both SQL and vector databases

#### Request Body:
```json
{
  "user_id": "string",
  "content": "string"
}
```

**Schema Details:**
- `user_id`: Unique identifier for the user submitting the brain dump
- `content`: Free-form text containing thoughts, ideas, and tasks to be processed

#### Response Body:
```json
{
  "id": "string",
  "user_id": "string",
  "content": "string",
  "created_at": "2023-01-01T00:00:00.000Z"
}
```

**Schema Details:**
- `id`: Unique identifier for the brain dump record
- `user_id`: User identifier associated with this brain dump
- `content`: Original content submitted
- `created_at`: Timestamp when the brain dump was created

**Functionality:**
1. Creates a new brain dump record in the database
2. Extracts structured tasks from the content using AI (Google Gemini)
3. For each extracted task:
   - Assigns a title, category, priority, and estimated minutes
   - Generates embeddings for semantic search
   - Stores in both SQL database and Chroma vector store
4. Returns the created brain dump record

### POST `/generate-weekly-plan`

Generate a personalized weekly plan based on user tasks. Supports different planning modes for flexible task scheduling.

#### Request Body:
```json
{
  "user_id": "string",
  "k": 10,
  "mode": "all",
  "category": "string"
}
```

**Schema Details:**
- `user_id`: Unique identifier for the user requesting the plan
- `k`: Number of top tasks to consider when using semantic_top_k mode (optional, defaults to 10)
- `mode`: Planning mode - either "all" (plan all tasks) or "semantic_top_k" (plan most relevant tasks)
- `category`: Optional category focus to prioritize certain types of tasks (e.g., "work", "health")

#### Response Body:
```json
{
  "id": "string",
  "user_id": "string",
  "plan_json": "string",
  "created_at": "2023-01-01T00:00:00.000Z"
}
```

**Schema Details:**
- `id`: Unique identifier for the weekly plan
- `user_id`: User identifier associated with this plan
- `plan_json`: JSON string containing the structured weekly plan (with tasks distributed across 7 days)
- `created_at`: Timestamp when the plan was created

**Functionality:**
1. Based on the mode parameter:
   - If mode="all": Retrieves all tasks for the user
   - If mode="semantic_top_k": Uses vector embeddings to find the top-k most relevant tasks
2. If a category is provided, applies soft-focus to prioritize tasks in that category
3. Sends tasks to the AI planning system with constraints:
   - Maximum 8 hours (480 minutes) per day
   - Tasks distributed across all 7 days
   - Higher priority tasks scheduled earlier
   - Balanced workload across the week
4. Runs a validation and correction loop to ensure plan quality
5. Stores the generated plan in the database and returns it

**Plan JSON Structure:**
The `plan_json` field contains a JSON object with exactly 7 keys (Monday through Sunday), each containing an array of scheduled tasks:

```json
{
  "Monday": [
    {
      "title": "Task title",
      "minutes": 60
    }
  ],
  "Tuesday": [],
  "Wednesday": [],
  "Thursday": [],
  "Friday": [],
  "Saturday": [],
  "Sunday": []
}
```

## Data Models

- **BrainDump**: Stores raw user input containing thoughts and ideas
- **Task**: Structured tasks with title, category, priority, and estimated time
- **WeeklyPlan**: Generated weekly plans containing scheduled tasks

## Architecture

The backend follows a service-oriented architecture with:
- **Main API Layer**: FastAPI routes handling requests
- **Service Layer**: Business logic for task extraction and planning
- **Database Layer**: SQLModel for relational data storage
- **Vector Store Layer**: ChromaDB for semantic search capabilities
- **LLM Integration**: Google Gemini for AI-powered processing

## Running Tests

Currently, there are no automated tests included. This would be a good addition for future development.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini for the AI capabilities
- FastAPI for the excellent web framework
- ChromaDB for the vector database solution
- SQLModel for the database ORM