# 🧠 User Intelligence Agent

An AI-powered system that analyzes conversation history, extracts user persona insights, and answers questions about the user using Retrieval-Augmented Generation (RAG).

The application:

* Processes conversation data from a CSV file.
* Detects and summarizes discussion topics.
* Extracts user habits, traits, facts, and communication style.
* Stores insights in ChromaDB for semantic search.
* Provides an interactive dashboard for querying user information.

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Add Conversation Data

Place your conversation dataset in:

```text
conversations.csv
```

### 4. Run the Application

```bash
uvicorn main:app --reload
```

### 5. Open in Browser

```text
http://localhost:8000
```

## 💡 Usage

1. Start the application.
2. The system automatically analyzes the conversation dataset.
3. View the extracted persona on the dashboard.
4. Ask questions such as:

```text
What are the user's interests?
What personality traits can be inferred?
What technologies does the user frequently discuss?
```

The system retrieves relevant conversation insights and generates answers based on the analyzed data.
