# AIFlow Backend

This is the FastAPI backend foundation for AIFlow.

## Requirements
- Python 3.11+
- virtualenv

## Local Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\Activate.ps1`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a local environment file:
   ```bash
   copy NUL .env
   ```
   or on Unix/macOS:
   ```bash
   touch .env
   ```

5. Add provider credentials in `.env` without committing secrets:
   ```env
   GEMINI_API_KEY="your-gemini-api-key"
   GROQ_API_KEY="your-groq-api-key"
   ```

6. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.
   API Documentation (Swagger UI) is available at `http://127.0.0.1:8000/api/v1/openapi.json` or `http://127.0.0.1:8000/docs`.

## Running Tests
Run the test suite using pytest:
```bash
python -m pytest -q
```

To run one provider-focused test module while developing:
```bash
python -m pytest tests/test_providers.py -q
```

After starting the server, verify that it is responding:
```bash
curl http://127.0.0.1:8000/health
```

## Supported Providers

AIFlow currently supports the following AI providers:

### Google Gemini (REST API)
To enable Gemini, provide your API key in the `.env` file:
```env
GEMINI_API_KEY="your-gemini-api-key"
```
The application will automatically register the `GeminiProvider` on startup if the key is present. Supported models include `gemini-1.5-flash`, `gemini-1.5-pro`, and `gemini-1.0-pro`.

### Groq (REST API)
To enable Groq, provide your API key in the `.env` file:
```env
GROQ_API_KEY="your-groq-api-key"
```
The application will automatically register the `GroqProvider` on startup if the key is present. Supported models include `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, and `gemma2-9b-it`.
