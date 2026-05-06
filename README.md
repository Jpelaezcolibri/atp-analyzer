# ATP Port Analyzer

ATP Port Analyzer is a stateless FastAPI REST API that receives an image URL for an ATP fiber optic distribution box and returns a structured JSON analysis of the 8 port occupancy statuses. It downloads the image in memory, sends it to Anthropic Claude Opus 4.5 Vision, validates the resulting port logic, and returns occupied ports, available ports, device label data, observations, and confidence.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Anthropic API key with access to your configured vision model
- An API key value of your choice for protecting `/analyze`

## Local Setup

```bash
git clone <your-repository-url>
cd atp-analyzer
cp .env.example .env
```

Fill in `.env`:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-opus-4-5
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.1
API_KEY=your_api_secret_key_here
PORT=8000
ENVIRONMENT=development
```

Start the API:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

## Model Selection

The default provider is Anthropic, matching the production specification. You can change the Anthropic vision model without changing code by editing `.env`:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-opus-4-5
```

For this use case, a strong alternative is OpenAI `gpt-5.1` through the Responses API because it supports image input and structured outputs. To use it, set:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.1
```

Keep `API_KEY` configured either way; it protects the public `/analyze` endpoint.

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

## Analyze An Image

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"image_url": "https://your-image-host.com/atp-box.jpg"}'
```

## Full Example Request

```json
{
  "image_url": "https://your-image-host.com/atp-box.jpg"
}
```

## Full Example Response

```json
{
  "success": true,
  "image_url": "https://your-image-host.com/atp-box.jpg",
  "analyzed_at": "2026-05-05T14:30:00Z",
  "result": {
    "total_puertos": 8,
    "total_ocupados": 4,
    "puertos_ocupados": [1, 2, 3, 4],
    "total_disponibles": 4,
    "puertos_disponibles": [5, 6, 7, 8]
  },
  "device_info": {
    "codigo_dispositivo": "NE-312099-CO2-TARAP-CL6",
    "ubicacion": "RESERVA SERRAT (SELVA) TO 2",
    "observaciones": null
  },
  "confidence": "high"
}
```

## Error Response Format

```json
{
  "success": false,
  "error": "description of what went wrong",
  "image_url": "https://your-image-host.com/atp-box.jpg",
  "analyzed_at": "2026-05-05T14:30:00Z"
}
```

## Deployment

This API works on any VPS or on Docker-based platforms such as Railway and Render. Provide `LLM_PROVIDER`, the matching provider key and model, `API_KEY`, `PORT`, and `ENVIRONMENT` as environment variables, then deploy the Docker container.
