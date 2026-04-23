# AI Router Project Context

This project is a lightweight Go client for the Subiz AI Proxy, providing a unified interface for multiple LLM providers (OpenAI, Google Gemini, etc.).

## Project Overview

- **Purpose**: Abstract various LLM APIs into a single, provider-agnostic Go interface.
- **Main Technologies**: Go (1.26.1+), Subiz AI Proxy, Protobuf, JSON.
- **Backend**: Communicates with `https://api.subiz.com.vn/4.1/ai` (configurable via `BACKEND` variable).
- **Core Features**: Chat Completion, Text Embedding, and Reranking.

## Architecture & Logic

- **Unified Interface**: The library maps high-level requests (e.g., `CompletionInput`) to provider-specific payloads (OpenAI/Gemini).
- **Model Aliasing**: Standardizes model names across providers (e.g., `Gpt_4o`, `Gemini_2_5_pro`).
- **Initialization**:
    - `airouter.Init(apiKey)`: Used by clients with a Subiz API key.
    - `airouter.InitAPI(geminiKey, openaiKey)`: Used for direct provider access (server-side).

## Key Files

- `airouter.go`: Main entry point containing core client logic, model constants, and implementation of `Complete`, `GetEmbedding`, and `Rerank`.
- `api.go`: Defines the data structures (structs) for requests and responses, including provider-specific types for OpenAI and Gemini.
- `airouter_test.go`: Contains the test suite, heavily utilizing data-driven tests.
- `testcases/`: A directory containing JSON files used for regression testing of request/response conversions.
- `go.mod`: Project dependencies (Subiz internal libraries like `executor`, `header`, `log`).

## Building and Running

- **Install Dependencies**: `go mod tidy`
- **Build**: `go build ./...`
- **Test**: `go test ./...`
- **Installation**: `go get github.com/subiz/airouter`

## Development Conventions

- **Testing**: New features or bug fixes should be accompanied by test cases. Add new JSON test cases to the `testcases/` directory if they involve request/response transformations.
- **Model Constants**: Add new models as constants in `airouter.go` and update the `ToModel` function for proper aliasing.
- **Error Handling**: Uses custom error parsing logic (see `parseError` in `airouter.go`) to handle both unified and provider-specific error formats.
- **Dependencies**: Prefer using the existing Subiz utility libraries (`executor`, `header`, `log`) for consistency.
