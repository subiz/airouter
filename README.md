# AI Router

[![Go Reference](https://pkg.go.dev/badge/github.com/subiz/airouter.svg)](https://pkg.go.dev/github.com/subiz/airouter)
[![Go Report Card](https://goreportcard.com/badge/github.com/subiz/airouter)](https://goreportcard.com/report/github.com/subiz/airouter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI Router is a lightweight Go client for the Subiz AI Proxy, providing a unified interface to multiple LLM providers including OpenAI, Google Gemini, and more.

## Features

- **Unified API**: Single interface for Chat Completion, Text Embedding, and Reranking across different providers.
- **Provider Agnostic**: Switch between GPT-4, GPT-5, Gemini, and other models without changing your core logic.
- **Simplified Configuration**: Easy initialization with a Subiz API key.
- **Reasoning Support**: Support for modern reasoning models with configurable effort levels.

## Installation

```bash
go get github.com/subiz/airouter
```

## Getting Started

### Initialization

Initialize the library with your Subiz API key:

```go
import "github.com/subiz/airouter"

func init() {
    airouter.Init("YOUR_SUBIZ_API_KEY")
}
```

### Usage Examples

#### 1. Chat Completion

```go
package main

import (
    "context"
    "fmt"
    "github.com/subiz/airouter"
)

func main() {
    airouter.Init("YOUR_SUBIZ_API_KEY")

    ctx := context.Background()
    output, _, err := airouter.Complete(ctx, airouter.CompletionInput{
        Model: airouter.Gpt_5_nano,
        Reasoning: &airouter.CompletionReasoning{
            Effort: "low",
        },
        Instruct: "Tell a short story about a brave dragon.",
    })

    if err != nil {
        panic(err)
    }

    fmt.Println(output) // The dragon’s name was Ember, and his light was not a blaze but a memory of dawn...
}
```

#### 2. Text Embedding

```go
package main

import (
    "context"
    "fmt"
    "github.com/subiz/airouter"
)

func main() {
    airouter.Init("YOUR_SUBIZ_API_KEY")

    vector, _, err := airouter.GetEmbedding(context.Background(), airouter.Text_embedding_3_small, "Hello world")
    if err != nil {
        panic(err)
    }

    fmt.Printf("Vector length: %d\n", len(vector))
    fmt.Println(vector[:5]) // Print first 5 dimensions
}
```

#### 3. Rerank

```go
package main

import (
    "context"
    "fmt"
    "github.com/subiz/airouter"
)

func main() {
    airouter.Init("YOUR_SUBIZ_API_KEY")

    records := []*airouter.RerankRecord{
        {Id: "1", Title: "Greeting", Content: "Xin chào thế giới"},
        {Id: "2", Title: "Farewell", Content: "Goodbye world"},
    }

    output, err := airouter.Rerank(context.Background(), "google", "hello", records)
    if err != nil {
        panic(err)
    }

    fmt.Println("Rerank results:", output)
}
```

## Supported Models

### Chat Completion

| Provider | Model Aliases |
|----------|---------------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano` |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite-preview` |

### Text Embedding

- `gemini-embedding-001`
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

### Reranking

- `google` (Powered by Google Vertex AI)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
