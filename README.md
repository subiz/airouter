# AI Router

[![Go Reference](https://pkg.go.dev/badge/github.com/subiz/airouter.svg)](https://pkg.go.dev/github.com/subiz/airouter)
[![Go Report Card](https://goreportcard.com/badge/github.com/subiz/airouter)](https://goreportcard.com/report/github.com/subiz/airouter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI Router is a lightweight Go client for the Subiz AI Proxy, providing a unified interface to multiple LLM providers including OpenAI, Google Gemini, and more.

## Features

- **Unified API**: Single interface for Chat Completion, Text Embedding, and Reranking across different providers.
- **Provider Agnostic**: Switch between GPT-4, GPT-5, Gemini, and other models without changing your core logic.
- **Simplified Configuration**: Easy initialization with a Subiz API key.

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
	accid := "your-account-id"
    output, _, err := airouter.Complete(ctx, accid, airouter.CompletionInput{
        Model: airouter.Gpt_5_nano,
        ReasoningEffort: "low",
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

### Prompt Caching for `gpt-6-luna`

`gpt-6-luna` (like other gpt-5.6+ models) caches the prompt only at **breakpoints**, not at any shared prefix. A request reads the cache only when everything up to one of its breakpoints is byte-identical to an earlier request, so changing a single character before the breakpoint misses it.

By default the model places one implicit breakpoint at the end of the latest user message. When only the question at the end changes (same instruct and documents), that breakpoint never matches, every request writes the whole prompt again, and cache writes cost 1.25x the input price. The router does not place any breakpoint by itself, the client decides what is cached:

- **`prompt_cache_key`**: required by gpt-5.6+ models for reliable matching. When the client sends none, the router uses `md5(model + "." + instruct)`. Send your own key (e.g. per document) for high traffic, a single key receiving many requests per minute spreads over several machines and misses more.
- **Client breakpoints**: put the stable part (instruct, documents) before the changing part (question) and mark the end of the stable part. The cached prefix runs from the start of the request through the end of the marked message or block (inclusive).

```json
{
  "model": "gpt-6-luna",
  "prompt_cache_key": "doc-240147sgo",
  "instruct": "...",
  "messages": [
    { "role": "user", "content": "<document>", "cache_breakpoint": true },
    { "role": "user", "contents": [
      { "type": "text", "text": "<catalog>", "prompt_cache_breakpoint": { "ttl": "12h" } },
      { "type": "text", "text": "<history>" }
    ]},
    { "role": "user", "content": "<question>" }
  ]
}
```

| Field | Where | Effect |
|-------|-------|--------|
| `cache_breakpoint: true` | message | breakpoint at the end of the message (its last text block), default 30m ttl |
| `prompt_cache_breakpoint: {mode, ttl}` | `contents` text block | breakpoint at the end of that block, `mode` defaults to `explicit`, `ttl` is `30m` or `12h` |

Rules:

- A breakpoint must sit on a message with text. An empty marker message is dropped. Breakpoints on `assistant` and `tool` messages are ignored, only input text blocks accept them.
- At most 4 cache writes per request. The implicit breakpoint takes one, leaving 3 for the client.
- The prefix through a breakpoint must be at least 1,024 tokens to be cached.
- Cache writes cost 1.25x (30m) or 2.5x (12h) the input price, cache reads 0.05x.
- Other models never receive the breakpoint fields (older OpenAI models, chat completions, Gemini).

## Supported Models

### Chat Completion

| Provider | Model Aliases |
|----------|---------------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano` |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it` |

Gemma 4 aliases `gemma-4-31b` and `gemma-4-26b-a4b` are normalized to the Gemini API model IDs `gemma-4-31b-it` and `gemma-4-26b-a4b-it`. Both Gemma 4 models have a 256K token context length.

### Text Embedding

- `gemini-embedding-001`
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

### Reranking

- `google` (Powered by Google Vertex AI)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
