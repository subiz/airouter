package airouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"github.com/subiz/log"
)

const Gpt_4o_mini = "gpt-4o-mini"
const Gpt_4o = "gpt-4o"
const Gpt_4_1_mini = "gpt-4.1-mini"
const Gpt_4_1_nano = "gpt-4.1-nano"
const Gpt_4_1 = "gpt-4.1"

const Gemini_2_0_flash = "gemini-2.0-flash"
const Gemini_2_5_pro = "gemini-2.5-pro"
const Gemini_2_5_flash = "gemini-2.5-flash"
const Gemini_1_5_flash = "gemini-1.5-flash"

const Text_embedding_3_small = "text-embedding-3-small"
const Text_embedding_3_large = "text-embedding-3-large"
const Text_embedding_ada_002 = "text-embedding-ada-002"

const Gemini_embedding_3_small = "gemini-embedding-001"

const USD_TO_VND = 25_575

func ToModel(model string) string {
	if model == "gpt-4o-mini" || strings.HasPrefix(model, "gpt-4o-mini-2") {
		return Gpt_4o_mini
	}

	if model == "gpt-4o" || strings.HasPrefix(model, "gpt-4o-2") {
		return Gpt_4o
	}

	if model == "gpt-4.1" || strings.HasPrefix(model, "gpt-4.1-2") {
		return Gpt_4_1
	}

	if model == "gpt-4.1-nano" || strings.HasPrefix(model, "gpt-4.1-nano-2") {
		return Gpt_4_1_nano
	}

	if model == "gpt-4.1-mini" || strings.HasPrefix(model, "gpt-4.1-mini-2") {
		return Gpt_4_1_mini
	}

	// fallback for gpt
	if strings.HasPrefix(model, "gpt") {
		return Gpt_4o_mini
	}

	if model == "gemini-1.5-flash" {
		return Gemini_1_5_flash
	}

	if model == "gemini-2.0-flash" {
		return Gemini_2_0_flash
	}

	if model == "gemini-2.5-flash" {
		return Gemini_2_5_flash
	}

	if model == "gemini-2.5-pro" {
		return Gemini_2_5_pro
	}

	if strings.HasPrefix(model, "gemini") {
		return Gemini_2_0_flash
	}
	return Gpt_4o_mini
}

func ToGeminiModel(model string) string {
	model = ToModel(model)

	if model == Gpt_4o_mini || model == Gpt_4_1_mini {
		return Gemini_2_0_flash
	}

	if model == Gpt_4_1_nano {
		return Gemini_1_5_flash
	}

	if model == Gpt_4o || model == Gpt_4_1 {
		return Gemini_2_5_pro
	}

	if model == Gpt_4_1_mini {
		return Gemini_2_5_flash
	}

	if !strings.HasPrefix(model, "gemini") {
		return Gemini_2_5_flash
	}
	return model
}

// per 1M tokens
var llmmodelinputprice = map[string]float64{
	"gpt-4.1":      2,
	"gpt-4.1-mini": 0.4,
	"gpt-4.1-nano": 0.1,
	"gpt-4o-mini":  0.15,
	"gpt-4o":       2.5,

	"gemini-1.5-flash": 0.15,
	"gemini-2.0-flash": 0.1,
	"gemini-2.5-flash": 0.30,
	"gemini-2.5-pro":   2.5,
}

// per 1M tokens
var llmmodeloutputprice = map[string]float64{
	"gpt-4.1":      8,
	"gpt-4.1-mini": 1.6,
	"gpt-4.1-nano": 0.4,
	"gpt-4o":       10,
	"gpt-4o-mini":  0.6,

	"gemini-1.5-flash": 0.6,
	"gemini-2.0-flash": 0.4,
	"gemini-2.5-flash": 2.5,
	"gemini-2.5-pro":   15,
}

// per 1M tokens
var llmmodelcachedprice = map[string]float64{
	"gpt-4.1":      8,
	"gpt-4.1-mini": 0.1,
	"gpt-4.1-nano": 0.025,
	"gpt-4o":       1.25,
	"gpt-4o-mini":  0.075,

	"gemini-1.5-flash": 0.15, // nocaching
	"gemini-2.0-flash": 0.1,  // no caching
	"gemini-2.5-flash": 0.225,
	"gemini-2.5-pro":   11.25,
}

// OpenAIChatMessage mimics the structure of a message in an OpenAI Chat Completion request.
type OpenAIChatMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"` // Use pointer to allow for null content
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallId string     `json:"tool_call_id,omitempty"`
	Refusal    string     `json:"refusal,omitempty"`
}

func (m *OpenAIChatMessage) GetContent() string {
	if m.Content == nil {
		return ""
	}

	return *m.Content
}

// ToolCall represents a tool call in an OpenAI response.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents the function details in a tool call.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAITool mimics the structure of a tool in an OpenAI Chat Completion request.
type OpenAITool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type JSONSchema struct {
	Title                string                 `protobuf:"bytes,2,opt,name=title,proto3" json:"title,omitempty"`
	Type                 string                 `protobuf:"bytes,3,opt,name=type,proto3" json:"type,omitempty"` // string, number, object, array, boolean, null
	Description          string                 `protobuf:"bytes,5,opt,name=description,proto3" json:"description,omitempty"`
	Properties           map[string]*JSONSchema `protobuf:"bytes,6,rep,name=properties,proto3" json:"properties,omitempty" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	Items                *JSONSchema            `protobuf:"bytes,7,opt,name=items,proto3" json:"items,omitempty"` // used for type array
	MinItems             int64                  `protobuf:"varint,8,opt,name=minItems,proto3" json:"minItems,omitempty"`
	UniqueItems          bool                   `protobuf:"varint,9,opt,name=uniqueItems,proto3" json:"uniqueItems,omitempty"`
	ExclusiveMinimum     int64                  `protobuf:"varint,10,opt,name=exclusiveMinimum,proto3" json:"exclusiveMinimum,omitempty"`
	Required             []string               `protobuf:"bytes,11,rep,name=required,proto3" json:"required,omitempty"`
	AdditionalProperties bool                   `protobuf:"varint,12,opt,name=additionalProperties,proto3" json:"additionalProperties"` // chatgpt required this
	Enum                 []string               `protobuf:"bytes,13,rep,name=enum,proto3" json:"enum,omitempty"`
}

// Function mimics the structure of a function in an OpenAI Chat Completion request.
type Function struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  *JSONSchema `json:"parameters"`
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type       string       `json:"type,omitempty"`
	JSONSchema *RJSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema represents the JSON schema for the response format.
type RJSONSchema struct {
	Name   string      `json:"name"`
	Strict bool        `json:"strict"`
	Schema *JSONSchema `json:"schema"`
}

// OpenAIChatRequest mimics the structure of an OpenAI Chat Completion request.
type OpenAIChatRequest struct {
	Seed           int                 `json:"seed,omitempty"`
	Messages       []OpenAIChatMessage `json:"messages"`
	Model          string              `json:"model"`
	Temperature    float32             `json:"temperature"`
	TopP           float32             `json:"top_p"`
	Tools          []OpenAITool        `json:"tools"`
	ResponseFormat *ResponseFormat     `json:"response_format,omitempty"`
	ToolChoice     string              `json:"tool_choice,omitempty"`
}

// OpenAIChoice mimics the structure of a choice in an OpenAI Chat Completion response.
type OpenAIChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

// PromptTokensDetails mimics the structure of the prompt_tokens_details field.
type PromptTokensDetails struct {
	CachedTokens int64 `json:"cached_tokens"`
}

// Usage mimics the structure of the usage field in an OpenAI Chat Completion response.
type Usage struct {
	PromptTokens        int64                `json:"prompt_tokens"`
	CompletionTokens    int64                `json:"completion_tokens"`
	TotalTokens         int64                `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// OpenAIError represents the structure of an error in an OpenAI Chat Completion response.
type OpenAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
}

// OpenAIChatResponse mimics the structure of an OpenAI Chat Completion response.
type OpenAIChatResponse struct {
	ID          string         `json:"id,omitempty"`
	Created     int64          `json:"created,omitempty"`
	Object      string         `json:"object,omitempty"`
	Model       string         `json:"model,omitempty"`
	Choices     []OpenAIChoice `json:"choices,omitempty"`
	Usage       *Usage         `json:"usage,omitempty"`
	Error       *OpenAIError   `json:"error,omitempty"`
	ServiceTier string         `json:"service_tier,omitempty"`
}

func ChatCompleteAPI(ctx context.Context, apikey, model string, request []byte) (OpenAIChatResponse, error) {
	var output []byte
	var err error
	if strings.HasPrefix(model, "gemini") {
		output, err = chatCompleteGemini(ctx, apikey, model, request)
	} else {
		output, err = chatCompleteChatGPT(ctx, apikey, model, request)
	}

	if len(output) == 0 && err != nil {
		return OpenAIChatResponse{
			Error: &OpenAIError{Message: err.Error(), Type: "internal_error"},
		}, err
	}
	response := OpenAIChatResponse{}
	if err := json.Unmarshal(output, &response); err != nil {
		return OpenAIChatResponse{Error: &OpenAIError{Message: "Invalid response JSON format", Type: "internal_error"}},
			log.EProvider(nil, "model", "completion", log.M{"_payload": output})
	}
	return response, err
}

func GetEmbeddingAPI(ctx context.Context, apikey, model, text string) (OpenAIEmbeddingResponse, error) {
	if strings.HasPrefix(model, "gemini") {
		return getGeminiEmbedding(ctx, apikey, model, text)
	}
	return getOpenAIEmbedding(ctx, apikey, model, text)
}

var chatgpttimeouterr = []byte(`{"error": {"message": "Error: Timeout was reached","type": "timeout"}}`)

func chatCompleteChatGPT(ctx context.Context, apikey, model string, request []byte) ([]byte, error) {
	rq := &OpenAIChatRequest{}
	json.Unmarshal(request, rq)
	request, _ = json.Marshal(rq)
	url := "https://api.openai.com/v1/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(request))
	if err != nil {
		return nil, log.EServer(err)
	}
	req.Header.Set("Authorization", "Bearer "+apikey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	output := buf.Bytes()

	if resp.StatusCode != 200 {
		if strings.EqualFold(string(output), "Error: Timeout was reached") {
			return chatgpttimeouterr, log.EProvider(nil, "openai", "completion", log.M{"status": resp.StatusCode, "_payload": output})
		}

		return output, log.EProvider(nil, "openai", "completion", log.M{"status": resp.StatusCode, "_payload": output})
	}

	return output, nil
}

// return 1000 usd fpv
func CalculateCost(model string, usage *Usage) int64 {
	if usage == nil {
		return 0
	}

	if model == "gemini-embedding-001" {
		return int64(1000 * float64(usage.TotalTokens) * 0.15) // $0.15 per 1M token
	}

	if model == "text-embedding-3-small" {
		return int64(1000 * float64(usage.TotalTokens) * 0.02) // $0.02 per 1M token
	}

	if model == "text-embedding-3-large" {
		return int64(1000 * float64(usage.TotalTokens) * 0.13) // $0.13 per 1M token
	}

	if model == "text-embedding-ada-002" {
		return int64(1000 * float64(usage.TotalTokens) * 0.1) // $0.10 per 1M token
	}

	inputtoken := usage.PromptTokens
	var cachedtoken int64
	if usage.PromptTokensDetails != nil {
		inputtoken -= usage.PromptTokensDetails.CachedTokens
		cachedtoken = usage.PromptTokensDetails.CachedTokens
	}
	outputtoken := usage.CompletionTokens
	return int64(math.Ceil(1000 * (float64(inputtoken)*llmmodelinputprice[ToModel(model)] +
		float64(cachedtoken)*llmmodelcachedprice[ToModel(model)] +
		float64(outputtoken)*llmmodeloutputprice[ToModel(model)])))
}

type OpenAIEmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type OpenAIEmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type OpenAIEmbeddingResponse struct {
	Object string                `json:"object"`
	Data   []OpenAIEmbeddingData `json:"data"`
	Model  string                `json:"model"`
	Usage  Usage                 `json:"usage"`
	Error  *OpenAIError          `json:"error,omitempty"`
}

// model text-embedding-3-small
func getOpenAIEmbedding(ctx context.Context, apiKey, model, text string) (OpenAIEmbeddingResponse, error) {
	url := "https://api.openai.com/v1/embeddings"
	reqBody := OpenAIEmbeddingRequest{Input: text, Model: model}
	out := OpenAIEmbeddingResponse{}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		out.Error = &OpenAIError{Message: fmt.Sprintf("error marshalling request body: %s", err.Error())}
		return out, fmt.Errorf("error marshalling request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		out.Error = &OpenAIError{Message: fmt.Sprintf("error creating request: %s", err.Error())}
		return out, fmt.Errorf("error creating request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		out.Error = &OpenAIError{Message: fmt.Sprintf("error sending request: %s", err.Error())}
		return out, log.EProvider(err, "openai", "embedding", log.M{"status": resp.StatusCode, "model": model})
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	err = json.Unmarshal(bodyBytes, &out)
	if resp.StatusCode != http.StatusOK {
		return out, log.EProvider(err, "openai", "embedding", log.M{"model": model, "status": resp.StatusCode, "_payload": bodyBytes})
	}
	return out, err
}
