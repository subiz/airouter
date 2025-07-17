package airouter

import (
	"bytes"
	"math"
	"context"
	"encoding/json"
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

	"gemini-2.0-flash": 0.4, // no caching
	"gemini-2.5-flash": 0.225,
	"gemini-2.5-pro":   11.25,
}

// OpenAIChatMessage mimics the structure of a message in an OpenAI Chat Completion request.
type OpenAIChatMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"` // Use pointer to allow for null content
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallId string     `json:"tool_call_id,omitempty"`
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

// Function mimics the structure of a function in an OpenAI Chat Completion request.
type Function struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

// Parameters mimics the structure of the parameters in an OpenAI Chat Completion request.
type Parameters struct {
	Type                 string              `json:"type,omitempty"`
	Properties           map[string]Property `json:"properties"`
	Required             []string            `json:"required"`
	AdditionalProperties bool                `json:"additionalProperties"`
}

// Property mimics the structure of a property in an OpenAI Chat Completion request.
type Property struct {
	Type        string `json:"type,omitempty"`
	Description string `json:"description"`
}

// OpenAIChatRequest mimics the structure of an OpenAI Chat Completion request.
type OpenAIChatRequest struct {
	Messages    []OpenAIChatMessage `json:"messages"`
	Model       string              `json:"model"`
	Temperature float32             `json:"temperature"`
	TopP        float32             `json:"top_p"`
	Tools       []OpenAITool        `json:"tools"`
}

// OpenAIChoice mimics the structure of a choice in an OpenAI Chat Completion response.
type OpenAIChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

// PromptTokensDetails mimics the structure of the prompt_tokens_details field.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// Usage mimics the structure of the usage field in an OpenAI Chat Completion response.
type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
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
	ID      string         `json:"id,omitempty"`
	Object  string         `json:"object,omitempty"`
	Model   string         `json:"model,omitempty"`
	Choices []OpenAIChoice `json:"choices,omitempty"`
	Usage   *Usage         `json:"usage,omitempty"`
	Error   *OpenAIError   `json:"error,omitempty"`
}

func ChatCompleteAPI(ctx context.Context, apikey, model string, request []byte) (*OpenAIChatResponse, error) {
	var output []byte
	var err error
	if strings.HasPrefix(model, "gemini") {
		output, err = chatCompleteGemini(ctx, apikey, model, request)
	} else {
		output, err = chatCompleteChatGPT(ctx, apikey, model, request)
	}

	response := &OpenAIChatResponse{}
	err = json.Unmarshal(output, response)
	return response, err
}

func chatCompleteChatGPT(ctx context.Context, apikey, model string, request []byte) ([]byte, error) {
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

	// fmt.Println("REQ", string(request))

	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	output := buf.Bytes()

	if resp.StatusCode != 200 {
		return nil, log.EProvider(nil, "openai", "completion", log.M{"status": resp.StatusCode, "_payload": output})
	}

	return output, nil
}

// usd fpv
func CalculateCost(model string, usage *Usage) int64 {
	inputtoken := usage.PromptTokens
	var cachedtoken int
	if usage.PromptTokensDetails != nil {
		inputtoken -= usage.PromptTokensDetails.CachedTokens
		cachedtoken = usage.PromptTokensDetails.CachedTokens
	}
	outputtoken := usage.CompletionTokens
	return int64(math.Ceil(float64(inputtoken)*llmmodelinputprice[ToModel(model)] +
		float64(cachedtoken)*llmmodelcachedprice[ToModel(model)] +
		float64(outputtoken)*llmmodeloutputprice[ToModel(model)]))
}
