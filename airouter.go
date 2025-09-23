package airouter

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	neturl "net/url"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/subiz/header"
	"github.com/subiz/log"
	"google.golang.org/protobuf/proto"
)

var BACKEND = "https://api.subiz.com.vn/4.1/ai"

const Gpt_4o_mini = "gpt-4o-mini"
const Gpt_4o = "gpt-4o"
const Gpt_4_1_mini = "gpt-4.1-mini"
const Gpt_5_mini = "gpt-5-mini"
const Gpt_5_nano = "gpt-5-nano"
const Gpt_5 = "gpt-5"
const Gpt_4_1_nano = "gpt-4.1-nano"
const Gpt_4_1 = "gpt-4.1"

const Gemini_2_0_flash = "gemini-2.0-flash"
const Gemini_2_0_flash_lite = "gemini-2.0-flash-lite"
const Gemini_2_5_pro = "gemini-2.5-pro"
const Gemini_2_5_flash = "gemini-2.5-flash"
const Gemini_2_5_flash_lite = "gemini-2.5-flash-lite"

const Text_embedding_3_small = "text-embedding-3-small"
const Text_embedding_3_large = "text-embedding-3-large"
const Text_embedding_ada_002 = "text-embedding-ada-002"

const Gemini_embedding_001 = "gemini-embedding-001"

// ToModel converts model to the closest standardized model
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

	if model == "gpt-5" || strings.HasPrefix(model, "gpt-5-2") {
		return Gpt_5
	}

	if model == "gpt-5-mini" || strings.HasPrefix(model, "gpt-5-mini-2") {
		return Gpt_5_mini
	}

	if model == "gpt-5-nano" || strings.HasPrefix(model, "gpt-5-nano-2") {
		return Gpt_5_nano
	}

	// fallback for gpt
	if strings.HasPrefix(model, "gpt") {
		return Gpt_4o_mini
	}

	if model == "gemini-1.5-flash" {
		return Gemini_2_0_flash
	}

	if model == "gemini-2.0-flash-lite" {
		return Gemini_2_0_flash_lite
	}

	if model == "gemini-2.0-flash" {
		return Gemini_2_0_flash
	}

	if model == "gemini-2.5-flash-lite" {
		return Gemini_2_5_flash_lite
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

// ToGeminiModel converts a model name to its equivalent Gemini model.
func ToGeminiModel(model string) string {
	model = ToModel(model)

	if model == Gpt_4_1_nano || model == Gpt_5_nano {
		return Gemini_2_5_flash_lite
	}

	if model == Gpt_4o_mini || model == Gpt_5_mini {
		return Gemini_2_0_flash
	}

	if model == Gpt_4o || model == Gpt_4_1 || model == Gpt_5 {
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

// ToOpenAIModel converts a model name to its equivalent OpenAI model.
func ToOpenAIModel(model string) string {
	model = ToModel(model)

	if strings.HasPrefix(model, "gpt") {
		return model
	}

	if model == Gemini_2_0_flash_lite {
		return Gpt_4_1_nano
	}

	if model == Gemini_2_5_flash_lite {
		return Gpt_5_nano
	}

	if model == Gemini_2_0_flash {
		return Gpt_4o_mini
	}

	if model == Gemini_2_5_pro {
		return Gpt_4o
	}

	if model == Gemini_2_5_flash {
		return Gpt_4_1_mini
	}

	// fallback for gemini
	if strings.HasPrefix(model, "gemini") {
		return Gpt_4o_mini
	}

	return model
}

// GetFallbackChatModel returns a fallback model for a given model.
// If the model is a GPT model, it returns the equivalent Gemini model.
// If the model is a Gemini model, it returns the equivalent OpenAI model.
// Otherwise, it returns a default Gemini model.
func GetFallbackChatModel(model string) string {
	if strings.HasPrefix(model, "gpt") {
		return ToGeminiModel(model)
	}

	if strings.HasPrefix(model, "gemini") {
		return ToOpenAIModel(model)
	}

	return Gemini_2_0_flash
}

// per 1M tokens
var llmmodelinputprice = map[string]float64{
	"gpt-4.1":      2,
	"gpt-4.1-mini": 0.4,
	"gpt-4.1-nano": 0.1,
	"gpt-4o-mini":  0.15,
	"gpt-4o":       2.5,
	"gpt-5":        1.25,
	"gpt-5-mini":   0.25,
	"gpt-5-nano":   0.05,

	"gemini-2.0-flash":      0.1,
	"gemini-2.5-flash":      0.30,
	"gemini-2.0-flash-lite": 0.075,
	"gemini-2.5-flash-lite": 0.1,
	"gemini-2.5-pro":        2.5,
}

// per 1M tokens
var llmmodeloutputprice = map[string]float64{
	"gpt-4.1":      8,
	"gpt-4.1-mini": 1.6,
	"gpt-4.1-nano": 0.4,
	"gpt-4o":       10,
	"gpt-4o-mini":  0.6,
	"gpt-5":        10,
	"gpt-5-mini":   2,
	"gpt-5-nano":   0.4,

	"gemini-2.0-flash-lite": 0.3,
	"gemini-2.5-flash-lite": 0.4,
	"gemini-2.0-flash":      0.4,
	"gemini-2.5-flash":      2.5,
	"gemini-2.5-pro":        15,
}

// per 1M tokens
var llmmodelcachedprice = map[string]float64{
	"gpt-4.1":      8,
	"gpt-4.1-mini": 0.1,
	"gpt-4.1-nano": 0.025,
	"gpt-4o":       1.25,
	"gpt-4o-mini":  0.075,
	"gpt-5":        0.125,
	"gpt-5-mini":   0.025,
	"gpt-5-nano":   0.005,

	"gemini-2.0-flash":      0.1, // no caching
	"gemini-2.5-flash":      0.225,
	"gemini-2.0-flash-lite": 0.075, // no caching
	"gemini-2.5-flash-lite": 0.25,
	"gemini-2.5-pro":        11.25,
}

// OpenAIChatMessage mimics the structure of a message in an OpenAI Chat Completion request.
type OpenAIChatMessage struct {
	Role       string                 `json:"role"`
	Content    *string                `json:"content"` // Use pointer to allow for null content
	Name       string                 `json:"name,omitempty"`
	Contents   []OpenAIMessageContent `json:"contents,omitempty"` // Use pointer to allow for null content
	ToolCalls  []ToolCall             `json:"tool_calls,omitempty"`
	ToolCallId string                 `json:"tool_call_id,omitempty"`
	Refusal    string                 `json:"refusal,omitempty"`
}

type OpenAIMessageContentImageUrl struct {
	Url string `json:"url,omitempty"`
}

type OpenAIMessageContent struct {
	Text     string                        `json:"text,omitempty"`
	Type     string                        `json:"type,omitempty"` // text, image_url
	ImageUrl *OpenAIMessageContentImageUrl `json:"image_url,omitempty"`
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
	Type     string    `json:"type"`
	Function *Function `json:"function"`
	Output   string    `json:"output,omitempty"` // our fake
}

// Function mimics the structure of a function in an OpenAI Chat Completion request.
type Function struct {
	Name        string             `json:"name,omitempty"`
	Description string             `json:"description,omitempty"`
	Parameters  *header.JSONSchema `json:"parameters,omitempty"`

	Handler func(ctx context.Context, arg, callid string, ctxm map[string]any) (string, bool) `json:"-"` // "", true -> abandon completion, stop the flow immediately
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type       string                              `json:"type,omitempty"`
	JSONSchema *header.LLMResponseJSONSchemaFormat `json:"json_schema,omitempty"`
}

type OpenAIReasoning struct {
	Effort string `json:"effort,omitempty"`
}

// OpenAIChatRequest mimics the structure of an OpenAI Chat Completion request.
/*
type OpenAIChatRequest struct {
	Seed           int                 `json:"seed,omitempty"`
	Messages       []OpenAIChatMessage `json:"messages,omitempty"`
	Model          string              `json:"model,omitempty"`
	Temperature    float32             `json:"temperature,omitempty"`
	TopP           float32             `json:"top_p,omitempty"`
	Tools          []OpenAITool        `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat     `json:"response_format,omitempty"`
	ToolChoice     string              `json:"tool_choice,omitempty"`
	Reasoning      *OpenAIReasoning    `json:"reasoning,omitempty"`
  }
*/

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
	Code    string `json:"code,omitempty"`
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

func ChatCompleteAPI(ctx context.Context, payload []byte) (OpenAIChatResponse, error) {
	var output []byte
	var err error

	request := CompletionInput{}
	json.Unmarshal(payload, &request)
	model := request.Model

	if strings.HasPrefix(model, "gemini") {
		apikey := _geminiapikey
		output, err = chatCompleteGemini(ctx, apikey, request)
	} else {
		apikey := _openaiapikey
		output, err = chatCompleteChatGPT(ctx, apikey, request)
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

func GetEmbeddingAPI(ctx context.Context, model, text string) (OpenAIEmbeddingResponse, error) {
	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return OpenAIEmbeddingResponse{}, nil
	}
	if model == "gemini-embedding-001" {
		apikey := _geminiapikey
		return getGeminiEmbedding(ctx, apikey, model, text)
	}

	if model == Text_embedding_3_small || model == Text_embedding_3_large || model == Text_embedding_ada_002 {
		apikey := _openaiapikey
		return getOpenAIEmbedding(ctx, apikey, model, text)
	}
	return OpenAIEmbeddingResponse{
		Error: &OpenAIError{
			Message: "The model `" + model + "` does not exist or you do not have access to it.",
			Type:    "invalid_request_error",
			Code:    "model_not_found",
		},
	}, fmt.Errorf("model not found: %s", model)
}

var chatgpttimeouterr = []byte(`{"error": {"message": "Error: Timeout was reached","type": "timeout"}}`)

func chatCompleteChatGPT(ctx context.Context, apikey string, request CompletionInput) ([]byte, error) {
	model := request.Model
	if strings.HasPrefix(model, "gpt-5-") {
		// those models only support temperatture parameters = 1
		// https://community.openai.com/t/temperature-in-gpt-5-models/1337133/4
		request.Temperature = 1
	}
	var err error
	requestb, err := ToOpenAICompletionJSON(request)
	if err != nil {
		return nil, err
	}
	fmt.Println("OPENAIREQ", string(requestb), apikey)
	url := "https://api.openai.com/v1/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestb))
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
	Usage  *Usage                `json:"usage"`
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

// for testing only
func FakeBackend(rawURL string, input []byte) (int, []byte) {
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		return 400, []byte("invalid url")
	}

	if parsedURL.Host == "test" && parsedURL.Path == "/embeddings" {
		model := parsedURL.Query().Get("model")
		out, err := GetEmbeddingAPI(context.Background(), model, string(input))
		if err != nil {
			return 500, []byte(err.Error())
		}
		b, _ := json.Marshal(out)
		return 200, b
	}

	if parsedURL.Host == "test" && parsedURL.Path == "/completions" {
		request := CompletionInput{}
		json.Unmarshal(input, &request)
		out, err := ChatCompleteAPI(context.Background(), input)
		if err != nil {
			return 500, nil
		}
		b, _ := json.Marshal(out)
		return 200, b
	}

	return 404, []byte("not found " + rawURL)
}

type EmbeddingOutput struct {
	Text        string    `json:"text"`
	Vector      []float32 `json:"vector"`
	TotalTokens int64     `json:"total_tokens"`
	Created     int64     `json:"created"`
	DurationMs  int64     `json:"duration_ms"`
	KfpvCostUSD int64     `json:"kfpv_cost_usd"`
}

type CompletionOutput struct {
	Content           string `json:"content"`
	Refusal           string `json:"refusal"`
	Request           []byte `json:"request"`
	InputTokens       int64  `json:"input_tokens"`
	OutputTokens      int64  `json:"output_tokens"`
	InputCachedTokens int64  `json:"input_cached_tokens"`
	OuputCachedTokens int64  `json:"output_cached_tokens"`
	Created           int64  `json:"created"`
	DurationMs        int64  `json:"duration_ms"`
	KfpvCostUSD       int64  `json:"kfpv_cost_usd"` // 1 usd -> 1000_000_000 kfpvusd
}

var _apikey string // subiz api key
var _openaiapikey string
var _geminiapikey string

// Init setups the Subiz API Key client. Only client should call this
func Init(apikey string) {
	_apikey = apikey
}

// InitAPI setups API Keys for the server. Only server should call this
func InitAPI(geminiKey, openaiKey string) {
	_openaiapikey = openaiKey
	_geminiapikey = geminiKey
}

type TotalCost struct {
	USD int64 `json:"usd"` // 1000000000
}

type CompletionReasoning struct {
	Effort    string `json:"effort,omitempty"` // high, medium, low
	MaxTokens int    `json:"max_tokens,omitempty"`
}

type CompletionInput struct {
	Seed           int                           `json:"seed,omitempty"`
	Model          string                        `json:"model,omitempty"`
	NoLog          bool                          `json:"-,omitempty"` // disable log
	Instruct       string                        `json:"instruct,omitempty"`
	Messages       []*header.LLMChatHistoryEntry `json:"messages,omitempty"`
	ResponseFormat *ResponseFormat               `json:"response_format,omitempty"`
	ToolChoice     string                        `json:"tool_choice,omitempty"`
	Reasoning      *CompletionReasoning          `json:"reasoning,omitempty"`
	Temperature    float32                       `json:"temperature,omitempty"`
	TopP           float32                       `json:"top_p,omitempty"`
	Tools          []OpenAITool                  `json:"tools,omitempty"`
}

func Complete(ctx context.Context, input CompletionInput) (string, CompletionOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()
	var fnM = map[string]*Function{}
	for _, fn := range input.Tools {
		if fn.Type == "function" && fn.Function != nil {
			fnM[fn.Function.Name] = fn.Function
		}
	}

	messages := []*header.LLMChatHistoryEntry{}
	for _, msg := range input.Messages {
		msg = proto.Clone(msg).(*header.LLMChatHistoryEntry)
		msg.Id = ""
		msg.Created = 0
		messages = append(messages, msg)
	}
	input.Messages = messages

	functioncalled := false
	var requestbody []byte
	completion := &OpenAIChatResponse{}

	var totalCost float64
	tokenUsages := []*Usage{}

	for range 5 { // max 5 loops
		// Retrieve the value and assert it as a string
		accid, _ := ctx.Value("account_id").(string)
		convoid, _ := ctx.Value("conversation_id").(string)

		// send to subiz server
		q := neturl.Values{}
		if accid != "" {
			q.Set("account_id", accid)
		}
		if id, _ := ctx.Value("conversation_id").(string); id != "" {
			q.Set("x-conversation-id", id)
		}
		if id, _ := ctx.Value("trace_id").(string); id != "" {
			q.Set("x-trace-id", id)
		}
		if id, _ := ctx.Value("purpose").(string); id != "" {
			q.Set("x-purpose", id)
		}

		url := BACKEND + "/completions?" + q.Encode()
		requestbody, _ = json.Marshal(input)
		md5sum := GetMD5Hash(accid + "." + string(requestbody))
		cachepath := "./.cache/cc-" + md5sum
		cache, err := os.ReadFile(cachepath)
		if err != nil {
			if _, err := os.Stat("./.cache"); os.IsNotExist(err) {
				os.MkdirAll("./.cache", os.ModePerm)
			}
			_, err := os.Stat(cachepath)
			if err == nil || !os.IsNotExist(err) {
				log.Err(accid, err, "CANNOT CACHE")
			}
		}

		if !input.NoLog {
			cachehit := "HIT"
			if len(cache) == 0 {
				cachehit = "MISS"
			}
			log.Info(accid, log.Stack(), "LLM", cachehit, convoid, md5sum, string(requestbody))
		}

		if len(cache) > 0 {
			if err = json.Unmarshal(cache, completion); err != nil {
				cache = nil // invalid cache
			}
		}

		if len(cache) == 0 {
			resp, output, err := sendPOST(url, _apikey, requestbody)
			if err != nil {
				return "", CompletionOutput{}, log.EProvider(err, "openai", "completion")
			}
			if resp.StatusCode != 200 {
				return "", CompletionOutput{}, log.EProvider(nil, "openai", "completion", log.M{"status": resp.StatusCode, "_payload": output})
			}

			pricestr := resp.Header.Get("X-Cost-USD")
			pricef, _ := strconv.ParseFloat(pricestr, 64)
			totalCost += pricef
			json.Unmarshal(output, completion)
			os.WriteFile(cachepath, output, 0644)
		}

		if len(completion.Choices) == 0 {
			break
		}
		if completion.Usage != nil {
			tokenUsages = append(tokenUsages, completion.Usage)
		}
		if len(completion.Choices) == 0 {
			break
		}
		toolCalls := completion.Choices[0].Message.ToolCalls
		// Abort early if there are no tool calls
		if len(toolCalls) == 0 || functioncalled {
			break // only allow function call once
		}

		// If there is a was a function call, continue the conversation
		c0 := completion.Choices[0].Message
		c0content := ""
		if c0.Content != nil {
			c0content = *c0.Content
		}
		input.ToolChoice = "" // reset tool choice
		input.Messages = append(input.Messages, &header.LLMChatHistoryEntry{
			Role:       c0.Role,
			Content:    c0content,
			Name:       c0.Name,
			ToolCallId: c0.ToolCallId,
			Refusal:    c0.Refusal,
			ToolCalls:  toOurToolCalls(c0.ToolCalls),
		})
		for _, toolCall := range toolCalls {
			var output string
			var callid string
			var stop bool
			if fn := fnM[toolCall.Function.Name]; fn != nil {
				functioncalled = true
				callid = toolCall.ID
				output, stop = fn.Handler(ctx, toolCall.Function.Arguments, callid, nil)
			}

			// m := openai.ToolMessage(output, toolCall.ID)
			if stop {
				goto exit
			}
			input.Messages = append(input.Messages, &header.LLMChatHistoryEntry{
				Content:    output,
				Role:       "tool",
				ToolCallId: toolCall.ID,
			})
		}
	}

exit:
	completionoutput := CompletionOutput{
		Request:     requestbody,
		DurationMs:  time.Now().UnixMilli() - start.UnixMilli(),
		Created:     start.UnixMilli(),
		KfpvCostUSD: int64(totalCost * 1000),
	}

	for _, tokenUsage := range tokenUsages {
		completionoutput.InputTokens += tokenUsage.PromptTokens
		completionoutput.OutputTokens += tokenUsage.CompletionTokens
	}

	if len(completion.Choices) > 0 {
		completionoutput.Refusal = completion.Choices[0].Message.Refusal
		completionoutput.Content = completion.Choices[0].Message.GetContent()
	}

	if totalprice, _ := ctx.Value("total_cost").(*TotalCost); totalprice != nil {
		totalprice.USD += int64(totalCost * 1000)
	}
	return completionoutput.Content, completionoutput, nil
}

func GetEmbedding(ctx context.Context, model string, text string) ([]float32, EmbeddingOutput, error) {
	defer header.KLock(text)()

	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return nil, EmbeddingOutput{}, nil
	}

	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()
	response := &OpenAIEmbeddingResponse{}
	// Retrieve the value and assert it as a string
	accid, _ := ctx.Value("account_id").(string)
	convoid, _ := ctx.Value("conversation_id").(string)

	te := time.Now()

	// send to subiz server
	q := neturl.Values{}
	if accid != "" {
		q.Set("account_id", accid)
	}
	if id, _ := ctx.Value("conversation_id").(string); id != "" {
		q.Set("x-conversation-id", id)
	}
	if id, _ := ctx.Value("trace_id").(string); id != "" {
		q.Set("x-trace-id", id)
	}
	if id, _ := ctx.Value("purpose").(string); id != "" {
		q.Set("x-purpose", id)
	}

	q.Set("model", model)

	url := BACKEND + "/embeddings?" + q.Encode()
	md5sum := GetMD5Hash(text)
	cachepath := "./.cache/eb-" + md5sum
	cache, err := os.ReadFile(cachepath)
	if err != nil {
		if _, err := os.Stat("./.cache"); os.IsNotExist(err) {
			os.MkdirAll("./.cache", os.ModePerm)
		}
		_, err := os.Stat(cachepath)
		if err == nil || !os.IsNotExist(err) {
			log.Err(accid, err, "CANNOT CACHE")
		}
	}

	if len(cache) > 0 {
		output := EmbeddingOutput{}
		if err := json.Unmarshal(cache, &output); err == nil {
			return output.Vector, output, nil
		}
	}

	log.Info(accid, log.Stack(), "EMBEDDING", convoid, text, time.Since(te))
	resp, resoutput, err := sendPOST(url, _apikey, []byte(text))
	if err != nil {
		return nil, EmbeddingOutput{}, log.EProvider(err, "openai", "embedding")
	}
	if resp.StatusCode != 200 {
		return nil, EmbeddingOutput{}, log.EProvider(nil, "openai", "embedding", log.M{"status": resp.StatusCode, "_payload": resoutput})
	}

	json.Unmarshal(resoutput, response)
	pricestr := resp.Header.Get("X-Cost-USD") // fpv
	pricef, _ := strconv.ParseFloat(pricestr, 64)

	embeddingoutput := EmbeddingOutput{
		Text:        text,
		DurationMs:  time.Now().UnixMilli() - start.UnixMilli(),
		Created:     start.UnixMilli(),
		KfpvCostUSD: int64(pricef * 1000),
	}

	if len(response.Data) > 0 {
		embeddingoutput.Vector = response.Data[0].Embedding
	}

	if totalprice, _ := ctx.Value("total_cost").(*TotalCost); totalprice != nil {
		totalprice.USD += int64(pricef * 1000)
	}
	cache, _ = json.Marshal(embeddingoutput)
	os.WriteFile(cachepath, cache, 0644)
	return embeddingoutput.Vector, embeddingoutput, nil
}

func sendPOST(url, token string, payload []byte) (*http.Response, []byte, error) {
	if strings.HasPrefix(url, "https://test/") {
		status, body := FakeBackend(url, payload)
		return &http.Response{StatusCode: status}, body, nil
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payload))
	if err != nil {
		return nil, nil, err
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, nil, log.EProvider(err, "openai", "completion")
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	return resp, buf.Bytes(), nil
}

func GetMD5Hash(text string) string {
	hash := md5.Sum([]byte(text))
	return hex.EncodeToString(hash[:])
}

func addAdditionalProp(schema map[string]any) {
	propsi := schema["properties"]
	if propsi == nil {
		return
	}
	props, _ := propsi.(map[string]any)
	for _, propi := range props {
		if prop, _ := propi.(map[string]any); prop != nil {
			typi := prop["type"]
			if typi == nil {
				continue
			}
			typ := typi.(string)

			if typ == "array" {
				itemsi := prop["items"]
				if itemsi != nil {
					items := itemsi.(map[string]any)
					api := items["additionalProperties"]
					pass := false
					if api != nil {
						b, _ := api.(bool)
						if b {
							pass = true
						}
					}

					if !pass {
						items["additionalProperties"] = false
					}
					addAdditionalProp(items)
				}
			}
		}
	}
}

// badstring: UGjhuqduIGjGsOG7m25nIGThuqtuIGPDoGkgxJHhurd0IHThu7EgxJHhu5luZyBn4butaSBaTlMgdHLDqm4gU3ViaXouCiAzOkNo4buNbiBt4bqrdSB0aW4gWk5TW+KAi10oI2IlQzYlQjAlRTElQkIlOUJjLTMlRTElQkIlOERuLW0lRTElQkElQUJ1LXRpbi16bnMgIsSQxrDhu51uZyBk4bqrbiB0cuG7sWMgdGnhur9wIMSR4bq/biBixrDhu5tjLTPhu41uLW3huqt1LXRpbi16bnMiKQohW10oaHR0cHM6Ly92Y2RuLnN1Yml6LWNkbi5jb20vZmlsZS83ZjczNDllNWNjYTNmNzczYTlkODIwODIxZTg3ZmI1NTEwYzlhODgwNDc0MGQ0ZjMzOWY3YzlkMzdiN2IzNmZiX2FjcHhrZ3VtaWZ1b29mb29zYmxlKQojIyMgQsaw4bubYyA0OiBI4bq5biBs4buLY2ggZ+G7rWkgdOG7sSDEkeG7mW5nIFpOU1vigItdKCNiJUM2JUIwJUUxJUJCJTlCYy00LWglRTElQkElQjluLWwlRTElQkIlOEJjaC1nJUUxJUJCJUFEaS10JUUxJUJCJUIxLSVDNCU5MSVFMSVCQiU5OW5nLXpucyAixJDGsOG7nW5nIGThuqtuIHRy4buxYyB0aQDhur9w
func CleanString(str string) string {
	str = strings.Join(strings.Split(str, "\000"), "")
	return strings.ToValidUTF8(str, "")
}

type OpenAIJSONSchema struct {
	Title                string                       `json:"title,omitempty"`
	Type                 string                       `json:"type,omitempty"` // string, number, object, array, boolean, null
	Description          string                       `protobuf:"bytes,5,opt,name=description,proto3" json:"description,omitempty"`
	Properties           map[string]*OpenAIJSONSchema `protobuf:"bytes,6,rep,name=properties,proto3" json:"properties,omitempty" protobuf_key:"bytes,1,opt,name=key" proobuf_val:"bytes,2,opt,name=value"`
	Items                *OpenAIJSONSchema            `protobuf:"bytes,7,opt,name=items,proto3" json:"items,omitempty"` // used for type array
	MinItems             int64                        `protobuf:"varint,8,opt,name=minItems,proto3" json:"minItems,omitempty"`
	UniqueItems          bool                         `protobuf:"varint,9,opt,name=uniqueItems,proto3" json:"uniqueItems,omitempty"`
	ExclusiveMinimum     int64                        `protobuf:"varint,10,opt,name=exclusiveMinimum,proto3" json:"exclusiveMinimum,omitempty"`
	Required             []string                     `protobuf:"bytes,11,rep,name=required,proto3" json:"required,omitempty"`
	AdditionalProperties bool                         `json:"additionalProperties"` // chatgpt required this
	Enum                 []string                     `json:"enum,omitempty"`
}

func toOpenAISchema(h *header.JSONSchema) *OpenAIJSONSchema {
	if h == nil {
		return &OpenAIJSONSchema{}
	}

	p := &OpenAIJSONSchema{
		Title:                h.Title,
		Type:                 h.Type,
		Description:          h.Description,
		MinItems:             h.MinItems,
		UniqueItems:          h.UniqueItems,
		ExclusiveMinimum:     h.ExclusiveMinimum,
		Required:             h.Required,
		AdditionalProperties: h.AdditionalProperties,
		Enum:                 h.Enum,
	}

	if h.Items != nil {
		p.Items = toOpenAISchema(h.Items)
	}

	if len(h.Properties) > 0 {
		p.Properties = map[string]*OpenAIJSONSchema{}
	}
	for k, v := range h.Properties {
		p.Properties[k] = toOpenAISchema(v)
	}
	return p
}

func ToOpenAICompletionJSON(req CompletionInput) ([]byte, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	m := map[string]any{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}

	changed := false
	if _, hasDash := m["-"]; hasDash {
		changed = true
	}

	// tools
	tools := []any{}
	for _, tool := range req.Tools {
		if tool.Type == "function" && tool.Function != nil {
			newfn := map[string]any{
				"name":        tool.Function.Name,
				"description": tool.Function.Description,
			}
			newtool := map[string]any{
				"type":     "function",
				"function": newfn,
			}
			var params map[string]any
			if tool.Function.Parameters != nil {
				paramb, _ := json.Marshal(tool.Function.Parameters)

				params = map[string]any{}

				json.Unmarshal(paramb, &params)
				params["additionalProperties"] = tool.Function.Parameters.AdditionalProperties
				props := map[string]any{}
				for k, v := range tool.Function.Parameters.Properties {
					props[k] = toOpenAISchema(v)
				}
				if len(props) > 0 {
					params["properties"] = props
				}
				newfn["parameters"] = params
			}
			tools = append(tools, newtool)
			continue
		}
		tools = append(tools, tool)
		continue
	}

	if len(tools) > 0 {
		changed = true
		delete(m, "tools")
		m["tools"] = tools
	}

	if req.ResponseFormat != nil && req.ResponseFormat.JSONSchema != nil {
		delete(m, "response_format")
		changed = true
		m["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   req.ResponseFormat.JSONSchema.GetName(),
				"strict": req.ResponseFormat.JSONSchema.GetStrict(),
				"schema": toOpenAISchema(req.ResponseFormat.JSONSchema.GetSchema()),
			},
		}
	}

	messagesi := m["messages"]
	if messagesi == nil {
		messagesi = []any{}
	}

	messages, ok := messagesi.([]any)
	if !ok {
		return b, nil
	}

	for _, msgi := range messages {
		msg, ok := msgi.(map[string]any)
		if !ok {
			return b, nil
		}

		contentsi, has := msg["contents"]
		if has {
			if arr, ok := contentsi.([]any); ok {
				if len(arr) > 0 {
					msg["content"] = contentsi
				}
			}
			delete(msg, "contents")
			changed = true
		}
		// tool
		if _, has := msg["content"]; !has {
			if _, hastool := msg["tool_calls"]; !hastool {
				msg["content"] = "" // fill default, this is not valid 	{ "role": "tool", "tool_call_id": "call_DYllFTrudNWsPNg4jKhsdEaw" }
				changed = true
			}
		}
	}

	if req.Instruct != "" {
		messages = append([]any{&header.LLMChatHistoryEntry{Role: "system", Content: req.Instruct}}, messages...)
		m["messages"] = messages
		changed = true
	}

	if changed {
		delete(m, "-")
		delete(m, "instruct")
		b, _ = json.Marshal(m)
	}
	return b, nil
}

func toOurToolCalls(toolcalls []ToolCall) []*header.LLMToolCall {
	if len(toolcalls) == 0 {
		return nil
	}

	calls := []*header.LLMToolCall{}
	for _, call := range toolcalls {
		b, _ := json.Marshal(call)
		ourcall := &header.LLMToolCall{}
		json.Unmarshal(b, ourcall)
		calls = append(calls, ourcall)
	}
	return calls
}

// GeminiRequest represents the structure of a request to the Gemini API.
// https://ai.google.dev/api/generate-content#v1beta.GenerationConfig
type GeminiRequest struct {
	Model             string                  `json:"model"`
	SystemInstruction *GeminiContent          `json:"systemInstruction,omitempty"`
	Contents          []*GeminiContent        `json:"contents"`
	Tools             []*GeminiTool           `json:"tools,omitempty"`
	GenerationConfig  *GeminiGenerationConfig `json:"generationConfig,omitempty"`
}

// GeminiContent represents a content block in a Gemini request.
type GeminiContent struct {
	Parts []*GeminiPart `json:"parts"`
	Role  string        `json:"role,omitempty"`
}

// GeminiPart represents a part of a content block.
type GeminiPart struct {
	Text             *string                 `json:"text,omitempty"`
	FunctionCall     *GeminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *GeminiFunctionResponse `json:"functionResponse,omitempty"`
	InlineData       *GeminiPartInlineData   `json:"inline_data,omitempty"`
}

type GeminiPartInlineData struct {
	MimeType string `json:"mime_type,omitempty"`
	Data     string `json:"data,omitempty"`
}

// https://ai.google.dev/api/generate-content#ThinkingConfig
type GeminiThinkingConfig struct {
	// The number of thoughts tokens that the model should generate.
	ThinkingBudget int `json:"thinkingBudget,omitempty"`

	// Indicates whether to include thoughts in the response. If true, thoughts are returned only when available.
	IncludeThoughts bool `json:"includeThoughts"`
}

// GeminiGenerationConfig represents the generation configuration.
type GeminiGenerationConfig struct {
	StopSequences []string `json:"stopSequences,omitempty"`
	// CandidateCount   int                   `json:"candidateCount,omitempty"`
	MaxOutputTokens  int                   `json:"maxOutputTokens,omitempty"`
	Temperature      float32               `json:"temperature,omitempty"`
	TopP             float32               `json:"topP,omitempty"`
	Seed             int                   `json:"seed,omitempty"`
	ResponseMIMEType string                `json:"responseMimeType,omitempty"`
	ResponseSchema   *header.JSONSchema    `json:"responseSchema,omitempty"`
	ThinkingConfig   *GeminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

// GeminiTool is a local struct to avoid genai dependency.
type GeminiTool struct {
	FunctionDeclarations []*GeminiFunctionDeclaration `json:"functionDeclarations"`
}

// GeminiFunctionDeclaration is a local struct to avoid genai dependency.
type GeminiFunctionDeclaration struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  *header.JSONSchema `json:"parameters"`
}

// GeminiFunctionCall is a local struct to avoid genai dependency.
type GeminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// GeminiFunctionResponse is a local struct to avoid genai dependency.
type GeminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

func toGeminiSchema(schema *header.JSONSchema) *header.JSONSchema {
	if schema == nil {
		return nil
	}
	geminiSchema := &header.JSONSchema{
		Type:        schema.Type,
		Description: schema.Description,
		Required:    schema.Required,
	}
	if len(schema.Properties) > 0 {
		geminiSchema.Properties = make(map[string]*header.JSONSchema)
		for key, prop := range schema.Properties {
			geminiSchema.Properties[key] = toGeminiSchema(prop)
		}
	}
	if schema.Items != nil {
		geminiSchema.Items = toGeminiSchema(schema.Items)
	}
	return geminiSchema
}

// ToGeminiRequestJSON converts an OpenAIChatRequest to a Gemini-compatible JSON request string.
func ToGeminiRequestJSON(req CompletionInput) ([]byte, error) {
	var geminiTools []*GeminiTool
	for _, tool := range req.Tools {
		decl := &GeminiFunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		if tool.Function.Parameters != nil {
			properties := make(map[string]*header.JSONSchema)
			for key, prop := range tool.Function.Parameters.Properties {
				properties[key] = &header.JSONSchema{
					Type:        prop.Type,
					Description: prop.Description,
				}
			}

			decl.Parameters = &header.JSONSchema{
				Type:       "object",
				Properties: properties,
				Required:   tool.Function.Parameters.Required,
			}
		}

		geminiTools = append(geminiTools, &GeminiTool{FunctionDeclarations: []*GeminiFunctionDeclaration{decl}})
	}

	geminiReq := GeminiRequest{
		Model: "models/" + ToGeminiModel(req.Model),
		GenerationConfig: &GeminiGenerationConfig{
			Seed:        req.Seed,
			Temperature: req.Temperature,
			TopP:        req.TopP,
		},
		Tools: geminiTools,
	}

	if req.Reasoning != nil {
		budget := 0
		switch req.Reasoning.Effort {
		case "low":
			budget = 256
		case "medium":
			budget = 1024
		case "high":
			budget = 4096
		}

		if budget > 0 {
			geminiReq.GenerationConfig.ThinkingConfig = &GeminiThinkingConfig{
				ThinkingBudget: budget,
			}
		}
	}

	if req.ResponseFormat != nil && req.ResponseFormat.JSONSchema != nil {
		geminiReq.GenerationConfig.ResponseMIMEType = "application/json"
		geminiReq.GenerationConfig.ResponseSchema = toGeminiSchema(req.ResponseFormat.JSONSchema.Schema)
	}

	var contents []*GeminiContent
	toolCallsByID := make(map[string]*header.LLMToolCall)

	messages := req.Messages
	if req.Instruct != "" {
		messages = append([]*header.LLMChatHistoryEntry{{Role: "system", Content: req.Instruct}}, messages...)
	}

	systemmsgs := []string{}
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			if msg.Content != "" {
				systemmsgs = append(systemmsgs, msg.Content)
			}
		case "user":
			gemContent := &GeminiContent{Role: "user"}
			if len(msg.Contents) == 0 {
				gemContent.Parts = []*GeminiPart{{Text: &msg.Content}}
			} else {
				for _, content := range msg.Contents {
					part := &GeminiPart{}
					if content.Type == "" || content.Type == "text" {
						text := content.Text
						part.Text = &text
					}
					if content.Type == "image_url" {
						url := ""
						if content.ImageUrl != nil {
							url = content.ImageUrl.Url
						}

						if !strings.HasPrefix(url, "data:image/") {
							continue
						}

						url = strings.TrimPrefix(url, "data:")
						urls := strings.Split(url, ";")
						if len(urls) < 2 {
							continue
						}
						part.InlineData = &GeminiPartInlineData{}
						part.InlineData.MimeType = urls[0]
						part.InlineData.Data = strings.TrimPrefix(urls[1], "base64,")
					}
					gemContent.Parts = append(gemContent.Parts, part)
				}
			}
			contents = append(contents, gemContent)

		case "assistant":
			if len(msg.ToolCalls) > 0 {
				var parts []*GeminiPart
				for _, tc := range msg.ToolCalls {
					toolCallsByID[tc.Id] = tc
					var args map[string]interface{}
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
						return nil, fmt.Errorf("failed to unmarshal tool call arguments: %w", err)
					}
					parts = append(parts, &GeminiPart{
						FunctionCall: &GeminiFunctionCall{
							Name: tc.Function.Name,
							Args: args,
						},
					})
				}
				contents = append(contents, &GeminiContent{Parts: parts, Role: "model"})
			} else if msg.Content != "" {
				contents = append(contents, &GeminiContent{
					Parts: []*GeminiPart{{Text: &msg.Content}},
					Role:  "model",
				})
			}
		case "tool":
			if toolCall, ok := toolCallsByID[msg.ToolCallId]; ok {
				contents = append(contents, &GeminiContent{
					Role: "user", // In Gemini, the function response is from the user
					Parts: []*GeminiPart{{
						FunctionResponse: &GeminiFunctionResponse{
							Name:     toolCall.Function.Name,
							Response: map[string]any{"text": msg.Content},
						},
					}},
				})
			}
		}
	}

	if len(contents) == 0 {
		contents = []*GeminiContent{{Role: "user", Parts: strsToParts(systemmsgs)}}
		systemmsgs = nil
	}

	if len(systemmsgs) > 0 {
		geminiReq.SystemInstruction = &GeminiContent{Parts: strsToParts(systemmsgs)}
	}

	// inject user or gemini will return
	// Please ensure that function call turn comes immediately after a user turn or after a function response turn.
	newcontents := []*GeminiContent{}
	for i, msg := range contents {
		if msg.Role == "model" {
			for _, part := range msg.Parts {
				if part.FunctionCall != nil {
					if i == 0 || contents[i-1].Role != "user" {
						newcontents = append(newcontents, &GeminiContent{Role: "user"})
					}
				}
			}
		}
		newcontents = append(newcontents, msg)
	}
	contents = newcontents

	if len(contents) > 0 {
		// make sure the last content must be user message
		// or gemini will emit
		// {
		//   "error": {
		//     "code": 400,
		//     "message": "Please ensure that single turn requests end with a user role or the role field is empty.",
		//     "status": "INVALID_ARGUMENT"
		//   }
		// }

		lastcontent := contents[len(contents)-1]
		if lastcontent.Role != "user" {
			contents = append(contents, &GeminiContent{Parts: strsToParts([]string{""}), Role: "user"})
		}

		geminiReq.Contents = contents
	}

	b, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Gemini request: %w", err)
	}
	return b, nil
}

// GeminiErrorDetail represents the details of a Gemini API error.
type GeminiErrorDetail struct {
	Type            string `json:"@type"`
	FieldViolations []struct {
		Field       string `json:"field"`
		Description string `json:"description"`
	} `json:"fieldViolations"`
}

// GeminiError represents a Gemini API error.
type GeminiError struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Status  string              `json:"status"`
	Details []GeminiErrorDetail `json:"details"`
}

// LocalCandidate mirrors genai.Candidate
type LocalCandidate struct {
	Content      *LocalContent `json:"content"`
	FinishReason string        `json:"finishReason"` // Using string instead of enum
}

// LocalContent mirrors genai.Content
type LocalContent struct {
	Parts []*LocalPart `json:"parts"`
	Role  string       `json:"role"`
}

// LocalPart is a struct to unmarshal different part types from JSON.
type LocalPart struct {
	Text         *string             `json:"text,omitempty"`
	FunctionCall *GeminiFunctionCall `json:"functionCall,omitempty"`
}

// LocalUsageMetadata mirrors genai.UsageMetadata
type LocalUsageMetadata struct {
	PromptTokenCount        int32 `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount    int32 `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount         int32 `json:"totalTokenCount,omitempty"`
	CachedContentTokenCount int32 `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      int32 `json:"thoughtsTokenCount,omitempty"`
}

// GeminiAPIResponse is a wrapper for the Gemini API response.
type GeminiAPIResponse struct {
	Candidates    []*LocalCandidate   `json:"candidates"`
	UsageMetadata *LocalUsageMetadata `json:"usageMetadata"`
	ModelVersion  string              `json:"modelVersion"`
	ResponseId    string              `json:"responseId"`
	Error         *GeminiError        `json:"error"`
}

// toOpenAIChatResponse converts a Gemini response to an OpenAI-compatible chat response.
func toOpenAIChatResponse(res *GeminiAPIResponse) (*OpenAIChatResponse, error) {
	if res.Error != nil {
		param := ""
		if len(res.Error.Details) > 0 && len(res.Error.Details[0].FieldViolations) > 0 {
			param = res.Error.Details[0].FieldViolations[0].Field
		}
		return &OpenAIChatResponse{
			Error: &OpenAIError{
				Message: res.Error.Message,
				Type:    "invalid_request_error",
				Param:   param,
			},
		}, nil
	}
	if res == nil {
		return nil, fmt.Errorf("empty response from Gemini")
	}
	responseID := res.ResponseId
	choice := OpenAIChoice{Index: 0}
	if len(res.Candidates) > 0 {
		candidate := res.Candidates[0]
		if candidate.FinishReason != "STOP" {
			finishReasonText := fmt.Sprintf("Response stopped due to: %s", candidate.FinishReason)
			choice.Message = OpenAIChatMessage{
				Role:    "assistant",
				Content: &finishReasonText,
			}
			choice.FinishReason = "stop"
		} else {
			if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
				emptyContent := ""
				choice.Message = OpenAIChatMessage{
					Role:    "assistant",
					Content: &emptyContent,
				}
				choice.FinishReason = "stop"
			} else {
				var responseText string
				var toolCalls []ToolCall

				for i, part := range candidate.Content.Parts {
					if part.Text != nil {
						responseText += *part.Text
					} else if part.FunctionCall != nil {
						argsJSON, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							return nil, fmt.Errorf("failed to marshal function call arguments: %w", err)
						}
						toolCalls = append(toolCalls, ToolCall{
							ID:   fmt.Sprintf("call_%s_%d", responseID, i),
							Type: "function",
							Function: ToolFunction{
								Name:      part.FunctionCall.Name,
								Arguments: string(argsJSON),
							},
						})
					}
				}

				if len(toolCalls) > 0 {
					choice.Message = OpenAIChatMessage{
						Role:      "assistant",
						Content:   nil,
						ToolCalls: toolCalls,
					}
					choice.FinishReason = "tool_calls"
				} else {
					choice.Message = OpenAIChatMessage{
						Role:    "assistant",
						Content: &responseText,
					}
					choice.FinishReason = "stop"
				}
			}
		}
	}

	var usage *Usage

	if res.UsageMetadata != nil {
		usage = &Usage{
			PromptTokens:     int64(res.UsageMetadata.PromptTokenCount),
			CompletionTokens: int64(res.UsageMetadata.CandidatesTokenCount + res.UsageMetadata.ThoughtsTokenCount),
			TotalTokens:      int64(res.UsageMetadata.TotalTokenCount),
			PromptTokensDetails: &PromptTokensDetails{
				CachedTokens: int64(res.UsageMetadata.CachedContentTokenCount),
			},
		}
	}

	return &OpenAIChatResponse{
		ID:      responseID,
		Created: time.Now().Unix(),
		Object:  "chat.completion",
		Model:   res.ModelVersion,
		Choices: []OpenAIChoice{choice},
		Usage:   usage,
	}, nil
}

func chatCompleteGemini(ctx context.Context, apikey string, request CompletionInput) ([]byte, error) {
	model := request.Model
	var err error
	requestb, err := ToGeminiRequestJSON(request)
	if err != nil {
		return nil, err
	}

	fmt.Println("GEMINIREQ", string(requestb), apikey)
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + ToGeminiModel(model) + ":generateContent"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestb))
	if err != nil {
		return nil, log.EServer(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", apikey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	output := buf.Bytes()
	gemres := &GeminiAPIResponse{}
	json.Unmarshal(output, gemres)

	out, err := toOpenAIChatResponse(gemres)
	if err != nil {
		return nil, err
	}
	return json.Marshal(out)
}

func strsToParts(strs []string) []*GeminiPart {
	out := []*GeminiPart{}
	for _, str := range strs {
		out = append(out, &GeminiPart{Text: &str})
	}
	return out
}

// EmbeddingContent represents the content for an embedding request.
type EmbeddingContent struct {
	Parts []EmbeddingPart `json:"parts"`
}

// EmbeddingPart represents a part of the content for an embedding request.
type EmbeddingPart struct {
	Text string `json:"text"`
}

// EmbeddingRequest represents the request to the embedding model.
type EmbeddingRequest struct {
	Model   string           `json:"model"`
	Content EmbeddingContent `json:"content"`
}

// GeminiEmbeddingResponse represents the response from the embedding model.
type GeminiEmbeddingResponse struct {
	Embedding *Embedding `json:"embedding"`
}

// Embedding represents the embedding values.
type Embedding struct {
	Values []float32 `json:"values"`
}

// GetEmbedding takes a text and returns its embedding.
// It uses the 'embedding-001' model.
func getGeminiEmbedding(ctx context.Context, apiKey, model, text string) (OpenAIEmbeddingResponse, error) {
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=%s", apiKey)
	var out OpenAIEmbeddingResponse
	reqBody := EmbeddingRequest{
		Model: model, // "models/embedding-001",
		Content: EmbeddingContent{
			Parts: []EmbeddingPart{{Text: text}},
		},
	}

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
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		out.Error = &OpenAIError{Message: fmt.Sprintf("error making request: %s", err.Error())}
		return out, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		out.Error = &OpenAIError{Message: fmt.Sprintf("error making request: %d", resp.StatusCode)}
		return out, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var embeddingResp GeminiEmbeddingResponse
	bodyBytes, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(bodyBytes, &embeddingResp); err != nil {
		out.Error = &OpenAIError{Message: fmt.Sprintf("error decoding response body: %s", err.Error())}
		return out, fmt.Errorf("error decoding response body: %w", err)
	}

	out.Object = "list"
	out.Model = model
	out.Data = []OpenAIEmbeddingData{{Object: "embedding", Index: 0}}
	out.Usage = &Usage{
		PromptTokens: int64(1 + utf8.RuneCountInString(text)/4), // estimate number of tokens
		TotalTokens:  int64(1 + utf8.RuneCountInString(text)/4), // estimate number of tokens
	}

	var values []float32
	if embeddingResp.Embedding != nil {
		values = embeddingResp.Embedding.Values
	}

	out.Data[0].Embedding = values
	return out, nil
}
