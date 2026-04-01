package airouter

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"math"
	"net/http"
	neturl "net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/subiz/executor/v2"
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
const Gpt_5_4_nano = "gpt-5.4-nano"
const Gpt_5_4_mini = "gpt-5.4-mini"
const O3 = "o3"
const O4_mini = "o4-mini"

const Gemini_2_5_pro = "gemini-2.5-pro"
const Gemini_2_5_flash = "gemini-2.5-flash"
const Gemini_2_5_flash_lite = "gemini-2.5-flash-lite"

const Gemini_3_1_flash_lite = "gemini-3.1-flash-lite-preview"

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

	if model == "gpt-5.4-nano" || strings.HasPrefix(model, "gpt-5.4-nano-2") {
		return Gpt_5_4_nano
	}

	if model == "gpt-5.4-mini" || strings.HasPrefix(model, "gpt-5.4-mini-2") {
		return Gpt_5_4_mini
	}

	if model == O3 || strings.HasPrefix(model, "o3-2") {
		return O3
	}

	if model == O4_mini || strings.HasPrefix(model, "o4-mini-2") {
		return O4_mini
	}

	// fallback for gpt
	if strings.HasPrefix(model, "gpt") {
		return Gpt_4o_mini
	}

	if model == "gemini-1.5-flash" {
		return Gemini_3_1_flash_lite
	}

	if model == "gemini-2.0-flash-lite" {
		return Gemini_2_5_flash_lite
	}

	if model == "gemini-3.1-flash-lite" || model == "gemini-3.1-flash-lite-preview" {
		return Gemini_3_1_flash_lite
	}

	if model == "gemini-2.0-flash" {
		return Gemini_2_5_flash_lite
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
		return Gemini_2_5_flash_lite
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
		return Gemini_3_1_flash_lite
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

	if model == Gemini_2_5_flash_lite {
		return Gpt_5_nano
	}

	if model == Gemini_3_1_flash_lite {
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

	return Gemini_3_1_flash_lite
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
	"gpt-5.4-nano": 0.2,
	"gpt-5.4-mini": 0.75,

	"gemini-2.0-flash":              0.1,
	"gemini-2.5-flash":              0.3,
	"gemini-3-flash":                0.5,
	"gemini-2.0-flash-lite":         0.075,
	"gemini-2.5-flash-lite":         0.1,
	"gemini-3.1-flash-lite":         0.25,
	"gemini-3.1-flash-lite-preview": 0.25,
	"gemini-2.5-pro":                2.5,
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
	"gpt-5.4-nano": 1.25,
	"gpt-5.4-mini": 4.5,

	"gemini-2.0-flash-lite":         0.3,
	"gemini-2.5-flash-lite":         0.4,
	"gemini-3.1-flash-lite":         1.5,
	"gemini-3.1-flash-lite-preview": 1.5,
	"gemini-2.0-flash":              0.4,
	"gemini-2.5-flash":              2.5,
	"gemini-2.5-pro":                15,
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
	"gpt-5.4-nano": 0.02,
	"gpt-5.4-mini": 0.075,

	"gemini-2.0-flash": 0.1, // no caching
	"gemini-2.5-flash": 0.225,

	"gemini-2.0-flash-lite":         0.075, // no caching
	"gemini-2.5-flash-lite":         0.25,
	"gemini-3.1-flash-lite":         0.025,
	"gemini-3.1-flash-lite-preview": 0.025,

	"gemini-2.5-pro": 11.25,
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

	Handler func(ctx context.Context, arg, callid string, ctxm map[string]any) string `json:"-"` // "", true -> abandon completion
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type       string                              `json:"type,omitempty"`
	JSONSchema *header.LLMResponseJSONSchemaFormat `json:"json_schema,omitempty"`
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

	KFpvCostUSD int64 `json:"kfpv_cost_usd"` // our fields not openai
	CostVND     int64 `json:"cost_vnd"`      // our fields not openai
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

// return 1000 usd fpv
func CalculateCost(model string, usage *Usage, service_tier string) int64 {
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
	cost := 1000 * (float64(inputtoken)*llmmodelinputprice[ToModel(model)] +
		float64(cachedtoken)*llmmodelcachedprice[ToModel(model)] +
		float64(outputtoken)*llmmodeloutputprice[ToModel(model)])
	if service_tier == "flex" {
		cost = cost / 2
	}
	return int64(math.Ceil(cost))
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

// Init setups the Subiz API Key client. Only client should call this
func Init(apikey string) {
	_apikey = apikey
}

type TotalCost struct {
	USD int64 `json:"usd"` // kfpvusd = usd*1_000_000_000 = fpvusd * 1000
}

type CompletionReasoning struct {
	Effort    string `json:"effort,omitempty"` // high, medium, low
	MaxTokens int    `json:"max_tokens,omitempty"`
}

type CompletionInput struct {
	Seed                 int                           `json:"seed,omitempty"`
	PromptCacheKey       string                        `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string                        `json:"prompt_cache_retention,omitempty"`
	Verbosity            string                        `json:"verbosity,omitempty"`
	Stop                 []string                      `json:"stop,omitempty"`
	Model                string                        `json:"model,omitempty"`
	NoLog                bool                          `json:"-"` // disable log
	Instruct             string                        `json:"instruct,omitempty"`
	Messages             []*header.LLMChatHistoryEntry `json:"messages,omitempty"`
	MaxCompletionTokens  int                           `json:"max_completion_tokens,omitempty"`
	ResponseFormat       *ResponseFormat               `json:"response_format,omitempty"`
	ToolChoice           string                        `json:"tool_choice,omitempty"`
	Reasoning            *CompletionReasoning          `json:"reasoning,omitempty"`
	ReasoningEffort      string                        `json:"reasoning_effort,omitempty"`
	Temperature          float32                       `json:"temperature,omitempty"`
	TopP                 float32                       `json:"top_p,omitempty"`
	Tools                []OpenAITool                  `json:"tools,omitempty"`
	ServiceTier          string                        `json:"service_tier,omitempty"` // [auto, default], flex, priority, scale
	StopAfterToolCalled  bool                          `json:"stop_after_tool_called"`
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

	var requestbody []byte
	completion := &OpenAIChatResponse{}

	var totalCost float64
	tokenUsages := []*Usage{}

	turn := 0
	for range 8 { // max 5 loops
		if turn >= 6 {
			// no more function calls
			input.Tools = nil
		}
		turn++

		// Retrieve the value and assert it as a string
		accid, _ := ctx.Value("account_id").(string)
		convoid, _ := ctx.Value("conversation_id").(string)

		// send to subiz server
		q := neturl.Values{}
		if accid != "" {
			q.Set("account_id", accid)
		}
		if convoid != "" {
			q.Set("x-conversation-id", convoid)
		}
		if val, _ := ctx.Value("trace_id").(string); val != "" {
			q.Set("x-trace-id", val)
		}
		if val, _ := ctx.Value("purpose").(string); val != "" {
			q.Set("x-purpose", val) //
		}
		if val, _ := ctx.Value("is-ai-msg").(string); val != "" {
			q.Set("is-ai-msg", val)
		}
		if val, _ := ctx.Value("source_id").(string); val != "" {
			q.Set("x-source-id", val)
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
			completion = &OpenAIChatResponse{}
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
			completion = &OpenAIChatResponse{}
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
		// If there is a was a function call, continue the conversation
		toolCalls := completion.Choices[0].Message.ToolCalls
		// Abort early if there are no tool calls
		if len(toolCalls) == 0 {
			break // only allow function call once
		}
		input.ToolChoice = "" // reset tool choice
		c0 := completion.Choices[0].Message
		input.Messages = append(input.Messages, &header.LLMChatHistoryEntry{
			Role:       c0.Role,
			Name:       c0.Name,
			ToolCallId: c0.ToolCallId,
			Refusal:    c0.Refusal,
			ToolCalls:  toOurToolCalls(c0.ToolCalls),
		})

		tooloutmsgs := []*header.LLMChatHistoryEntry{}
		executor.Async(len(toolCalls), func(i int, lock *sync.Mutex) {
			toolCall := toolCalls[i]
			fn := fnM[toolCall.Function.Name]
			if fn == nil {
				return
			}
			callid := toolCall.ID
			output := fn.Handler(ctx, toolCall.Function.Arguments, callid, nil)

			lock.Lock()
			tooloutmsgs = append(tooloutmsgs, &header.LLMChatHistoryEntry{
				Content:    output,
				Role:       "tool",
				ToolCallId: callid,
			})
			lock.Unlock()

		}, 5)

		muststop := false
		if len(tooloutmsgs) > 0 {
			if input.StopAfterToolCalled {
				muststop = true
			}
			sort.Slice(tooloutmsgs, func(i, j int) bool {
				if tooloutmsgs[i].ToolCallId == tooloutmsgs[j].ToolCallId {
					return tooloutmsgs[i].Content < tooloutmsgs[j].Content
				}
				return tooloutmsgs[i].ToolCallId < tooloutmsgs[j].ToolCallId
			}) // make the order determistic (better caching)
			input.Messages = append(input.Messages, tooloutmsgs...)
		}
		if muststop {
			break
		}
	}

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
	if val, _ := ctx.Value("is-ai-msg").(string); val != "" {
		q.Set("is-ai-msg", val)
	}
	if val, _ := ctx.Value("source_id").(string); val != "" {
		q.Set("x-source-id", val)
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

// badstring: UGjhuqduIGjGsOG7m25nIGThuqtuIGPDoGkgxJHhurd0IHThu7EgxJHhu5luZyBn4butaSBaTlMgdHLDqm4gU3ViaXouCiAzOkNo4buNbiBt4bqrdSB0aW4gWk5TW+KAi10oI2IlQzYlQjAlRTElQkIlOUJjLTMlRTElQkIlOERuLW0lRTElQkElQUJ1LXRpbi16bnMgIsSQxrDhu51uZyBk4bqrbiB0cuG7sWMgdGnhur9wIMSR4bq/biBixrDhu5tjLTPhu41uLW3huqt1LXRpbi16bnMiKQohW10oaHR0cHM6Ly92Y2RuLnN1Yml6LWNkbi5jb20vZmlsZS83ZjczNDllNWNjYTNmNzczYTlkODIwODIxZTg3ZmI1NTEwYzlhODgwNDc0MGQ0ZjMzOWY3YzlkMzdiN2IzNmZiX2FjcHhrZ3VtaWZ1b29mb29zYmxlKQojIyMgQsaw4bubYyA0OiBI4bq5biBs4buLY2ggZ+G7rWkgdOG7sSDEkeG7mW5nIFpOU1vigItdKCNiJUM2JUIwJUUxJUJCJTlCYy00LWglRTElQkElQjluLWwlRTElQkIlOEJjaC1nJUUxJUJCJUFEaS10JUUxJUJCJUIxLSVDNCU5MSVFMSVCQiU5OW5nLXpucyAixJDGsOG7nW5nIGThuqtuIHRy4buxYyB0aQDhur9w
func CleanString(str string) string {
	str = strings.Join(strings.Split(str, "\000"), "")
	return strings.ToValidUTF8(str, "")
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

// Embedding represents the embedding values.
type Embedding struct {
	Values []float32 `json:"values"`
}

type RerankingResponse struct {
	Records []*RerankRecord `json:"records,omitempty"`
	Created int64           `json:"created,omitempty"` // sec
	Object  string          `json:"object,omitempty"`
	Model   string          `json:"model,omitempty"`
	Usage   *Usage          `json:"usage,omitempty"`
	Error   *OpenAIError    `json:"error,omitempty"`
}

type RerankOutput struct {
	Records     []*RerankRecord `json:"records,omitempty"`
	Created     int64           `json:"created,omitempty"`
	DurationMs  int64           `json:"duration_ms"`
	KfpvCostUSD int64           `json:"kfpv_cost_usd"` // 1 usd -> 1000_000_000 kfpvusd
}

type RerankRecord struct {
	Id      string  `json:"id,omitempty"`
	Title   string  `json:"title,omitempty"`
	Content string  `json:"content,omitempty"`
	Score   float32 `json:"score,omitempty"`
}

type RerankInput struct {
	Seed    int             `json:"seed,omitempty"`
	TopN    int             `json:"top_n,omitempty"`
	Model   string          `json:"model,omitempty"`
	Query   string          `json:"query,omitempty"`
	Records []*RerankRecord `json:"records,omitempty"`
}

func Rerank(ctx context.Context, model, query string, inrecords []*RerankRecord) (RerankOutput, error) {
	records := append([]*RerankRecord{}, inrecords...)
	// keeping thing determistic
	sort.Slice(records, func(i, j int) bool {
		titlei, titlej := records[i].Title, records[j].Title
		if titlei < titlej {
			return titlei < titlej
		}
		conti, contj := records[i].Content, records[j].Content
		return conti < contj
	})

	query = header.Norm(query, 1000)
	defer header.KLock(query)()

	if len(query) == 0 || len(records) == 0 {
		return RerankOutput{}, nil
	}

	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()
	response := &RerankingResponse{}
	// Retrieve the value and assert it as a string
	accid, _ := ctx.Value("account_id").(string)

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
	if val, _ := ctx.Value("is-ai-msg").(string); val != "" {
		q.Set("is-ai-msg", val)
	}

	q.Set("model", model)

	url := BACKEND + "/rerankings?" + q.Encode()
	rerankInput := RerankInput{
		Model:   model,
		Query:   query,
		Records: records,
	}
	payload, err := json.Marshal(rerankInput)
	if err != nil {
		return RerankOutput{}, err
	}
	md5sum := GetMD5Hash(string(payload))

	cachepath := "./.cache/rk-" + md5sum
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
		output := RerankOutput{}
		if err := json.Unmarshal(cache, &output); err == nil {
			return output, nil
		}
	}

	resp, resoutput, err := sendPOST(url, _apikey, payload)
	if err != nil {
		return RerankOutput{}, log.EProvider(err, "gemini", "reranking")
	}
	if resp.StatusCode != 200 {
		return RerankOutput{}, log.EProvider(nil, "gemini", "reranking", log.M{"status": resp.StatusCode, "_payload": resoutput, "url": url})
	}

	json.Unmarshal(resoutput, response)
	pricestr := resp.Header.Get("X-Cost-USD") // fpv
	pricef, _ := strconv.ParseFloat(pricestr, 64)

	rerankoutput := RerankOutput{
		Records:     response.Records,
		DurationMs:  time.Now().UnixMilli() - start.UnixMilli(),
		Created:     start.UnixMilli(),
		KfpvCostUSD: int64(pricef * 1000),
	}

	if totalprice, _ := ctx.Value("total_cost").(*TotalCost); totalprice != nil {
		totalprice.USD += int64(pricef * 1000)
	}
	cache, _ = json.Marshal(rerankoutput)
	os.WriteFile(cachepath, cache, 0644)
	return rerankoutput, nil
}
