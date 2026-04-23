package airouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/subiz/header"
	"github.com/subiz/log"
)

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

type OpenAIEmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

// GeminiEmbeddingResponse represents the response from the embedding model.
type GeminiEmbeddingResponse struct {
	Embedding *Embedding `json:"embedding"`
}

// GeminiEmbeddingContent represents the content for an embedding request.
type GeminiEmbeddingContent struct {
	Parts []GeminiEmbeddingPart `json:"parts"`
}

// GeminiEmbeddingPart represents a part of the content for an embedding request.
type GeminiEmbeddingPart struct {
	Text string `json:"text"`
}

// EmbeddingRequest represents the request to the embedding model.
type GeminiEmbeddingRequest struct {
	Model   string                 `json:"model"`
	Content GeminiEmbeddingContent `json:"content"`
}

type GeminiRerankRecord struct {
	Id      string  `json:"id,omitempty"`
	Title   string  `json:"title,omitempty"`
	Content string  `json:"content,omitempty"`
	Score   float32 `json:"score,omitempty"`
}

type GeminiRankingResponse struct {
	Records []*GeminiRerankRecord `json:"records,omitempty"`
	Error   *GeminiError          `json:"error"`
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

var _openaiapikey string
var _geminiapikey string

// InitAPI setups API Keys for the server. Only server should call this
func InitAPI(geminiKey, openaiKey string) {
	_openaiapikey = openaiKey
	_geminiapikey = geminiKey
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

	if parsedURL.Host == "test" && parsedURL.Path == "/rerankings" {
		token := os.Getenv("GEMINI_RERANK_TOKEN")                 // export GEMINI_RERANK_TOKEN=`gcloud auth print-access-token`
		out, err := RerankAPI(context.Background(), token, input) // token will be from env
		if err != nil {
			return 500, []byte(err.Error())
		}
		b, _ := json.Marshal(out)
		return 200, b
	}

	return 404, []byte("not found " + rawURL)
}

func ChatCompleteAPI(ctx context.Context, payload []byte) (OpenAIChatResponse, error) {
	var output []byte
	var err error

	request := CompletionInput{}
	json.Unmarshal(payload, &request)
	model := request.Model
	if model == "" {
		model = Gpt_4o_mini // default
	}

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
	if response.Usage != nil {
		tier := response.ServiceTier
		if tier == "" {
			tier = request.ServiceTier
		}
		response.Usage.KFpvCostUSD = CalculateCost(model, response.Usage, tier)
	}
	return response, err
}

var chatgpttimeouterr = []byte(`{"error": {"message": "Error: Timeout was reached","type": "timeout"}}`)

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

	model := ToOpenAIModel(req.Model)
	if model != req.Model {
		m["model"] = model
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

	if req.Verbosity != "" {
		if strings.HasPrefix(model, "gpt-4") || strings.HasPrefix(model, "gpt-3") {
			delete(m, "verbosity")
			changed = true
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

		if rolei := msg["role"]; rolei != nil {
			role, _ := rolei.(string)
			newrole := role
			newrole = strings.TrimSpace(strings.ToLower(role))
			if newrole == "agent" || newrole == "bot" {
				newrole = "assistant"
			}
			msg["role"] = newrole
			if newrole != role {
				changed = true
			}
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

	if req.MaxCompletionTokens > 0 {
		m["max_completion_tokens"] = req.MaxCompletionTokens
		changed = true
	}

	if req.Reasoning != nil {
		delete(m, "reasoning")
		changed = true
		m["reasoning_effort"] = req.Reasoning.Effort
	}

	if m["reasoning_effort"] != "" {
		if model == "gpt-4o" || model == "gpt-4o-mini" || model == "gpt-4.1-mini" || model == "gpt-4.1-nano" {
			delete(m, "reasoning_effort")
			changed = true
		}

		if model == "gpt-5.4-mini" || model == "gpt-5.4-nano" {
			// Function tools with reasoning_effort are not supported for gpt-5.4-nano in /v1/chat/completions. Please use /v1/responses instead.
			if len(req.Tools) > 0 {
				delete(m, "reasoning_effort")
				changed = true
			}
		}
	}

	if len(req.Stop) > 4 {
		// Not supported with latest reasoning models o3 and o4-mini.
		// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
		changed = true
		m["stop"] = req.Stop[:4]
		if model == "o4-mini" || model == "o3" {
			delete(m, "stop")
		}
	}

	if _, has := m["stop_after_tool_called"]; has {
		delete(m, "stop_after_tool_called")
		changed = true
	}

	if _, has := m["service_tier"]; has {
		// service_tier parameter, this only applied to model gpt-5* or o3 or o4 mini, unsupported: the rest (gpt-4.1-mini)
		if !(strings.HasPrefix(model, "gpt-5") || model == "o3" || model == "o4-mini") {
			delete(m, "service_tier")
			changed = true
		}
	}

	if changed {
		delete(m, "-")
		delete(m, "instruct")
		b, _ = json.Marshal(m)
	}
	return b, nil
}

// ToGeminiRequestJSON converts an OpenAIChatRequest to a Gemini-compatible JSON request string.
func ToGeminiRequestJSON(req CompletionInput) ([]byte, error) {
	var geminiTools []*GeminiTool
	for _, tool := range req.Tools {
		decl := &GeminiFunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
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

	if req.ReasoningEffort != "" && req.ReasoningEffort != "none" {
		budget := 0
		switch req.ReasoningEffort {
		case "minimal", "low":
			budget = 256
		case "medium":
			budget = 1024
		case "high", "xhigh":
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

	if req.MaxCompletionTokens > 0 {
		if geminiReq.GenerationConfig == nil {
			geminiReq.GenerationConfig = &GeminiGenerationConfig{}
		}
		geminiReq.GenerationConfig.MaxOutputTokens = req.MaxCompletionTokens
	}

	var contents []*GeminiContent
	toolCallsByID := make(map[string]*header.LLMToolCall)

	messages := req.Messages
	if req.Instruct != "" {
		messages = append([]*header.LLMChatHistoryEntry{{Role: "system", Content: req.Instruct}}, messages...)
	}

	systemmsgs := []string{}
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
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

		case "assistant", "agent":
			if len(msg.ToolCalls) > 0 {
				var parts []*GeminiPart
				for _, tc := range msg.ToolCalls {
					toolCallsByID[tc.Id] = tc
					var args map[string]any
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

	if len(req.Stop) > 0 {
		geminiReq.GenerationConfig.StopSequences = req.Stop
		if len(geminiReq.GenerationConfig.StopSequences) > 5 {
			geminiReq.GenerationConfig.StopSequences = geminiReq.GenerationConfig.StopSequences[:5]
		}
	}
	b, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Gemini request: %w", err)
	}
	return b, nil
}

func chatCompleteChatGPT(ctx context.Context, apikey string, request CompletionInput) ([]byte, error) {
	model := request.Model
	if strings.HasPrefix(model, "gpt-5-") {
		// those models only support temperatture parameters = 1
		// https://community.openai.com/t/temperature-in-gpt-5-models/1337133/4
		request.Temperature = 1
	}

	if request.ServiceTier == "flex" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 15*time.Minute)
		defer cancel()
	}

	var err error
	requestb, err := ToOpenAICompletionJSON(request)
	if err != nil {
		return nil, err
	}
	url := "https://api.openai.com/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestb))
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

// GetEmbedding takes a text and returns its embedding.
// It uses the 'embedding-001' model.
func getGeminiEmbedding(ctx context.Context, apiKey, model, text string) (OpenAIEmbeddingResponse, error) {
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=%s", apiKey)
	var out OpenAIEmbeddingResponse
	reqBody := GeminiEmbeddingRequest{
		Model: model, // "models/embedding-001",
		Content: GeminiEmbeddingContent{
			Parts: []GeminiEmbeddingPart{{Text: text}},
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

func RerankAPI(ctx context.Context, token string, payload []byte) (RerankingResponse, error) {
	request := RerankInput{}
	json.Unmarshal(payload, &request)
	model := request.Model
	if model == "" {
		model = "semantic-ranker-default@latest"
	}
	out, err := rerankingGemini(ctx, token, request)
	if out == nil && err != nil {
		return RerankingResponse{
			Error: &OpenAIError{Message: err.Error(), Type: "internal_error"},
		}, err
	}
	return *out, nil
}

func rerankingGemini(ctx context.Context, token string, request RerankInput) (*RerankingResponse, error) {
	records := request.Records
	query := header.Norm(request.Query, 1000)

	if len(query) == 0 || len(records) == 0 {
		return &RerankingResponse{
			Created: time.Now().UnixMilli(),
			Object:  "reranking",
			Model:   request.Model,
		}, nil
	}

	topn := request.TopN
	if topn <= 0 {
		topn = len(request.Records)
	}

	model := request.Model
	var err error
	requestb, err := json.Marshal(map[string]any{
		"model":                         model,
		"query":                         query,
		"records":                       records,
		"ignoreRecordDetailsInResponse": true,
		"topN":                          topn,
	})
	if err != nil {
		return nil, err
	}

	fmt.Println("GEMINIREQ", string(requestb), token)
	url := "https://discoveryengine.googleapis.com/v1/projects/subiz-version-4/locations/global/rankingConfigs/default_ranking_config:rank"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestb))
	if err != nil {
		return nil, log.EServer(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	output := buf.Bytes()
	gemres := &GeminiRankingResponse{}
	json.Unmarshal(output, gemres)

	return toOpenAIRerankResponse(gemres, model)
}

// toOpenAIChatResponse converts a Gemini response to an OpenAI-compatible chat response.
func toOpenAIRerankResponse(res *GeminiRankingResponse, model string) (*RerankingResponse, error) {
	if res.Error != nil {
		param := ""
		if len(res.Error.Details) > 0 && len(res.Error.Details[0].FieldViolations) > 0 {
			param = res.Error.Details[0].FieldViolations[0].Field
		}
		return &RerankingResponse{
			Created: time.Now().UnixMilli(),
			Model:   model,
			Object:  "reranking",
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

	records := []*RerankRecord{}
	for _, record := range res.Records {
		records = append(records, &RerankRecord{
			Id:      record.Id,
			Title:   record.Title,
			Content: record.Content,
			Score:   record.Score,
		})
	}
	return &RerankingResponse{
		Model:   model,
		Created: time.Now().UnixMilli(),
		Object:  "reranking",
		Records: records,
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
		Created: time.Now().UnixMilli(),
		Object:  "chat.completion",
		Model:   res.ModelVersion,
		Choices: []OpenAIChoice{choice},
		Usage:   usage,
	}, nil
}

func strsToParts(strs []string) []*GeminiPart {
	out := []*GeminiPart{}
	for _, str := range strs {
		out = append(out, &GeminiPart{Text: &str})
	}
	return out
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
