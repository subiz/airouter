package airouter

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"net/http"
	neturl "net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/subiz/header"
	"github.com/subiz/log"
)

type AIFunction struct {
	header.AIFunction
	Handler func(ctx context.Context, arg, callid string, ctxm map[string]any) (string, bool) // "", true, nil -> abandon completion, stop the flow immediately
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

func Init(apikey string) {
	_apikey = apikey
}

type TotalCost struct {
	USD int64 `json:"usd"` // 1000000000
}

func ChatComplete(ctx context.Context, model string, instruction string, histories []*header.LLMChatHistoryEntry, functions []*AIFunction, responseformat *header.LLMResponseJSONSchemaFormat, toolchoice string, stopAfterFunctionCall bool) (string, CompletionOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()
	var tools []OpenAITool
	var fnM = map[string]*AIFunction{}
	for _, fn := range functions {
		fnM[fn.Name] = fn
		f := OpenAITool{
			Type: "function",
			Function: Function{
				Name:        fn.Name,
				Description: fn.Description,
			},
		}
		if fn.Parameters != nil {
			properties := map[string]*JSONSchema{}
			for k, v := range fn.Parameters.Properties {
				properties[k] = toOpenAISchema(v)
			}
			f.Function.Parameters = &JSONSchema{
				Type:                 fn.Parameters.GetType(),
				AdditionalProperties: false,
				Required:             fn.Parameters.GetRequired(),
				Properties:           properties,
			}
		}
		tools = append(tools, f)
	}

	instruction = CleanString(instruction)
	params := OpenAIChatRequest{
		Seed:        0,
		Model:       model,
		Messages:    []OpenAIChatMessage{{Role: "system", Content: &instruction}},
		Temperature: 0.0,
		TopP:        1.0,
	}

	if toolchoice != "" {
		params.ToolChoice = toolchoice
	}

	for _, entry := range histories {
		content := entry.Content
		if entry.Role == "user" {
			// param.Messages = append(param.Messages, completion.Choices[0].Message.ToParam())
			// param.Messages = append(param.Messages, openai.UserMessage("How big are those?"))
			params.Messages = append(params.Messages, OpenAIChatMessage{Role: "user", Content: &content})
		} else if entry.Role == "system" {
			params.Messages = append(params.Messages, OpenAIChatMessage{Role: "system", Content: &content})
		} else {
			params.Messages = append(params.Messages, OpenAIChatMessage{Role: "assistant", Content: &content})
		}
	}

	if responseformat != nil {
		params.ResponseFormat = &ResponseFormat{
			Type: "json_schema",
			JSONSchema: &RJSONSchema{
				Name:   responseformat.GetName(),
				Strict: responseformat.GetStrict(),
				Schema: toOpenAISchema(responseformat.GetSchema()),
			},
		}
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	functioncalled := false
	var requestbody []byte
	completion := &OpenAIChatResponse{}

	var totalCost float64
	tokenUsages := []*Usage{}

	for range 5 { // max 5 loops
		if stopAfterFunctionCall && functioncalled {
			break
		}

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

		url := "https://api.subiz.com.vn/4.1/ai/completions?" + q.Encode()
		requestbody, _ = json.Marshal(params)

		md5sum := GetMD5Hash(string(requestbody))
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

		cachehit := "HIT"
		if len(cache) == 0 {
			cachehit = "MISS"
		}
		log.Info(accid, log.Stack(), "SUBMITLLM", cachehit, convoid, string(requestbody), time.Since(te))
		if len(cache) > 0 {
			json.Unmarshal(cache, completion)
		} else {
			req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestbody))
			if err != nil {
				return "", CompletionOutput{}, log.EServer(err)
			}

			req.Header.Set("Authorization", "Bearer "+_apikey)
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return "", CompletionOutput{}, log.EProvider(err, "openai", "completion")
			}
			defer resp.Body.Close()
			buf := new(bytes.Buffer)
			buf.ReadFrom(resp.Body)
			output := buf.Bytes()
			if resp.StatusCode != 200 {
				return "", CompletionOutput{}, log.EProvider(err, "openai", "completion", log.M{"status": resp.StatusCode, "_payload": output})
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
		tokenUsages = append(tokenUsages, completion.Usage)
		if len(completion.Choices) == 0 {
			break
		}
		toolCalls := completion.Choices[0].Message.ToolCalls
		// Abort early if there are no tool calls
		if len(toolCalls) == 0 {
			break
		}

		// If there is a was a function call, continue the conversation
		params.Messages = append(params.Messages, completion.Choices[0].Message)
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
			params.Messages = append(params.Messages, OpenAIChatMessage{
				Content:    &output,
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

	text = strings.TrimSpace(text)
	url := "https://api.subiz.com.vn/4.1/ai/embeddings?" + q.Encode()
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
		json.Unmarshal(cache, &output)
		return output.Vector, output, nil
	}

	log.Info(accid, log.Stack(), "EMBEDDING", convoid, text, time.Since(te))

	req, err := http.NewRequest("POST", url, bytes.NewBuffer([]byte(text)))
	if err != nil {
		return nil, EmbeddingOutput{}, log.EServer(err)
	}

	req.Header.Set("Authorization", "Bearer "+_apikey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, EmbeddingOutput{}, log.EProvider(err, "openai", "embedding")
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	resoutput := buf.Bytes()
	if resp.StatusCode != 200 {
		return nil, EmbeddingOutput{}, log.EProvider(err, "openai", "embedding", log.M{"status": resp.StatusCode, "_payload": resoutput})
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
	return embeddingoutput.Vector, embeddingoutput, nil
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

func toOpenAISchema(h *header.JSONSchema) *JSONSchema {
	if h == nil {
		return &JSONSchema{}
	}

	p := &JSONSchema{
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
		p.Properties = map[string]*JSONSchema{}
	}
	for k, v := range h.Properties {
		p.Properties[k] = toOpenAISchema(v)
	}
	return p
}
