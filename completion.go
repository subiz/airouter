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

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/subiz/header"
	"github.com/subiz/log"
)

type AIFunction struct {
	header.AIFunction
	Handler func(ctx context.Context, arg, callid string, ctxm map[string]any) (string, bool, error) // "", true, nil -> abandon completion, stop the flow immediately
}

type CompletionOutput struct {
	Content           string `json:"content"`
	Refusal           string `json:"refusal"`
	Request           []byte `json:"request"`
	Response          []byte `json:"response"`
	InputTokens       int64  `json:"input_tokens"`
	OutputTokens      int64  `json:"output_tokens"`
	InputCachedTokens int64  `json:"input_cached_tokens"`
	OuputCachedTokens int64  `json:"output_cached_tokens"`
	Created           int64  `json:"created"`
	DurationMs        int64  `json:"duration_ms"`
	FpvCostUSD        int64  `json:"fpv_cost_usd"`
}

var _apikey string

func Init(apikey string) {
	_apikey = apikey
}

func ChatComplete(ctx context.Context, model string, instruction string, histories []*header.LLMChatHistoryEntry, functions []*AIFunction, responseformat *header.LLMResponseJSONSchemaFormat, toolchoice string, stopAfterFunctionCall bool) (string, *CompletionOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()
	var tools []openai.ChatCompletionToolParam
	var fnM = map[string]*AIFunction{}
	for _, fn := range functions {
		fnM[fn.Name] = fn
		f := openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        fn.Name,
				Description: openai.String(fn.Description),
				Parameters: openai.FunctionParameters(map[string]any{
					"type":                 fn.Parameters.GetType(),
					"properties":           fn.Parameters.GetProperties(),
					"additionalProperties": false,
					"required":             fn.Parameters.GetRequired(),
				}),
			},
		}
		if fn.Parameters == nil {
			f.Function.Parameters = openai.FunctionParameters(map[string]any{
				"type":       "object",
				"properties": &header.JSONSchema{},
				"required":   []string{},
			})
		}
		tools = append(tools, f)
	}

	instruction = CleanString(instruction)
	params := openai.ChatCompletionNewParams{
		Seed:  openai.Int(0),
		Model: model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(instruction),
		},
		Temperature: openai.Float(0.0),
		TopP:        openai.Float(1.0),
	}

	if toolchoice != "" {
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String(toolchoice)}
	}

	for _, entry := range histories {
		if entry.Role == "user" {
			// param.Messages = append(param.Messages, completion.Choices[0].Message.ToParam())
			// param.Messages = append(param.Messages, openai.UserMessage("How big are those?"))
			params.Messages = append(params.Messages, openai.UserMessage(entry.Content))
		} else if entry.Role == "system" {
			params.Messages = append(params.Messages, openai.SystemMessage(entry.Content))
		} else {
			params.Messages = append(params.Messages, openai.AssistantMessage(entry.Content))
		}
	}

	if responseformat != nil {
		b, _ := json.Marshal(responseformat.Schema)
		schema := map[string]any{}
		json.Unmarshal(b, &schema)
		if !responseformat.Schema.AdditionalProperties {
			schema["additionalProperties"] = false // omitted when json marshal but required by OPENAI, so we must manually set it
		}
		addAdditionalProp(schema)

		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
					Name: responseformat.Name,
					// Description: openai.F("Notable information about a person"),
					Schema: any(schema),
					Strict: openai.Bool(responseformat.Strict),
				},
			},
		}

	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	completion := &openai.ChatCompletion{}
	var err error
	functioncalled := false
	requestbody, _ := json.Marshal(params)
	md5sum := GetMD5Hash(string(requestbody))
	cachepath := "./.cache/" + md5sum
	cache, err := os.ReadFile(cachepath)
	if err != nil {
		if _, err := os.Stat("./.cache"); os.IsNotExist(err) {
			os.MkdirAll("./.cache", os.ModePerm)
		}
		_, err := os.Stat(cachepath)
		if err == nil || !os.IsNotExist(err) {
			panic(err)
		}
	}
	if len(cache) > 0 {
		completion := &CompletionOutput{}
		json.Unmarshal(cache, completion)
		return completion.Content, completion, nil
	}

	var totalCost int64
	tokenUsages := []openai.CompletionUsage{}

	for i := 0; i < 5; i++ { // max 5 loops
		requestbody, _ = json.Marshal(params)
		if stopAfterFunctionCall && functioncalled {
			break
		}

		// Retrieve the value and assert it as a string
		accid, _ := ctx.Value("account_id").(string)
		convoid, _ := ctx.Value("conversation_id").(string)

		te := time.Now()

		// send to subiz server
		q := neturl.Values{}
		if accid, _ := ctx.Value("account_id").(string); accid != "" {
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

		url := "https://api.subiz.com.vn/ai/completions?" + q.Encode()
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestbody))
		if err != nil {
			return "", nil, log.EServer(err)
		}

		req.Header.Set("Authorization", "Bearer "+_apikey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		log.Info(accid, log.Stack(), "SUBMITCHATGTPLLM INSTRUCTION", convoid, string(requestbody), time.Since(te))

		if err != nil {
			return "", nil, log.EProvider(err, "openai", "completion")
		}

		defer resp.Body.Close()
		buf := new(bytes.Buffer)
		buf.ReadFrom(resp.Body)
		output := buf.Bytes()

		if resp.StatusCode != 200 {
			return "", nil, log.EProvider(err, "openai", "completion", log.M{"status": resp.StatusCode, "_payload": output})
		}

		pricestr := resp.Header.Get("X-Cost-USD")
		price, _ := strconv.Atoi(pricestr)
		totalCost += int64(price)

		json.Unmarshal(output, completion)
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
		params.Messages = append(params.Messages, completion.Choices[0].Message.ToParam())
		for _, toolCall := range toolCalls {
			var output string
			var callid string
			var stop bool
			if fn := fnM[toolCall.Function.Name]; fn != nil {
				functioncalled = true
				callid = toolCall.ID
				output, stop, err = fn.Handler(ctx, toolCall.Function.Arguments, callid, nil)
			}

			// m := openai.ToolMessage(output, toolCall.ID)
			if stop {
				goto exit
			}
			mx := openai.ToolMessage(output, toolCall.ID)
			params.Messages = append(params.Messages, mx)
		}
	}

exit:
	completionoutput := &CompletionOutput{
		Request:    requestbody,
		DurationMs: time.Now().UnixMilli() - start.UnixMilli(),
		Created:    start.UnixMilli(),
		FpvCostUSD: totalCost,
	}

	for _, tokenUsage := range tokenUsages {
		completionoutput.InputTokens += tokenUsage.PromptTokens
		completionoutput.OutputTokens += tokenUsage.CompletionTokens
	}

	if len(completion.Choices) > 0 {
		completionoutput.Refusal = completion.Choices[0].Message.Refusal
		completionoutput.Content = completion.Choices[0].Message.Content
	}
	b, _ := json.Marshal(completionoutput)
	err = os.WriteFile(cachepath, b, 0644)
	return completionoutput.Content, completionoutput, err
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
