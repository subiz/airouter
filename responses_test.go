package airouter

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/subiz/header"
)

func TestUseOpenAIResponsesAPI(t *testing.T) {
	tools := []OpenAITool{{Type: "function", Function: &Function{Name: "get_weather"}}}
	cases := []struct {
		input CompletionInput
		want  bool
	}{
		{CompletionInput{Model: "gpt-6-luna", Tools: tools, ReasoningEffort: "low"}, true},
		{CompletionInput{Model: "gpt-5.4-mini", Tools: tools, Reasoning: &CompletionReasoning{Effort: "medium"}}, true},
		{CompletionInput{Model: "gpt-6-luna", Tools: tools, ReasoningEffort: "none"}, true},
		{CompletionInput{Model: "gpt-6-luna"}, true},
		{CompletionInput{Model: "gpt-5.4-mini", Tools: tools, ReasoningEffort: "none"}, false},
		{CompletionInput{Model: "gpt-5.4-nano", Tools: tools}, false},
		{CompletionInput{Model: "gpt-5.4-mini", ReasoningEffort: "low"}, false},
		{CompletionInput{Model: "gpt-5", Tools: tools, ReasoningEffort: "low"}, true},
	}
	for i, c := range cases {
		if got := useOpenAIResponsesAPI(c.input); got != c.want {
			t.Errorf("case %d: expect %v, got %v", i, c.want, got)
		}
	}
}

func TestToOpenAIResponsesJSON(t *testing.T) {
	output := `[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc1"},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"let me check"}]},` +
		`{"type":"reasoning","id":"rs_2","summary":[],"encrypted_content":"enc2"},` +
		`{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"hanoi\"}"}]`
	input := CompletionInput{
		Model:               "gpt-6-luna",
		Instruct:            "be nice",
		ReasoningEffort:     "low",
		MaxCompletionTokens: 100,
		Temperature:         0.5,
		Stop:                []string{"x"},
		ServiceTier:         "flex",
		Tools: []OpenAITool{{Type: "function", Function: &Function{
			Name:        "get_weather",
			Description: "get weather",
			Parameters:  &header.JSONSchema{Type: "object"},
		}}},
		Messages: []*header.LLMChatHistoryEntry{
			{Role: "user", Contents: []*header.OpenAIMessageContent{
				{Type: "text", Text: "weather in this city?"},
				{Type: "image_url", ImageUrl: &header.OpenAIMessageContentImageUrl{Url: "https://img/hanoi.png", Detail: "low"}},
			}},
			{
				Role:      "assistant",
				Content:   "let me check",
				ToolCalls: []*header.LLMToolCall{{Id: "call_1", Type: "function", Function: &header.LLMToolFunction{Name: "get_weather", Arguments: `{"city":"hanoi"}`}}},
				Fields:    map[string]string{outputItemsField: output},
			},
			{Role: "tool", ToolCallId: "call_1", Content: "30C"},
			// without output items (e.g. no reasoning), the message is rebuilt
			{
				Role:      "assistant",
				Content:   "and saigon",
				ToolCalls: []*header.LLMToolCall{{Id: "call_2", Type: "function", Function: &header.LLMToolFunction{Name: "get_weather", Arguments: `{"city":"saigon"}`}}},
			},
			{Role: "tool", ToolCallId: "call_2", Content: "34C"},
		},
	}

	b, err := ToOpenAIResponsesJSON(input)
	if err != nil {
		t.Fatal(err)
	}
	actual := map[string]any{}
	json.Unmarshal(b, &actual)

	expected := map[string]any{}
	json.Unmarshal([]byte(`{
		"model": "gpt-6-luna",
		"store": false,
		"include": ["reasoning.encrypted_content"],
		"service_tier": "flex",
		"prompt_cache_key": "5871e58410cf47e6c2bdc08398e57831",
		"max_output_tokens": 100,
		"reasoning": {"effort": "low"},
		"tools": [{"type": "function", "strict": false, "name": "get_weather", "description": "get weather", "parameters": {"type": "object", "additionalProperties": false}}],
		"input": [
			{"role": "system", "content": "be nice"},
			{"role": "user", "content": [
				{"type": "input_text", "text": "weather in this city?"},
				{"type": "input_image", "image_url": "https://img/hanoi.png", "detail": "low"}
			]},
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc1"},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "let me check"}]},
			{"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc2"},
			{"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"hanoi\"}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "30C"},
			{"role": "assistant", "content": "and saigon"},
			{"type": "function_call", "call_id": "call_2", "name": "get_weather", "arguments": "{\"city\":\"saigon\"}"},
			{"type": "function_call_output", "call_id": "call_2", "output": "34C"}
		]
	}`), &expected)

	if !cmp.Equal(expected, actual) {
		t.Errorf("Responses request mismatch (-want +got):\n%s", cmp.Diff(expected, actual))
	}

	// chat completions must not receive our output items field
	chatb, _ := ToOpenAICompletionJSON(input)
	if strings.Contains(string(chatb), outputItemsField) {
		t.Errorf("chat completion request should not contain %s: %s", outputItemsField, chatb)
	}
}

func TestLunaPromptCacheKey(t *testing.T) {
	// without a client key, the key is derived from the instruct, no breakpoint is placed
	messages := []*header.LLMChatHistoryEntry{{Role: "user", Content: "document"}, {Role: "user", Content: "question"}}
	b, _ := ToOpenAIResponsesJSON(CompletionInput{Model: Gpt_6_luna, Instruct: "be nice", Messages: messages})
	actual := map[string]any{}
	json.Unmarshal(b, &actual)
	if actual["prompt_cache_key"] != GetMD5Hash("gpt-6-luna.be nice") || strings.Contains(string(b), "breakpoint") {
		t.Errorf("unexpected request %s", b)
	}

	b, _ = ToOpenAIResponsesJSON(CompletionInput{Model: Gpt_6_luna, PromptCacheKey: "k", Messages: messages})
	actual = map[string]any{}
	json.Unmarshal(b, &actual)
	if actual["prompt_cache_key"] != "k" {
		t.Errorf("unexpected request %s", b)
	}

	// older models keep their automatic caching
	b, _ = ToOpenAIResponsesJSON(CompletionInput{
		Model:           "gpt-5.4-mini",
		ReasoningEffort: "low",
		Tools:           []OpenAITool{{Type: "function", Function: &Function{Name: "f"}}},
		Messages:        messages,
	})
	if strings.Contains(string(b), "prompt_cache_key") {
		t.Errorf("unexpected gpt-5.4-mini request %s", b)
	}
}

func TestLunaClientCacheBreakpoint(t *testing.T) {
	// breakpoints are copied on input text blocks only
	messages := []*header.LLMChatHistoryEntry{
		{Role: "user", Content: "instructions", CacheBreakpoint: true},
		{Role: "user", Contents: []*header.OpenAIMessageContent{
			{Type: "text", Text: "doc1", PromptCacheBreakpoint: &header.OpenAIMessageContentPromptCacheBreakpoint{Ttl: "12h"}},
			{Type: "text", Text: "doc2"},
		}},
		{Role: "assistant", Content: "ok", CacheBreakpoint: true},
		{Role: "user", Content: "question"},
	}
	b, _ := ToOpenAIResponsesJSON(CompletionInput{Model: Gpt_6_luna, PromptCacheKey: "k", Messages: messages})
	actual := map[string]any{}
	json.Unmarshal(b, &actual)
	expected := []any{}
	json.Unmarshal([]byte(`[
		{"role": "user", "content": [{"type": "input_text", "text": "instructions", "prompt_cache_breakpoint": {"mode": "explicit"}}]},
		{"role": "user", "content": [
			{"type": "input_text", "text": "doc1", "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "12h"}},
			{"type": "input_text", "text": "doc2"}
		]},
		{"role": "assistant", "content": "ok"},
		{"role": "user", "content": "question"}
	]`), &expected)
	if !cmp.Equal(expected, actual["input"]) {
		t.Errorf("Responses input mismatch (-want +got):\n%s", cmp.Diff(expected, actual["input"]))
	}

	// cache_breakpoint marks the last text block of a multi part message
	b, _ = ToOpenAIResponsesJSON(CompletionInput{Model: Gpt_6_luna, Messages: []*header.LLMChatHistoryEntry{
		{Role: "user", CacheBreakpoint: true, Contents: []*header.OpenAIMessageContent{
			{Type: "text", Text: "doc1"},
			{Type: "text", Text: "doc2"},
			{Type: "image_url", ImageUrl: &header.OpenAIMessageContentImageUrl{Url: "https://img/a.png"}},
		}},
		{Role: "user", Content: "question"},
	}})
	actual = map[string]any{}
	json.Unmarshal(b, &actual)
	expected = []any{}
	json.Unmarshal([]byte(`[
		{"role": "user", "content": [
			{"type": "input_text", "text": "doc1"},
			{"type": "input_text", "text": "doc2", "prompt_cache_breakpoint": {"mode": "explicit"}},
			{"type": "input_image", "image_url": "https://img/a.png"}
		]},
		{"role": "user", "content": "question"}
	]`), &expected)
	if !cmp.Equal(expected, actual["input"]) {
		t.Errorf("Responses input mismatch (-want +got):\n%s", cmp.Diff(expected, actual["input"]))
	}

	// older models and chat completions never receive breakpoints
	input := CompletionInput{Model: "gpt-5.4-mini", ReasoningEffort: "low", Tools: []OpenAITool{{Type: "function", Function: &Function{Name: "f"}}}, Messages: messages}
	b, _ = ToOpenAIResponsesJSON(input)
	if strings.Contains(string(b), "breakpoint") {
		t.Errorf("unexpected gpt-5.4-mini request %s", b)
	}
	input.Model = "gpt-4.1-mini"
	b, _ = ToOpenAICompletionJSON(input)
	if strings.Contains(string(b), "breakpoint") {
		t.Errorf("unexpected chat completion request %s", b)
	}
}

func TestToOpenAIChatResponseFromResponses(t *testing.T) {
	output := `{
		"id": "resp_1", "created_at": 1700000000, "model": "gpt-6-luna", "status": "completed", "service_tier": "default",
		"output": [
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "{}", "status": "completed"}
		],
		"usage": {"input_tokens": 300000, "input_tokens_details": {"cached_tokens": 100000}, "output_tokens": 1000, "total_tokens": 301000},
		"error": null
	}`
	res := &OpenAIResponsesResponse{}
	if err := json.Unmarshal([]byte(output), res); err != nil {
		t.Fatal(err)
	}
	chat := toOpenAIChatResponseFromResponses(res)
	if len(chat.Choices) != 1 || chat.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("unexpected choices %+v", chat.Choices)
	}
	msg := chat.Choices[0].Message
	if msg.Content != nil || len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_1" || msg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("unexpected message %+v", msg)
	}
	if len(msg.OutputItems) != 2 || !strings.Contains(string(msg.OutputItems[0]), `"enc"`) || !strings.Contains(string(msg.OutputItems[1]), `"fc_1"`) {
		t.Errorf("unexpected output items %s", msg.OutputItems)
	}
	if cost := CalculateCost(res.Model, chat.Usage, chat.ServiceTier); cost != 40_000_000+2_000_000+750_000 {
		t.Errorf("unexpected cost %d", cost)
	}

	text := `{"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
		"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hel"}]}]}`
	res = &OpenAIResponsesResponse{}
	json.Unmarshal([]byte(text), res)
	chat = toOpenAIChatResponseFromResponses(res)
	if chat.Choices[0].Message.GetContent() != "hel" || chat.Choices[0].FinishReason != "length" {
		t.Errorf("unexpected choice %+v", chat.Choices[0])
	}
}

func TestCompleteToolsWithReasoning(t *testing.T) {
	if os.Getenv("CHATGPT_API_KEY") == "" {
		t.Skip("CHATGPT_API_KEY is not set")
	}
	InitAPI(os.Getenv("GEMINI_API_KEY"), os.Getenv("CHATGPT_API_KEY"))
	BACKEND = "https://test"

	for _, model := range []string{Gpt_6_luna, Gpt_5_4_mini} {
		t.Run(model, func(t *testing.T) {
			called := 0
			input := CompletionInput{
				Model:           model,
				ReasoningEffort: "low",
				NoLog:           true,
				Seed:            int(time.Now().UnixNano() % 1000000), // bypass the local cache
				Messages:        []*header.LLMChatHistoryEntry{{Role: "user", Content: "What is the weather in Hanoi? Answer with the temperature only."}},
				Tools: []OpenAITool{{Type: "function", Function: &Function{
					Name:        "get_weather",
					Description: "get the current weather of a city",
					Parameters: &header.JSONSchema{
						Type:       "object",
						Required:   []string{"city"},
						Properties: map[string]*header.JSONSchema{"city": {Type: "string"}},
					},
					Handler: func(ctx context.Context, arg, callid string, ctxm map[string]any) string {
						called++
						return "31C, sunny"
					},
				}}},
			}

			content, output, err := Complete(context.Background(), "", input)
			if err != nil {
				t.Fatal(err)
			}
			if called == 0 {
				t.Fatalf("tool was not called, content %q", content)
			}
			if !strings.Contains(content, "31") {
				t.Errorf("unexpected content %q", content)
			}
			// the model may skip reasoning on easy prompts, so only log whether it was carried over
			hasreasoning := strings.Contains(string(output.Request), outputItemsField)

			// cost reported by the (fake) server must match the token usage
			expected := CalculateCost(model, &Usage{PromptTokens: output.InputTokens, CompletionTokens: output.OutputTokens}, "")
			if output.KfpvCostUSD <= 0 || math.Abs(float64(output.KfpvCostUSD-expected)) > 10 {
				t.Errorf("expect cost about %d, got %d", expected, output.KfpvCostUSD)
			}
			t.Logf("reasoning carried over %v", hasreasoning)
			t.Logf("content %q, input %d, output %d tokens, cost %d kfpvusd", content, output.InputTokens, output.OutputTokens, output.KfpvCostUSD)
		})
	}
}

func TestToOpenAIResponsesJSONParams(t *testing.T) {
	tools := []OpenAITool{{Type: "function", Function: &Function{Name: "get_weather"}}}

	// gpt-5.4 keeps prompt_cache_retention, drops sampling params when reasoning
	b, _ := ToOpenAIResponsesJSON(CompletionInput{Model: "gpt-5.4-mini", Tools: tools, ReasoningEffort: "low", Temperature: 0.5})
	m := map[string]any{}
	json.Unmarshal(b, &m)
	if m["prompt_cache_retention"] != "24h" || m["prompt_cache_options"] != nil || m["temperature"] != nil {
		t.Errorf("unexpected gpt-5.4-mini request %s", b)
	}

	// gpt-6-luna without reasoning keeps sampling params
	b, _ = ToOpenAIResponsesJSON(CompletionInput{Model: "gpt-6-luna", ReasoningEffort: "none", Temperature: 0.5, TopP: 0.9})
	m = map[string]any{}
	json.Unmarshal(b, &m)
	if m["temperature"] != 0.5 || m["top_p"] != 0.9 || m["prompt_cache_retention"] != nil {
		t.Errorf("unexpected gpt-6-luna request %s", b)
	}
}

func TestLunaPromptCache30m(t *testing.T) {
	if os.Getenv("CHATGPT_API_KEY") == "" {
		t.Skip("CHATGPT_API_KEY is not set")
	}
	InitAPI(os.Getenv("GEMINI_API_KEY"), os.Getenv("CHATGPT_API_KEY"))

	// unique long prefix, so the first call writes the cache and the second one reads it
	instruct := fmt.Sprint(time.Now().UnixNano()) + " " + strings.Repeat("The quick brown fox jumps over the lazy dog. ", 300)
	for i := range 2 {
		payload, _ := json.Marshal(CompletionInput{
			Model:               Gpt_6_luna,
			Instruct:            instruct,
			MaxCompletionTokens: 50,
			Messages:            []*header.LLMChatHistoryEntry{{Role: "user", Content: "say hi"}},
		})
		resp, err := ChatCompleteAPI(context.Background(), payload)
		if err != nil {
			t.Fatal(err)
		}
		usage := resp.Usage
		details := usage.PromptTokensDetails
		if details == nil {
			t.Fatalf("call %d: no prompt tokens details", i)
		}
		if i == 0 && (details.CacheWriteTokensDetails == nil || details.CacheWriteTokensDetails.Ttl30mTokens == 0 || details.CacheWriteTokensDetails.Ttl12hTokens != 0) {
			t.Errorf("call %d: expect 30m cache writes only, got %+v", i, details)
		}
		if i == 1 && details.CachedTokens == 0 {
			t.Errorf("call %d: expect cache reads, got %+v", i, details)
		}

		uncached := usage.PromptTokens - details.CachedTokens - details.CacheWriteTokens
		expected := 100*float64(uncached) + 10*float64(details.CachedTokens) + 125*float64(details.CacheWriteTokens) + 500*float64(usage.CompletionTokens)
		if math.Abs(float64(usage.KFpvCostUSD)-expected) > 1 {
			t.Errorf("call %d: expect cost %.0f, got %d", i, expected, usage.KFpvCostUSD)
		}
		t.Logf("call %d: prompt %d, cached %d, write %d, output %d, cost %d kfpvusd", i, usage.PromptTokens, details.CachedTokens, details.CacheWriteTokens, usage.CompletionTokens, usage.KFpvCostUSD)
		time.Sleep(2 * time.Second)
	}
}

func TestToOpenAIChatResponseFromResponsesFailed(t *testing.T) {
	res := &OpenAIResponsesResponse{}
	json.Unmarshal([]byte(`{"status": "failed", "error": {"code": "server_error", "message": "boom"}}`), res)
	chat := toOpenAIChatResponseFromResponses(res)
	if chat.Error == nil || chat.Error.Message != "boom" || len(chat.Choices) != 0 {
		t.Errorf("unexpected response %+v", chat)
	}

	// failed without error details still reports an error
	res = &OpenAIResponsesResponse{}
	json.Unmarshal([]byte(`{"status": "failed"}`), res)
	if chat := toOpenAIChatResponseFromResponses(res); chat.Error == nil {
		t.Errorf("expect an error, got %+v", chat)
	}

	// no reasoning, no output items to replay
	res = &OpenAIResponsesResponse{}
	json.Unmarshal([]byte(`{"status": "completed", "output": [{"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"}]}`), res)
	if chat := toOpenAIChatResponseFromResponses(res); len(chat.Choices[0].Message.OutputItems) != 0 {
		t.Errorf("expect no output items, got %s", chat.Choices[0].Message.OutputItems)
	}
}

type migrationTransport func(*http.Request) (*http.Response, error)

func (f migrationTransport) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestRetiredOpenAIModels(t *testing.T) {
	cases := []struct {
		alias, snapshot, target string
		pro                     bool
	}{
		{"gpt-5-mini", "gpt-5-mini-2025-08-07", Gpt_6_luna, false},
		{"gpt-5-nano", "gpt-5-nano-2025-08-07", Gpt_6_luna, false},
		{"gpt-5", "gpt-5-2025-08-07", Gpt_5_6_sol, false},
		{"o3", "o3-2025-04-16", Gpt_5_6_sol, false},
		{"gpt-5-pro", "gpt-5-pro-2025-10-06", Gpt_5_6_sol, true},
		{"o3-pro", "o3-pro-2025-06-10", Gpt_5_6_sol, true},
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	for _, tc := range cases {
		for _, alias := range []string{tc.alias, tc.snapshot} {
			t.Run(alias, func(t *testing.T) {
				calls := 0
				http.DefaultClient = &http.Client{Transport: migrationTransport(func(req *http.Request) (*http.Response, error) {
					calls++
					if req.URL.String() != "https://api.openai.com/v1/responses" {
						t.Fatalf("wrong endpoint: %s", req.URL)
					}
					var body map[string]any
					if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					if body["model"] != tc.target || body["max_output_tokens"] != float64(200) {
						t.Fatalf("wrong model or token limit: %v", body)
					}
					reasoning := body["reasoning"].(map[string]any)
					if reasoning["effort"] != "low" || reasoning["context"] != "current_turn" || (reasoning["mode"] == "pro") != tc.pro {
						t.Fatalf("wrong reasoning: %v", reasoning)
					}
					for _, field := range []string{"prompt_cache_retention", "temperature", "top_p", "stop", "seed"} {
						if _, has := body[field]; has {
							t.Errorf("unexpected upstream field %s", field)
						}
					}
					if body["prompt_cache_key"] != GetMD5Hash(tc.target+".Be helpful") {
						t.Errorf("missing cache key: %v", body)
					}
					// Provider tier must take precedence over the requested flex tier for billing.
					response := fmt.Sprintf(`{"id":"resp_test","model":%q,"status":"completed","service_tier":"default","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":1000,"output_tokens":20,"total_tokens":1020,"input_tokens_details":{"cached_tokens":200,"cache_write_tokens":300}}}`, tc.target)
					return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(response)), Header: http.Header{}}, nil
				})}
				payload, _ := json.Marshal(CompletionInput{
					Model: alias, Instruct: "Be helpful", ReasoningEffort: "minimal", ServiceTier: "flex",
					Temperature: 0.5, TopP: 0.8, Seed: 42, Stop: []string{"END"}, MaxCompletionTokens: 200,
					Messages: []*header.LLMChatHistoryEntry{{Role: "user", Content: "Hello"}},
				})
				out, err := ChatCompleteAPI(context.Background(), payload)
				if err != nil {
					t.Fatal(err)
				}
				if calls != 1 || out.Model != alias || out.Object != "chat.completion" || len(out.Choices) != 1 || out.Choices[0].Message.GetContent() != "hello" {
					t.Fatalf("client contract changed: %+v", out)
				}
				wantCost := int64(99_500) // Luna: 500 input + 200 cached + 300 writes + 20 output
				if tc.target == Gpt_5_6_sol {
					wantCost = 3_980_000
				}
				if out.Usage == nil || out.Usage.KFpvCostUSD != wantCost {
					t.Fatalf("cost = %+v, want %d", out.Usage, wantCost)
				}
			})
		}
	}
}

func TestMigratedOpenAIReasoningAndTools(t *testing.T) {
	for _, model := range []string{"gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5-pro", "o3", "o3-pro"} {
		t.Run(model, func(t *testing.T) {
			req := CompletionInput{
				Model: model, Reasoning: &CompletionReasoning{Effort: "none"}, ReasoningEffort: "high", Temperature: 0.5,
				Tools: []OpenAITool{{Type: "function", Function: &Function{Name: "get_weather"}}},
				Messages: []*header.LLMChatHistoryEntry{
					{Role: "user", Content: "Weather?"},
					{Role: "assistant", ToolCalls: toOurToolCalls([]ToolCall{{ID: "call_1", Type: "function", Function: ToolFunction{Name: "get_weather", Arguments: `{}`}}}),
						Fields: map[string]string{outputItemsField: `[{"type":"reasoning","encrypted_content":"encrypted"},{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{}"}]`}},
					{Role: "tool", ToolCallId: "call_1", Content: "31C"},
				},
			}
			b, err := ToOpenAIResponsesJSON(req)
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			json.Unmarshal(b, &body)
			if body["reasoning"].(map[string]any)["effort"] != "none" || body["temperature"] != 0.5 {
				t.Fatalf("explicit effort and sampling changed: %s", b)
			}
			input := body["input"].([]any)
			if len(input) != 4 || input[1].(map[string]any)["encrypted_content"] != "encrypted" || input[2].(map[string]any)["call_id"] != "call_1" || input[3].(map[string]any)["output"] != "31C" {
				t.Fatalf("tool continuation changed: %s", b)
			}
			if req.Reasoning.Effort != "none" || req.Model != model {
				t.Fatal("caller input mutated")
			}
		})
	}
}

func TestMigrationLeavesOtherModelsUnchanged(t *testing.T) {
	for _, model := range []string{Gpt_4o, Gpt_4o_mini, Gpt_5_4_mini, Gpt_5_4_nano, Gpt_6_luna, Gpt_5_6_sol, O4_mini} {
		if got := ToOpenAIModel(model); got != model {
			t.Errorf("%s unexpectedly maps to %s", model, got)
		}
	}
	if got := GetFallbackChatModel(Gemini_2_5_flash_lite); got != Gpt_6_luna {
		t.Errorf("fallback still targets retired model: %s", got)
	}
	// Keep the existing cross-provider fallbacks for callers using old aliases.
	if ToGeminiModel(Gpt_5_mini) != Gemini_3_1_flash_lite || ToGeminiModel(Gpt_5_nano) != Gemini_2_5_flash_lite {
		t.Fatal("existing Gemini fallbacks changed")
	}
}

func TestSolLongContextCost(t *testing.T) {
	// Long-context rates cover the entire request, including cached input and cache writes.
	usage := &Usage{PromptTokens: 300_000, CompletionTokens: 1000,
		PromptTokensDetails: &PromptTokensDetails{CachedTokens: 50_000, CacheWriteTokens: 10_000}}
	const want = int64(2_090_000_000) // 240K*8 + 50K*0.8 + 10K*10 + 1K*30, in kfpv USD
	if got := CalculateCost(Gpt_5_6_sol, usage, ""); got != want {
		t.Fatalf("long-context cost = %d, want %d", got, want)
	}
	if got := CalculateCost(Gpt_5_6_sol, usage, "flex"); got != want/2 {
		t.Fatalf("flex long-context cost = %d, want %d", got, want/2)
	}
}

func TestMigratedOpenAIError(t *testing.T) {
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: migrationTransport(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 429, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(
			`{"error":{"message":"Rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}`))}, nil
	})}
	out, err := ChatCompleteAPI(context.Background(), []byte(`{"model":"gpt-5-mini"}`))
	if err == nil || out.Error == nil || out.Error.Code != "rate_limit_exceeded" || out.Usage != nil {
		t.Fatalf("upstream error was lost or billed: response=%+v, err=%v", out, err)
	}
}

func TestProviderPromptUsageExcludesReplays(t *testing.T) {
	t.Chdir(t.TempDir())
	oldClient, oldBackend := http.DefaultClient, BACKEND
	t.Cleanup(func() { http.DefaultClient, BACKEND = oldClient, oldBackend })
	BACKEND = "https://gateway.example"
	requests := 0
	http.DefaultClient = &http.Client{Transport: migrationTransport(func(req *http.Request) (*http.Response, error) {
		requests++
		var input CompletionInput
		if err := json.NewDecoder(req.Body).Decode(&input); err != nil {
			return nil, err
		}
		if input.Name != "review" || req.URL.Query().Get("x-name") != input.Name || req.URL.Query().Has("x-purpose") {
			t.Fatal("gateway must receive the request name in the payload and x-name")
		}
		headers := http.Header{}
		if requests == 2 {
			headers.Set("X-Cache-Hit", "true")
		}
		body := `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":2000,"prompt_tokens_details":{"cached_tokens":1024,"cache_write_tokens_details":{"ttl_30m_tokens":300,"ttl_12h_tokens":400}}}}`
		return &http.Response{StatusCode: 200, Header: headers, Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	for i, prompt := range []string{"first", "first", "second"} {
		_, meta, err := Complete(context.Background(), "test", CompletionInput{Name: "review", Instruct: prompt, NoLog: true})
		if err != nil {
			t.Fatal(err)
		}
		if meta.InputTokens != 2000 || meta.InputCachedTokens != 1024 {
			t.Fatal("original response usage was lost")
		}
		if i == 0 {
			if meta.ProviderInputTokens != 2000 || meta.ProviderCachedTokens != 1024 || meta.ProviderCacheWriteTokens != 700 {
				t.Fatalf("incorrect provider usage %+v", meta)
			}
		} else if meta.ProviderInputTokens != 0 || meta.ProviderCachedTokens != 0 || meta.ProviderCacheWriteTokens != 0 {
			t.Fatalf("replayed provider usage on call %d", i)
		}
	}
	if requests != 2 {
		t.Fatalf("gateway requests = %d, want 2", requests)
	}
}

func TestRequestNameStaysAtGateway(t *testing.T) {
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	for _, tc := range []struct{ model, endpoint string }{
		{Gpt_4o_mini, "https://api.openai.com/v1/chat/completions"},
		{Gpt_6_luna, "https://api.openai.com/v1/responses"},
		{Gemini_2_5_flash_lite, "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			calls := 0
			http.DefaultClient = &http.Client{Transport: migrationTransport(func(req *http.Request) (*http.Response, error) {
				calls++
				if req.URL.String() != tc.endpoint {
					t.Fatalf("got endpoint %s, want %s", req.URL, tc.endpoint)
				}
				var body map[string]json.RawMessage
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if name, exists := body["name"]; exists {
					t.Errorf("internal request name reached provider: %s", name)
				}
				if !strings.Contains(string(body["tools"]), `"lookup_order"`) {
					t.Error("provider request lost the tool name")
				}
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("{}"))}, nil
			})}
			input := CompletionInput{
				Name: "ai_agent_review_and_rewrite", Model: tc.model,
				Messages: []*header.LLMChatHistoryEntry{{Role: "user", Content: "Hello"}},
				Tools:    []OpenAITool{{Type: "function", Function: &Function{Name: "lookup_order"}}},
			}
			payload, err := json.Marshal(input)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ChatCompleteAPI(context.Background(), payload); err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Fatalf("got %d provider calls, want 1", calls)
			}
		})
	}
}
