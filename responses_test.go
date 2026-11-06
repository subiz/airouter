package airouter

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
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
		{CompletionInput{Model: "gpt-5", Tools: tools, ReasoningEffort: "low"}, false},
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
