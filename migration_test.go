package airouter

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/subiz/header"
)

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
