package airouter

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/tidwall/sjson"
)

type RequestTestCase struct {
	OpenAI *OpenAIChatRequest `json:"openai"`
	Gemini *GeminiRequest     `json:"gemini"`
}

func TestRequestConversion(t *testing.T) {
	file, err := os.ReadFile("./testcases/request_testcases.json")
	if err != nil {
		t.Fatalf("Failed to read test cases file: %v", err)
	}

	var testCases map[string]RequestTestCase
	if err := json.Unmarshal(file, &testCases); err != nil {
		t.Fatalf("Failed to unmarshal test cases: %v", err)
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actualJSON, err := ToGeminiRequestJSON(*tc.OpenAI)
			if err != nil {
				t.Fatalf("ToGeminiRequestJSON failed: %v", err)
			}

			var actualMap, expectedMap map[string]interface{}
			if err := json.Unmarshal([]byte(actualJSON), &actualMap); err != nil {
				t.Fatalf("Failed to unmarshal actual JSON: %v", err)
			}

			gemb, _ := json.Marshal(tc.Gemini)
			if err := json.Unmarshal(gemb, &expectedMap); err != nil {
				t.Fatalf("Failed to unmarshal expected JSON: %v", err)
			}

			// 4. Marshal and unmarshal to maps for a more robust comparison.
			actualJSONb, err := json.Marshal(actualMap)
			if err != nil {
				t.Fatalf("Failed to marshal actual response: %v", err)
			}

			if err := json.Unmarshal(actualJSONb, &actualMap); err != nil {
				t.Fatalf("Failed to unmarshal actual JSON: %v", err)
			}

			if !cmp.Equal(expectedMap, actualMap) {
				t.Errorf("Request JSON mismatch (-want +got):\n%s", cmp.Diff(expectedMap, actualMap))
			}
		})
	}
}

// ResponseTestCase is a struct for a single test case from the JSON file.
type ResponseTestCase struct {
	Gemini *GeminiAPIResponse  `json:"gemini"`
	OpenAI *OpenAIChatResponse `json:"openai"`
}

func TestResponseConversion(t *testing.T) {
	// Read the test cases from the JSON file
	file, err := os.ReadFile("./testcases/response_testcases.json")
	if err != nil {
		t.Fatalf("Failed to read test cases file: %v", err)
	}

	var testCases map[string]ResponseTestCase
	if err := json.Unmarshal(file, &testCases); err != nil {
		t.Fatalf("Failed to unmarshal test cases: %v", err)
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			// 1. Manually construct the GeminiAPIResponse from the test case data.
			// 2. Call the function under test.
			actualResponse, err := toOpenAIChatResponse(tc.Gemini)
			if err != nil {
				t.Fatalf("toOpenAIChatResponse failed: %v", err)
			}

			// 4. Prepare for comparison.
			expectedResponse := tc.OpenAI
			if expectedResponse != nil {
				if len(actualResponse.Choices) > 0 && len(expectedResponse.Choices) > 0 {
					for i, choice := range actualResponse.Choices {
						if i < len(expectedResponse.Choices) {
							for j, toolCall := range choice.Message.ToolCalls {
								if j < len(expectedResponse.Choices[i].Message.ToolCalls) {
									expectedResponse.Choices[i].Message.ToolCalls[j].ID = toolCall.ID
								}
							}
						}
					}
				}
			}

			// 5. Marshal and unmarshal to maps for a more robust comparison.
			actualJSON, err := json.Marshal(actualResponse)
			if err != nil {
				t.Fatalf("Failed to marshal actual response: %v", err)
			}
			expectedJSON, err := json.Marshal(expectedResponse)
			if err != nil {
				t.Fatalf("Failed to marshal expected response: %v", err)
			}

			var actualMap, expectedMap map[string]interface{}
			if err := json.Unmarshal(actualJSON, &actualMap); err != nil {
				t.Fatalf("Failed to unmarshal actual JSON: %v", err)
			}
			if err := json.Unmarshal(expectedJSON, &expectedMap); err != nil {
				t.Fatalf("Failed to unmarshal expected JSON: %v", err)
			}

			// 6. Compare the maps.
			if !cmp.Equal(expectedMap, actualMap) {
				t.Errorf("Response JSON mismatch (-want +got):\n%s", cmp.Diff(expectedMap, actualMap))
			}
		})
	}
}

// ResponseTestCase is a struct for a single test case from the JSON file.
type ChatTestCase struct {
	Output *OpenAIChatResponse `json:"output"`
	Input  *OpenAIChatRequest  `json:"input"`
}

func TestChatCompletion(t *testing.T) {
	// Read the test cases from the JSON file
	file, err := os.ReadFile("./testcases/chat_testcases.json")
	if err != nil {
		t.Fatalf("Failed to read test cases file: %v", err)
	}

	var testCases map[string]ChatTestCase
	if err := json.Unmarshal(file, &testCases); err != nil {
		t.Fatalf("Failed to unmarshal test cases: %v", err)
	}

	openai_apikey := os.Getenv("OPENAI_APIKEY")
	gemini_apikey := os.Getenv("GEMINI_APIKEY")
	for name, tc := range testCases {
		if name != "20" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			var output []byte
			b, _ := json.Marshal(tc.Input)

			if strings.HasPrefix(tc.Input.Model, "gpt") {
				output, err = chatCompleteChatGPT(ctx, openai_apikey, tc.Input.Model, b)
			}

			if strings.HasPrefix(tc.Input.Model, "gemini") {
				output, err = chatCompleteGemini(ctx, gemini_apikey, tc.Input.Model, b)
			}

			if err != nil {
				t.Fatalf("Failed to unmarshal test cases: %v", err)
			}

			expectedJSON, err := json.Marshal(tc.Output)
			if err != nil {
				t.Fatalf("Failed to marshal expected response: %v", err)
			}

			// skip these fields
			for _, f := range []string{"id", "created", "service_tier", "usage.prompt_tokens_details.audio_tokens", "usage.completion_tokens_details", "system_fingerprint",
				"choices.0.index", "choices.0.logprobs", "choices.0.message.refusal", "choices.0.message.annotations", "choices.0.message.tool_call_id",
				"choices.1.index", "choices.1.logprobs", "choices.1.message.refusal", "choices.1.message.annotations", "choices.1.message.tool_call_id",
				"choices.2.index", "choices.2.logprobs", "choices.2.message.refusal", "choices.2.message.annotations", "choices.2.message.tool_call_id",
				"choices.3.index", "choices.3.logprobs", "choices.3.message.refusal", "choices.3.message.annotations", "choices.3.message.tool_call_id",
				"choices.0.message.tool_calls.0.id", "choices.0.message.tool_calls.1.id", "choices.0.message.tool_calls.2.id", "choices.0.message.tool_calls.3.id",
				"choices.1.message.tool_calls.0.id", "choices.1.message.tool_calls.1.id", "choices.1.message.tool_calls.2.id", "choices.1.message.tool_calls.3.id",
				"choices.2.message.tool_calls.0.id", "choices.2.message.tool_calls.1.id", "choices.2.message.tool_calls.2.id", "choices.2.message.tool_calls.3.id",
			} {
				// Remove the 'email' field
				output, err = sjson.DeleteBytes(output, f)
				if err != nil {
					t.Fatal(err)
				}

				expectedJSON, err = sjson.DeleteBytes(expectedJSON, f)
				if err != nil {
					t.Fatal(err)
				}
			}

			var actualMap, expectedMap map[string]interface{}
			if err := json.Unmarshal(output, &actualMap); err != nil {
				t.Fatalf("Failed to unmarshal actual JSON: %v", err)
			}
			if err := json.Unmarshal(expectedJSON, &expectedMap); err != nil {
				t.Fatalf("Failed to unmarshal expected JSON: %v", err)
			}

			// 5. Compare the maps.
			if !cmp.Equal(expectedMap, actualMap) {
				t.Errorf("Response JSON mismatch (-want +got):\n%s", cmp.Diff(expectedMap, actualMap))
			}
		})
	}
}

type CostTestCase struct {
	Cost  int64  `json:"cost"`
	Model string `json:"model"`
	Usage *Usage `json:"usage"`
}

func TestCost(t *testing.T) {
	// Read the test cases from the JSON file
	file, err := os.ReadFile("./testcases/costcalc_testcase.json")
	if err != nil {
		t.Fatalf("Failed to read test cases file: %v", err)
	}

	var testCases map[string]CostTestCase
	if err := json.Unmarshal(file, &testCases); err != nil {
		t.Fatalf("Failed to unmarshal test cases: %v", err)
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			output := CalculateCost(tc.Model, tc.Usage)
			if output != tc.Cost {
				t.Errorf("Testcase %s, expect %d, got %d", name, tc.Cost, output)
			}
		})
	}
}
