package airouter

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/subiz/header"
)

func TestGeminiResponseSchemaPreservesEnums(t *testing.T) {
	schema := &header.LLMResponseJSONSchemaFormat{
		Name:   "follow_up",
		Strict: true,
		Schema: &header.JSONSchema{
			Type: "object",
			Properties: map[string]*header.JSONSchema{
				"action": {Type: "string", Enum: []string{"skip", "delay", "customer_time"}},
				"timing": {
					Type: "object",
					Properties: map[string]*header.JSONSchema{
						"delay": {Type: "string", Enum: []string{"", "10m", "24h"}},
					},
					Required: []string{"delay"},
				},
				"owners": {Type: "array", Items: &header.JSONSchema{Type: "string", Enum: []string{"customer", "seller"}}},
			},
			Required: []string{"action", "timing", "owners"},
		},
	}
	for name, input := range map[string]CompletionInput{
		"response_format": {Model: Gemini_2_5_flash_lite, ResponseFormat: &ResponseFormat{Type: "json_schema", JSONSchema: schema}},
		"json_schema":     {Model: Gemini_2_5_flash_lite, JSONSchema: schema},
	} {
		t.Run(name, func(t *testing.T) {
			body, err := ToGeminiRequestJSON(input)
			if err != nil {
				t.Fatal(err)
			}
			var request GeminiRequest
			if err := json.Unmarshal([]byte(body), &request); err != nil {
				t.Fatal(err)
			}
			if request.GenerationConfig == nil || request.GenerationConfig.ResponseSchema == nil {
				t.Fatal("missing response schema")
			}
			// Compare wire JSON to avoid protobuf internal implementation fields.
			want, _ := json.Marshal(schema.Schema)
			got, _ := json.Marshal(request.GenerationConfig.ResponseSchema)
			if diff := cmp.Diff(string(want), string(got)); diff != "" {
				t.Fatalf("response schema changed (-want +got):\n%s", diff)
			}
		})
	}
}
