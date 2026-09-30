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
				"delay":  {Type: "string", Enum: []string{"", "5m", "10m", "30m", "1h", "2h", "3h", "4h", "6h", "12h", "24h", "2d", "3d", "4d", "6d", "7d"}},
				"action": {Type: "string", Enum: []string{"skip", "delay", "customer_time"}},
				"timing": {
					Type: "object",
					Properties: map[string]*header.JSONSchema{
						"delay": {Type: "string", Enum: []string{"", "10m", "24h"}},
					},
					Required: []string{"delay"},
				},
				"owners": {Type: "array", Items: &header.JSONSchema{Type: "string", Enum: []string{"", "customer", "seller"}}},
			},
			Required: []string{"action", "delay", "timing", "owners"},
		},
	}
	for name, input := range map[string]CompletionInput{
		"response_format": {Model: Gemini_2_5_flash_lite, ResponseFormat: &ResponseFormat{Type: "json_schema", JSONSchema: schema}},
		"json_schema":     {Model: Gemini_2_5_flash, JSONSchema: schema},
		"json_schema_overrides_response_format": {
			Model: Gemini_2_5_flash, JSONSchema: schema,
			ResponseFormat: &ResponseFormat{Type: "json_schema", JSONSchema: &header.LLMResponseJSONSchemaFormat{Schema: &header.JSONSchema{Type: "string"}}},
		},
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
			if request.GenerationConfig == nil || request.GenerationConfig.ResponseJSONSchema == nil {
				t.Fatal("missing response schema")
			}
			var wire struct {
				Config map[string]json.RawMessage `json:"generationConfig"`
			}
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			if _, ok := wire.Config["responseSchema"]; ok {
				t.Fatal("responseSchema must be omitted when using responseJsonSchema")
			}
			if _, ok := wire.Config["responseJsonSchema"]; !ok {
				t.Fatal("missing responseJsonSchema on the wire")
			}
			if request.GenerationConfig.ResponseMIMEType != "application/json" {
				t.Fatal("JSON schema requires application/json response MIME type")
			}
			// Compare wire JSON to avoid protobuf internal implementation fields.
			want, _ := json.Marshal(schema.Schema)
			got, _ := json.Marshal(request.GenerationConfig.ResponseJSONSchema)
			if diff := cmp.Diff(string(want), string(got)); diff != "" {
				t.Fatalf("response schema changed (-want +got):\n%s", diff)
			}
		})
	}
}
