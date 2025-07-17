package airouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/subiz/header"
	"github.com/subiz/log"
)

// GeminiRequest represents the structure of a request to the Gemini API.
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
}

// GeminiGenerationConfig represents the generation configuration.
type GeminiGenerationConfig struct {
	CandidateCount   int                `json:"candidateCount"`
	Temperature      float32            `json:"temperature"`
	TopP             float32            `json:"topP"`
	ResponseMIMEType string             `json:"responseMimeType,omitempty"`
	ResponseSchema   *header.JSONSchema `json:"responseSchema,omitempty"`
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

// ToGeminiRequestJSON converts an OpenAIChatRequest to a Gemini-compatible JSON request string.
func ToGeminiRequestJSON(req OpenAIChatRequest) ([]byte, error) {
	var geminiTools []*GeminiTool
	for _, tool := range req.Tools {
		properties := make(map[string]*header.JSONSchema)
		for key, prop := range tool.Function.Parameters.Properties {
			properties[key] = &header.JSONSchema{
				Type:        prop.Type,
				Description: prop.Description,
			}
		}

		var decls []*GeminiFunctionDeclaration
		decls = append(decls, &GeminiFunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters: &header.JSONSchema{
				Type:       "object",
				Properties: properties,
				Required:   tool.Function.Parameters.Required,
			},
		})
		geminiTools = append(geminiTools, &GeminiTool{FunctionDeclarations: decls})
	}

	geminiReq := GeminiRequest{
		Model: "models/" + ToGeminiModel(req.Model),
		GenerationConfig: &GeminiGenerationConfig{
			CandidateCount: 1,
			Temperature:    req.Temperature,
			TopP:           req.TopP,
		},
		Tools: geminiTools,
	}

	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" {
		geminiReq.GenerationConfig.ResponseMIMEType = "application/json"
		schema := req.ResponseFormat.JSONSchema.Schema
		geminiSchema := &header.JSONSchema{
			Type:       schema.Type,
			Required:   schema.Required,
			Properties: make(map[string]*header.JSONSchema),
		}
		for key, prop := range schema.Properties {
			geminiProp := &header.JSONSchema{
				Type:        prop.Type,
				Description: prop.Description,
			}
			if prop.Items != nil {
				geminiProp.Items = &header.JSONSchema{
					Type:        prop.Items.Type,
					Description: prop.Items.Description,
				}
			}
			geminiSchema.Properties[key] = geminiProp
		}
		geminiReq.GenerationConfig.ResponseSchema = geminiSchema
	}

	var contents []*GeminiContent
	toolCallsByID := make(map[string]ToolCall)

	systemmsgs := []string{}
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			if msg.Content != nil {
				systemmsgs = append(systemmsgs, *msg.Content)
			}
		case "user":
			contents = append(contents, &GeminiContent{
				Parts: []*GeminiPart{{Text: msg.Content}},
				Role:  "user",
			})
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				var parts []*GeminiPart
				for _, tc := range msg.ToolCalls {
					toolCallsByID[tc.ID] = tc
					var args map[string]interface{}
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
			} else if msg.Content != nil {
				contents = append(contents, &GeminiContent{
					Parts: []*GeminiPart{{Text: msg.Content}},
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

	if len(contents) > 0 {
		geminiReq.Contents = contents
	}

	b, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Gemini request: %w", err)
	}
	return b, nil
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
	if res == nil || len(res.Candidates) == 0 {
		return nil, fmt.Errorf("empty response from Gemini")
	}
	responseID := res.ResponseId
	choice := OpenAIChoice{Index: 0}
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
		Object:  "chat.completion",
		Model:   res.ModelVersion,
		Choices: []OpenAIChoice{choice},
		Usage:   usage,
	}, nil
}

func chatCompleteGemini(ctx context.Context, apikey, model string, requestb []byte) ([]byte, error) {
	request := OpenAIChatRequest{}
	json.Unmarshal(requestb, &request)

	var err error
	requestb, err = ToGeminiRequestJSON(request)
	if err != nil {
		return nil, err
	}

	// fmt.Println("REA", string(requestb))
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + ToGeminiModel(model) + ":generateContent?key=" + apikey
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestb))
	if err != nil {
		return nil, log.EServer(err)
	}
	req.Header.Set("Content-Type", "application/json")
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

func strsToParts(strs []string) []*GeminiPart {
	out := []*GeminiPart{}
	for _, str := range strs {
		str = strings.TrimSpace(str)
		if str != "" {
			out = append(out, &GeminiPart{Text: &str})
		}
	}
	return out
}
