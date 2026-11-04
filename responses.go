package airouter

import (
	"context"
	"encoding/json"

	"github.com/subiz/header"
	"github.com/subiz/log"
)

// outputItemsField is the LLMChatHistoryEntry.Fields key holding the Responses API output items
// (encrypted reasoning, message, function calls, in order) of an assistant message, so they can be
// sent back as is in the next tool turn
const outputItemsField = "openai_response_output_items"

// useOpenAIResponsesAPI reports whether the request must be sent to /v1/responses instead of
// /v1/chat/completions. gpt-6-luna always uses the Responses API, gpt-5.4 models
// don't support function tools with reasoning_effort in /v1/chat/completions
func useOpenAIResponsesAPI(req CompletionInput) bool {
	model := ToOpenAIModel(req.Model)
	if model == Gpt_6_luna {
		return true
	}

	if len(req.Tools) == 0 {
		return false
	}

	if effort := resolveEffort(req); effort == "" || effort == "none" {
		return false
	}

	return model == Gpt_5_4_mini || model == Gpt_5_4_nano
}

// ToOpenAIResponsesJSON converts a CompletionInput to an OpenAI /v1/responses request
func ToOpenAIResponsesJSON(req CompletionInput) ([]byte, error) {
	model := ToOpenAIModel(req.Model)
	out := map[string]any{
		"model":   model,
		"store":   false,
		"include": []string{"reasoning.encrypted_content"},
	}

	// gpt-5.6+ models replace prompt_cache_retention with prompt_cache_options, we keep their default
	// 30m ttl since 12h cache writes cost 2.5x the input price (vs 1.25x for 30m)
	if model != Gpt_6_luna {
		out["prompt_cache_retention"] = "24h"
	}
	if req.PromptCacheKey != "" {
		out["prompt_cache_key"] = req.PromptCacheKey
	}
	if req.ServiceTier != "" {
		out["service_tier"] = req.ServiceTier
	}
	if req.ToolChoice != "" {
		out["tool_choice"] = req.ToolChoice
	}
	if req.MaxCompletionTokens > 0 {
		out["max_output_tokens"] = req.MaxCompletionTokens
	}

	// stop and seed are not supported by the Responses API
	effort := resolveEffort(req)
	if effort != "" {
		out["reasoning"] = map[string]any{"effort": effort}
	}

	// sampling params are only supported without reasoning
	if effort == "none" {
		if req.Temperature != 0 {
			out["temperature"] = req.Temperature
		}
		if req.TopP != 0 {
			out["top_p"] = req.TopP
		}
	}

	text := map[string]any{}
	if req.Verbosity != "" {
		text["verbosity"] = req.Verbosity
	}
	if format := toOpenAIJSONSchemaFormat(req); format != nil {
		format["type"] = "json_schema"
		text["format"] = format
	} else if req.ResponseFormat != nil && req.ResponseFormat.Type != "" {
		text["format"] = map[string]any{"type": req.ResponseFormat.Type} // json_object, text
	}
	if len(text) > 0 {
		out["text"] = text
	}

	tools := []any{}
	for _, tool := range req.Tools {
		if tool.Type != "function" || tool.Function == nil {
			tools = append(tools, map[string]any{"type": tool.Type})
			continue
		}
		newtool := map[string]any{
			"type":        "function",
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"strict":      false,
		}
		if tool.Function.Parameters != nil {
			newtool["parameters"] = toOpenAIToolParameters(tool.Function.Parameters)
		}
		tools = append(tools, newtool)
	}
	if len(tools) > 0 {
		out["tools"] = tools
	}

	input := []any{}
	if req.Instruct != "" {
		input = append(input, map[string]any{"role": "system", "content": req.Instruct})
	}
	for _, msg := range req.Messages {
		role := normalizeOpenAIRole(msg.GetRole())
		if role == "tool" {
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": msg.GetToolCallId(),
				"output":  toResponsesText(msg),
			})
			continue
		}

		if itemsstr := msg.GetFields()[outputItemsField]; itemsstr != "" && len(msg.GetToolCalls()) > 0 {
			items := []json.RawMessage{}
			if err := json.Unmarshal([]byte(itemsstr), &items); err == nil && len(items) > 0 {
				// replay the original output, reasoning items must stay right before their following item
				for _, item := range items {
					input = append(input, item)
				}
				continue
			}
		}

		if content := toResponsesContent(role, msg); content != nil {
			input = append(input, map[string]any{"role": role, "content": content})
		}

		for _, call := range msg.GetToolCalls() {
			input = append(input, map[string]any{
				"type":      "function_call",
				"call_id":   call.GetId(),
				"name":      call.GetFunction().GetName(),
				"arguments": call.GetFunction().GetArguments(),
			})
		}
	}
	out["input"] = input
	return json.Marshal(out)
}

// toResponsesContent converts a message content to a Responses API message content,
// returns nil if the message has no content
func toResponsesContent(role string, msg *header.LLMChatHistoryEntry) any {
	if len(msg.GetContents()) == 0 {
		if msg.GetContent() == "" {
			return nil
		}
		return msg.GetContent()
	}

	texttype := "input_text"
	if role == "assistant" {
		texttype = "output_text"
	}
	parts := []any{}
	for _, part := range msg.GetContents() {
		switch part.GetType() {
		case "text":
			parts = append(parts, map[string]any{"type": texttype, "text": part.GetText()})
		case "image_url":
			image := map[string]any{"type": "input_image", "image_url": part.GetImageUrl().GetUrl()}
			if detail := part.GetImageUrl().GetDetail(); detail != "" {
				image["detail"] = detail
			}
			parts = append(parts, image)
		}
	}
	if len(parts) == 0 {
		return nil
	}
	return parts
}

// toResponsesText returns the text of a message, contents take precedence over content
func toResponsesText(msg *header.LLMChatHistoryEntry) string {
	if len(msg.GetContents()) == 0 {
		return msg.GetContent()
	}
	text := ""
	for _, part := range msg.GetContents() {
		text += part.GetText()
	}
	return text
}

type OpenAIResponsesOutputContent struct {
	Type    string `json:"type"` // output_text, refusal
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

type OpenAIResponsesOutputItem struct {
	Type      string                         `json:"type"` // reasoning, message, function_call
	Role      string                         `json:"role,omitempty"`
	Content   []OpenAIResponsesOutputContent `json:"content,omitempty"`
	CallId    string                         `json:"call_id,omitempty"`
	Name      string                         `json:"name,omitempty"`
	Arguments string                         `json:"arguments,omitempty"`
}

type OpenAIResponsesUsage struct {
	InputTokens        int64                `json:"input_tokens"`
	InputTokensDetails *PromptTokensDetails `json:"input_tokens_details,omitempty"`
	OutputTokens       int64                `json:"output_tokens"`
	TotalTokens        int64                `json:"total_tokens"`
}

// OpenAIResponsesResponse mimics the structure of an OpenAI /v1/responses response
type OpenAIResponsesResponse struct {
	ID                string            `json:"id,omitempty"`
	CreatedAt         int64             `json:"created_at,omitempty"`
	Model             string            `json:"model,omitempty"`
	Status            string            `json:"status,omitempty"` // completed, incomplete, failed
	ServiceTier       string            `json:"service_tier,omitempty"`
	Output            []json.RawMessage `json:"output,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"` // max_output_tokens, content_filter
	} `json:"incomplete_details,omitempty"`
	Usage *OpenAIResponsesUsage `json:"usage,omitempty"`
	Error *OpenAIError          `json:"error,omitempty"`
}

// toOpenAIChatResponseFromResponses converts a /v1/responses response to the chat completion format
func toOpenAIChatResponseFromResponses(res *OpenAIResponsesResponse) *OpenAIChatResponse {
	out := &OpenAIChatResponse{
		ID:          res.ID,
		Created:     res.CreatedAt,
		Object:      "chat.completion",
		Model:       res.Model,
		ServiceTier: res.ServiceTier,
		Error:       res.Error,
	}

	if res.Usage != nil {
		out.Usage = &Usage{
			PromptTokens:        res.Usage.InputTokens,
			CompletionTokens:    res.Usage.OutputTokens,
			TotalTokens:         res.Usage.TotalTokens,
			PromptTokensDetails: res.Usage.InputTokensDetails,
		}
	}

	if res.Status == "failed" {
		if out.Error == nil {
			out.Error = &OpenAIError{Message: "response failed", Type: "server_error"}
		}
		return out
	}

	msg := OpenAIChatMessage{Role: "assistant"}
	text := ""
	hasreasoning := false
	for _, raw := range res.Output {
		item := OpenAIResponsesOutputItem{}
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		switch item.Type {
		case "reasoning":
			hasreasoning = true
		case "message":
			for _, c := range item.Content {
				text += c.Text
				msg.Refusal += c.Refusal
			}
		case "function_call":
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:       item.CallId,
				Type:     "function",
				Function: ToolFunction{Name: item.Name, Arguments: item.Arguments},
			})
		}
	}
	if text != "" || len(msg.ToolCalls) == 0 {
		msg.Content = &text
	}
	if hasreasoning {
		msg.OutputItems = res.Output
	}

	finishreason := "stop"
	if len(msg.ToolCalls) > 0 {
		finishreason = "tool_calls"
	}
	if res.Status == "incomplete" && res.IncompleteDetails != nil {
		finishreason = res.IncompleteDetails.Reason
		if finishreason == "max_output_tokens" {
			finishreason = "length"
		}
	}

	out.Choices = []OpenAIChoice{{Index: 0, Message: msg, FinishReason: finishreason}}
	return out
}

// chatCompleteOpenAIResponses sends the request to /v1/responses and returns a chat completion JSON
func chatCompleteOpenAIResponses(ctx context.Context, apikey string, request CompletionInput) ([]byte, error) {
	requestb, err := ToOpenAIResponsesJSON(request)
	if err != nil {
		return nil, err
	}
	output, err := postOpenAI(ctx, "https://api.openai.com/v1/responses", apikey, requestb, "responses")
	if err != nil {
		return output, err
	}

	res := &OpenAIResponsesResponse{}
	if err := json.Unmarshal(output, res); err != nil {
		return nil, log.EProvider(err, "openai", "responses", log.M{"_payload": output})
	}

	chat := toOpenAIChatResponseFromResponses(res)
	chatb, _ := json.Marshal(chat)
	if chat.Error != nil {
		return chatb, log.EProvider(nil, "openai", "responses", log.M{"_payload": output})
	}
	return chatb, nil
}
