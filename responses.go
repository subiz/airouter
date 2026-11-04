package airouter

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/subiz/log"
)

// reasoningItemsField is the LLMChatHistoryEntry.Fields key holding the (encrypted) reasoning items
// of an assistant message, so they can be sent back to the Responses API in the next tool turn
const reasoningItemsField = "openai_reasoning_items"

// useOpenAIResponsesAPI reports whether the request must be sent to /v1/responses instead of
// /v1/chat/completions. gpt-6-luna always uses the Responses API (12h prompt cache), gpt-5.4 models
// don't support function tools with reasoning_effort in /v1/chat/completions
func useOpenAIResponsesAPI(req CompletionInput) bool {
	model := ToOpenAIModel(req.Model)
	if model == Gpt_6_luna {
		return true
	}

	if len(req.Tools) == 0 {
		return false
	}

	effort := req.ReasoningEffort
	if req.Reasoning != nil {
		effort = req.Reasoning.Effort
	}
	if effort == "" || effort == "none" {
		return false
	}

	return model == Gpt_5_4_mini || model == Gpt_5_4_nano
}

// ToOpenAIResponsesJSON converts a CompletionInput to an OpenAI /v1/responses request
func ToOpenAIResponsesJSON(req CompletionInput) ([]byte, error) {
	// reasoning items of assistant messages, keyed by their first tool call id
	reasoningM := map[string][]json.RawMessage{}
	for _, msg := range req.Messages {
		itemsstr := msg.GetFields()[reasoningItemsField]
		if itemsstr == "" || len(msg.GetToolCalls()) == 0 {
			continue
		}
		items := []json.RawMessage{}
		if err := json.Unmarshal([]byte(itemsstr), &items); err == nil {
			reasoningM[msg.GetToolCalls()[0].GetId()] = items
		}
	}

	// reuse the chat completion conversion, then reshape it
	chatb, err := ToOpenAICompletionJSON(req)
	if err != nil {
		return nil, err
	}
	m := map[string]any{}
	if err := json.Unmarshal(chatb, &m); err != nil {
		return nil, err
	}

	out := map[string]any{
		"model":   m["model"],
		"store":   false,
		"include": []string{"reasoning.encrypted_content"},
	}

	// stop and seed are not supported by the Responses API
	for _, key := range []string{"tool_choice", "service_tier", "prompt_cache_key", "prompt_cache_retention"} {
		if v, has := m[key]; has {
			out[key] = v
		}
	}

	if out["model"] == Gpt_6_luna {
		// gpt-5.6+ models replace prompt_cache_retention with prompt_cache_options
		delete(out, "prompt_cache_retention")
		out["prompt_cache_options"] = map[string]any{"ttl": "12h"}
	}

	if v, has := m["max_completion_tokens"]; has {
		out["max_output_tokens"] = v
	}

	// read from req, the chat completion conversion may drop reasoning_effort
	effort := req.ReasoningEffort
	if req.Reasoning != nil {
		effort = req.Reasoning.Effort
	}
	if effort != "" {
		out["reasoning"] = map[string]any{"effort": effort}
	}

	// sampling params are only supported without reasoning
	if effort == "none" {
		for _, key := range []string{"temperature", "top_p"} {
			if v, has := m[key]; has {
				out[key] = v
			}
		}
	}

	text := map[string]any{}
	if v, has := m["verbosity"]; has {
		text["verbosity"] = v
	}
	if rf, ok := m["response_format"].(map[string]any); ok {
		format := map[string]any{"type": rf["type"]}
		if js, ok := rf["json_schema"].(map[string]any); ok {
			for k, v := range js {
				format[k] = v
			}
		}
		text["format"] = format
	}
	if len(text) > 0 {
		out["text"] = text
	}

	tools := []any{}
	toolsi, _ := m["tools"].([]any)
	for _, tooli := range toolsi {
		tool, ok := tooli.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := tool["function"].(map[string]any)
		if tool["type"] != "function" || !ok {
			tools = append(tools, tool)
			continue
		}
		newtool := map[string]any{"type": "function", "strict": false}
		for k, v := range fn {
			newtool[k] = v
		}
		tools = append(tools, newtool)
	}
	if len(tools) > 0 {
		out["tools"] = tools
	}

	input := []any{}
	messagesi, _ := m["messages"].([]any)
	for _, msgi := range messagesi {
		msg, ok := msgi.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)

		if role == "tool" {
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": msg["tool_call_id"],
				"output":  toResponsesText(msg["content"]),
			})
			continue
		}

		toolcalls, _ := msg["tool_calls"].([]any)
		if len(toolcalls) > 0 {
			firstcall, _ := toolcalls[0].(map[string]any)
			firstid, _ := firstcall["id"].(string)
			for _, item := range reasoningM[firstid] {
				input = append(input, item)
			}
		}

		if content := toResponsesContent(role, msg["content"]); content != nil {
			input = append(input, map[string]any{"role": role, "content": content})
		}

		for _, calli := range toolcalls {
			call, _ := calli.(map[string]any)
			fn, _ := call["function"].(map[string]any)
			input = append(input, map[string]any{
				"type":      "function_call",
				"call_id":   call["id"],
				"name":      fn["name"],
				"arguments": fn["arguments"],
			})
		}
	}
	out["input"] = input
	return json.Marshal(out)
}

// toResponsesContent converts a chat completion message content to a Responses API message content,
// returns nil if the message has no content
func toResponsesContent(role string, contenti any) any {
	switch content := contenti.(type) {
	case string:
		if content == "" {
			return nil
		}
		return content
	case []any:
		texttype := "input_text"
		if role == "assistant" {
			texttype = "output_text"
		}
		parts := []any{}
		for _, parti := range content {
			part, _ := parti.(map[string]any)
			switch part["type"] {
			case "text":
				parts = append(parts, map[string]any{"type": texttype, "text": part["text"]})
			case "image_url":
				imageurl, _ := part["image_url"].(map[string]any)
				parts = append(parts, map[string]any{"type": "input_image", "image_url": imageurl["url"]})
			}
		}
		if len(parts) == 0 {
			return nil
		}
		return parts
	}
	return nil
}

func toResponsesText(contenti any) string {
	switch content := contenti.(type) {
	case string:
		return content
	case []any:
		text := ""
		for _, parti := range content {
			part, _ := parti.(map[string]any)
			if s, _ := part["text"].(string); s != "" {
				text += s
			}
		}
		return text
	}
	return ""
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
		return out
	}

	msg := OpenAIChatMessage{Role: "assistant"}
	text := ""
	for _, raw := range res.Output {
		item := OpenAIResponsesOutputItem{}
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		switch item.Type {
		case "reasoning":
			msg.ReasoningItems = append(msg.ReasoningItems, raw)
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
	url := "https://api.openai.com/v1/responses"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestb))
	if err != nil {
		return nil, log.EServer(err)
	}
	req.Header.Set("Authorization", "Bearer "+apikey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	output := buf.Bytes()

	if resp.StatusCode != 200 {
		// error body has the same {"error": {...}} shape as chat completion
		return output, log.EProvider(nil, "openai", "responses", log.M{"status": resp.StatusCode, "_payload": output})
	}

	res := &OpenAIResponsesResponse{}
	if err := json.Unmarshal(output, res); err != nil {
		return nil, log.EProvider(err, "openai", "responses", log.M{"_payload": output})
	}
	return json.Marshal(toOpenAIChatResponseFromResponses(res))
}
