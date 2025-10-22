package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// --- Emotion (/detect) ---
type EmoReq struct {
	Text string `json:"text"`
}
type EmoScore struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}
type EmoResp struct {
	Emotions        []EmoScore `json:"emotions"`
	DominantEmotion string     `json:"dominant_emotion"`
}

func (h *HTTP) Emotion(ctx context.Context, url, text string) (*EmoResp, error) {
	b, _ := json.Marshal(EmoReq{Text: text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/detect", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("emotion %s: %s", resp.Status, string(body))
	}

	var out EmoResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("emotion decode: %w", err)
	}
	return &out, nil
}
