package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
)

// --- Emotion (/detect) ---
type EmoReq struct {
	Text string `json:"text"`
}
type EmoScore struct {
	Label string
	Score float64
}
type EmoResp struct {
	Emotions        []EmoScore
	DominantEmotion string
}

func (h *HTTP) Emotion(ctx context.Context, url, text string) (*EmoResp, error) {
	b, _ := json.Marshal(EmoReq{Text: text})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url+"/detect", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out EmoResp
	return &out, json.NewDecoder(resp.Body).Decode(&out)
}
