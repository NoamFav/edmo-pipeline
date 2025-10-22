package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
)

// --- NLP (/analyze) ---
type NLPReq struct {
	Text string `json:"text"`
}
type Strategy struct{ Type, Text string }
type NLPResp struct {
	Strategies []Strategy `json:"strategies"`
}

func (h *HTTP) NLP(ctx context.Context, url, text string) (*NLPResp, error) {
	payload, _ := json.Marshal(NLPReq{Text: text})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url+"/analyze", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out NLPResp
	return &out, json.NewDecoder(resp.Body).Decode(&out)
}
