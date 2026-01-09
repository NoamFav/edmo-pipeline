package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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
	payload, err := json.Marshal(NLPReq{Text: text})
	if err != nil {
		return nil, fmt.Errorf("nlp marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/analyze", bytes.NewReader(payload))
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
		const maxErr = 4096
		lb := io.LimitReader(resp.Body, maxErr)
		b, _ := io.ReadAll(lb)
		return nil, fmt.Errorf("nlp %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	var out NLPResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("nlp decode: %w", err)
	}
	return &out, nil
}
