package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type TransSeg struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}
type ASRResp struct {
	Segments []TransSeg `json:"segments"`
	Language string     `json:"language"`
}

func (h *HTTP) ASR(ctx context.Context, url, wavPath string) (*ASRResp, error) {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	fw, err := w.CreateFormFile("file", filepath.Base(wavPath))
	if err != nil {
		return nil, fmt.Errorf("create form file: %w", err)
	}
	fd, err := os.Open(wavPath)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", wavPath, err)
	}
	defer fd.Close()

	if _, err = io.Copy(fw, fd); err != nil {
		return nil, fmt.Errorf("copy audio: %w", err)
	}
	if err = w.Close(); err != nil {
		return nil, fmt.Errorf("close multipart: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/transcribe", &b)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Expect", "100-continue")

	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		const maxErr = 4096
		lb := io.LimitReader(resp.Body, maxErr)
		body, _ := io.ReadAll(lb)
		return nil, fmt.Errorf("asr %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var out ASRResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("asr decode: %w", err)
	}
	return &out, nil
}
