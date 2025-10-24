// src/go_core/clients/diarization.go
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

type SpkSeg struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}
type DiarResp struct {
	Segments    []SpkSeg `json:"segments"`
	NumSpeakers int      `json:"num_speakers"`
}

func (h *HTTP) Diarize(ctx context.Context, baseURL, audioPath string) (*DiarResp, error) {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	fw, err := w.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return nil, fmt.Errorf("create form file: %w", err)
	}
	fd, err := os.Open(audioPath)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", audioPath, err)
	}
	defer fd.Close()
	if _, err = io.Copy(fw, fd); err != nil {
		return nil, fmt.Errorf("copy audio: %w", err)
	}
	if err = w.Close(); err != nil {
		return nil, fmt.Errorf("close multipart: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/diarize", &b)
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
		return nil, fmt.Errorf("diarize %s: %s",
			resp.Status, strings.TrimSpace(string(body)))
	}
	var out DiarResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("diarize decode: %w", err)
	}
	return &out, nil
}
