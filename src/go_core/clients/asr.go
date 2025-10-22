package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

// --- ASR ---
type TransSeg struct {
	Start, End float64
	Text       string
}
type ASRResp struct {
	Segments []TransSeg
	Language string
}

func (h *HTTP) ASR(ctx context.Context, url, wavPath string) (*ASRResp, error) {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	fw, _ := w.CreateFormFile("file", filepath.Base(wavPath))
	fd, _ := os.Open(wavPath)
	defer fd.Close()
	_, _ = io.Copy(fw, fd)
	w.Close()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url+"/transcribe", &b)
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out ASRResp
	return &out, json.NewDecoder(resp.Body).Decode(&out)
}
