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

// --- Visualization ---
type TimelineReq struct {
	Timestamps    []float64 `json:"timestamps"`
	Clusters      []int     `json:"clusters"`
	RobotProgress []float64 `json:"robot_progress,omitempty"`
	OutputDir     string    `json:"output_dir,omitempty"`
}



type TimelineResp struct{ Status, Path string }

func (h *HTTP) GenerateTimeline(ctx context.Context, url string, req TimelineReq) (*TimelineResp, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("timeline marshal: %w", err)
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/generate-timeline", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	resp, err := h.c.Do(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		const maxErr = 4096
		lb := io.LimitReader(resp.Body, maxErr)
		body, _ := io.ReadAll(lb)
		return nil, fmt.Errorf("viz timeline %s: %s",
			resp.Status, strings.TrimSpace(string(body)))
	}

	var out TimelineResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("viz timeline decode: %w", err)
	}
	return &out, nil
}

type RadarReq struct {
	Categories  []string  `json:"categories"`
	Values      []float64 `json:"values"`
	StudentName string    `json:"student_name"`
	OutputDir   string    `json:"output_dir,omitempty"`
}
type RadarResp struct{ Status, Path string }

func (h *HTTP) GenerateRadar(ctx context.Context, url string, req RadarReq) (*RadarResp, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("radar marshal: %w", err)
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/generate-radar", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	resp, err := h.c.Do(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		const maxErr = 4096
		lb := io.LimitReader(resp.Body, maxErr)
		body, _ := io.ReadAll(lb)
		return nil, fmt.Errorf("viz radar %s: %s",
			resp.Status, strings.TrimSpace(string(body)))
	}

	var out RadarResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("viz radar decode: %w", err)
	}
	return &out, nil
}
