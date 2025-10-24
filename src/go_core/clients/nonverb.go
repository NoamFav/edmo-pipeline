// src/go_core/clients/nonverb.go
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

type NVSpeakerFeatures struct {
	TotalSpeakingDuration float64            `json:"total_speaking_duration"`
	TotalTurns            int                `json:"total_turns"`
	SpeechRatio           float64            `json:"speech_ratio"`
	MeanTurnDuration      float64            `json:"mean_turn_duration"`
	MedianTurnDuration    float64            `json:"median_turn_duration"`
	StdTurnDuration       float64            `json:"std_turn_duration"`
	MinTurnDuration       float64            `json:"min_turn_duration"`
	MaxTurnDuration       float64            `json:"max_turn_duration"`
	Percentiles           map[string]float64 `json:"percentiles"`
	InterruptionsMade     int                `json:"interruptions_made"`
	InterruptionsReceived int                `json:"interruptions_received"`
	InterruptedBy         map[string]int     `json:"interrupted_by"`
}
type NVConversation struct {
	NumSpeakers        int     `json:"num_speakers"`
	TotalSpeakingTime  float64 `json:"total_speaking_time"`
	OverlapDuration    float64 `json:"overlap_duration"`
	SilenceDuration    float64 `json:"silence_duration"`
	OverlapRatio       float64 `json:"overlap_ratio"`
	SilenceRatio       float64 `json:"silence_ratio"`
	TotalInterruptions int     `json:"total_interruptions"`
	InterruptionRate   float64 `json:"interruption_rate"`
}
type NVSpeakerSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}
type NVDiarization struct {
	Segments    []NVSpeakerSegment `json:"segments"`
	NumSpeakers int                `json:"num_speakers"`
}
type NVBasicMetricsReq struct {
	Diarization NVDiarization `json:"diarization"`
	ConvLength  float64       `json:"conv_length"`
	Percentiles []int         `json:"percentiles,omitempty"`
}
type NVBasicMetricsResp struct {
	Speakers     map[string]NVSpeakerFeatures `json:"speakers"`
	Conversation NVConversation               `json:"conversation"`
}

func (h *HTTP) NonverbBasicMetrics(ctx context.Context, baseURL string, req NVBasicMetricsReq) (*NVBasicMetricsResp, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("nonverb marshal: %w", err)
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/basic_metrics", bytes.NewReader(b))
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
		return nil, fmt.Errorf("nonverb %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var out NVBasicMetricsResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("nonverb decode: %w", err)
	}
	return &out, nil
}
