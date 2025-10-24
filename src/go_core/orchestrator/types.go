package orchestrator

type Utterance struct {
	Start float64 `json:"start"` // seconds
	End   float64 `json:"end"`   // seconds
	Text  string  `json:"text"`
	Spk   string  `json:"speaker"` // "SPEAKER_0"...
}

type EmotionScore struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

type Window struct {
	T0   float64     `json:"t0"` // window start (s)
	T1   float64     `json:"t1"` // window end (s)
	Utts []Utterance `json:"utts"`
	// Aggregates
	SpeakingShare map[string]float64 `json:"speaking_share,omitempty"` // per speaker %
	AvgPitch      float64            `json:"avg_pitch,omitempty"`      // reserved
	OverlapRate   float64            `json:"overlap_rate,omitempty"`
	Emotions      map[string]float64 `json:"emotions,omitempty"` // label -> mean score
	Vector        []float64          `json:"vector,omitempty"`   // features for clustering
}
