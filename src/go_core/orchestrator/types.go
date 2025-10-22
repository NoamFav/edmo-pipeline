package orchestrator

type Utterance struct {
	Start float64 // sec
	End   float64 // sec
	Text  string
	Spk   string // "SPEAKER_0"...
}

type EmotionScore struct {
	Label string
	Score float64
}

type Window struct {
	T0, T1 float64
	Utts   []Utterance
	// Aggregates
	SpeakingShare map[string]float64 // per speaker %
	AvgPitch      float64            // optional later
	OverlapRate   float64
	Emotions      map[string]float64 // label -> mean score
	// Vector features for clustering
	Vector []float64
}
