package orchestrator

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"sort"

	"github.com/maastricht-university/edmo-pipeline/clients"
	cfg "github.com/maastricht-university/edmo-pipeline/config"
)

type Pipeline struct {
	cfg  *cfg.Root
	http *clients.HTTP
}

func NewPipeline(c *cfg.Root) *Pipeline {
	return &Pipeline{cfg: c, http: clients.NewHTTP()}
}

func (p *Pipeline) Run(ctx context.Context, wavPath string) error {
	// ASR
	log.Printf("ASR URL: %s", p.cfg.Services.ASR.URL)
	asr, err := p.http.ASR(ctx, p.cfg.Services.ASR.URL, wavPath)
	if err != nil {
		return err
	}
	log.Printf("ASR segments: %d", len(asr.Segments))
	if len(asr.Segments) == 0 {
		return fmt.Errorf("ASR returned no segments; check language/model or audio format")
	}
	// ensure chronological order
	sort.Slice(asr.Segments, func(i, j int) bool { return asr.Segments[i].Start < asr.Segments[j].Start })

	// Utterances
	utts := make([]Utterance, 0, len(asr.Segments))
	for _, s := range asr.Segments {
		utts = append(utts, Utterance{Start: s.Start, End: s.End, Text: s.Text, Spk: ""})
	}

	// Windows
	windows := p.window(utts)
	log.Printf("windows: %d", len(windows))
	if len(windows) == 0 {
		log.Println("no windows produced; skipping clustering/viz")
		return nil
	}

	// Per-utterance NLP + Emotion (best-effort)
	for wi := range windows {
		for ui := range windows[wi].Utts {
			u := &windows[wi].Utts[ui]
			if p.cfg.Services.NLP.URL != "" && u.Text != "" {
				if _, err := p.http.NLP(ctx, p.cfg.Services.NLP.URL, u.Text); err != nil {
					log.Printf("nlp error: %v", err)
				}
			}
			if p.cfg.Services.Emotion.URL != "" && u.Text != "" {
				emo, err := p.http.Emotion(ctx, p.cfg.Services.Emotion.URL, u.Text)
				if err == nil {
					for _, e := range emo.Emotions {
						if windows[wi].Emotions == nil {
							windows[wi].Emotions = map[string]float64{}
						}
						windows[wi].Emotions[e.Label] += e.Score
					}
				} else {
					log.Printf("emotion error: %v", err)
				}
			}
		}
		// aggregate + feature vector
		p.aggregate(&windows[wi])
		windows[wi].Vector = p.toVector(windows[wi])
	}

	// Prepare features
	features := make([][]float64, 0, len(windows))
	for _, w := range windows {
		features = append(features, w.Vector)
	}
	if len(features) == 0 {
		log.Println("no feature vectors; skipping clustering/viz")
		return nil
	}
	log.Printf("features: windows=%d dim=%d", len(features), len(features[0]))

	// Clustering
	clus, err := p.http.Cluster(ctx, p.cfg.Services.Clustering.URL, features, 5, 3, "PCA")
	if err != nil {
		return err
	}

	// Persist
	sid, winJSON, cluJSON, err := persist(p.cfg.Paths.Outputs, wavPath, windows, clus.ClusterLabels, clus.MembershipMatrix)
	if err != nil {
		return err
	}
	log.Printf("📦 saved: %s, %s (session=%s)", winJSON, cluJSON, sid)
	outDir := filepath.Dir(winJSON)

	// Viz: timeline
	timestamps := make([]float64, len(windows))
	for i, w := range windows {
		timestamps[i] = w.T0
	}
	tl, err := p.http.GenerateTimeline(ctx, p.cfg.Services.Visualization.URL, clients.TimelineReq{
		Timestamps: timestamps,
		Clusters:   clus.ClusterLabels,
		OutputDir:  outDir,
	})
	if err != nil {
		log.Printf("viz timeline error: %v", err)
	} else {
		log.Printf("🖼 timeline: %s", tl.Path)
	}

	// Viz: radar (avg of vector dims; keep in sync with toVector())
	var sum [8]float64 // [overlap, meanLen, joy, sadness, anger, surprise, fear, neutral]
	for _, w := range windows {
		vec := w.Vector
		if len(vec) >= 8 {
			for i := 0; i < 8; i++ {
				sum[i] += vec[i]
			}
		}
	}
	n := float64(len(windows))
	if n == 0 {
		n = 1
	}
	avg := []float64{
		sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n,
		sum[4] / n, sum[5] / n, sum[6] / n, sum[7] / n,
	}
	cats := []string{"overlap", "mean_utt", "joy", "sadness", "anger", "surprise", "fear", "neutral"}
	rad, err := p.http.GenerateRadar(ctx, p.cfg.Services.Visualization.URL, clients.RadarReq{
		Categories:  cats,
		Values:      avg,
		StudentName: "EDMO Session",
		OutputDir:   outDir,
	})
	if err != nil {
		log.Printf("viz radar error: %v", err)
	} else {
		log.Printf("🖼 radar: %s", rad.Path)
	}

	return nil
}
