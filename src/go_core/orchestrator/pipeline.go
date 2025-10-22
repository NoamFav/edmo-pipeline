package orchestrator

import (
	"context"
	"log"

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
	asr, err := p.http.ASR(ctx, p.cfg.Services.ASR.URL, wavPath)
	if err != nil {
		return err
	}

	utts := make([]Utterance, 0, len(asr.Segments))
	for _, s := range asr.Segments {
		utts = append(utts, Utterance{Start: s.Start, End: s.End, Text: s.Text, Spk: ""})
	}

	windows := p.window(utts)

	for wi := range windows {
		for ui := range windows[wi].Utts {
			u := &windows[wi].Utts[ui]
			if p.cfg.Services.NLP.URL != "" && u.Text != "" {
				_, _ = p.http.NLP(ctx, p.cfg.Services.NLP.URL, u.Text)
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
				}
			}
		}
		// aggregate basic speaking share + overlap
		p.aggregate(&windows[wi])
		// flatten into a feature vector
		windows[wi].Vector = p.toVector(windows[wi])
	}

	features := make([][]float64, 0, len(windows))
	for _, w := range windows {
		features = append(features, w.Vector)
	}
	clus, err := p.http.Cluster(ctx, p.cfg.Services.Clustering.URL, features, 5, 3)
	if err != nil {
		return err
	}

	log.Printf("clusters=%v explained=%.2f", clus.ClusterLabels, clus.ExplainedVariance)
	// TODO: write JSON to cfg.Paths.Outputs + POST to visualization service

	return nil
}
