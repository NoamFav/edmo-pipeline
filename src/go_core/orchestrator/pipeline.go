package orchestrator

import (
	"context"
	"errors"
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

func ctxErr(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

func (p *Pipeline) Run(ctx context.Context, wavPath string) error {
	// ASR
	log.Printf("ASR URL: %s", p.cfg.Services.ASR.URL)
	asr, err := p.http.ASR(ctx, p.cfg.Services.ASR.URL, wavPath)
	if err != nil {
		return err
	}
	log.Printf("ASR segments: %d (lang=%s)", len(asr.Segments), asr.Language)
	if err := ctxErr(ctx); err != nil {
		return err
	}
	if len(asr.Segments) == 0 {
		return fmt.Errorf("ASR returned no segments")
	}
	sort.Slice(asr.Segments, func(i, j int) bool { return asr.Segments[i].Start < asr.Segments[j].Start })

	// Diarization (optional)
	var diar *clients.DiarResp
	if p.cfg.Services.Diarization.URL != "" {
		diar, err = p.http.Diarize(ctx, p.cfg.Services.Diarization.URL, wavPath)
		if err := ctxErr(ctx); err != nil {
			return err
		}
		if err != nil {
			log.Printf("diarization error: %v", err)
		} else {
			log.Printf("diarization: %d speakers, %d segments", diar.NumSpeakers, len(diar.Segments))
		}
	}

	// Utterances from ASR
	utts := make([]Utterance, 0, len(asr.Segments))
	for _, s := range asr.Segments {
		utts = append(utts, Utterance{Start: s.Start, End: s.End, Text: s.Text})
	}

	// Assign speakers if diarization present
	if diar != nil && len(diar.Segments) > 0 {
		assignSpeakers(utts, diar.Segments)
	}

	// Windowing
	windows := p.window(utts)
	log.Printf("windows: %d", len(windows))
	if len(windows) == 0 {
		log.Println("no windows; stopping")
		return nil
	}

	// NLP + Emotion + feature extraction
	for wi := range windows {
		for ui := range windows[wi].Utts {
			u := &windows[wi].Utts[ui]

			if p.cfg.Services.NLP.URL != "" && u.Text != "" {
				if err := ctxErr(ctx); err != nil {
					return err
				}
				if _, err := p.http.NLP(ctx, p.cfg.Services.NLP.URL, u.Text); err != nil {
					log.Printf("nlp error: %v", err)
				}
			}
			if p.cfg.Services.Emotion.URL != "" && u.Text != "" {
				if err := ctxErr(ctx); err != nil {
					return err
				}
				emo, err := p.http.Emotion(ctx, p.cfg.Services.Emotion.URL, u.Text)
				if err == nil {
					if windows[wi].Emotions == nil {
						windows[wi].Emotions = map[string]float64{}
					}
					for _, e := range emo.Emotions {
						windows[wi].Emotions[e.Label] += e.Score
					}
				} else {
					log.Printf("emotion error: %v", err)
				}
			}
		}
		p.aggregate(&windows[wi])
		windows[wi].Vector = p.toVector(windows[wi])
	}

	// Non-verbal metrics (if diarization present)
	if p.cfg.Services.Nonverb.URL != "" && diar != nil {
		if err := ctxErr(ctx); err != nil {
			return err
		}
		convLen := 0.0
		if len(utts) > 0 {
			convLen = utts[len(utts)-1].End - utts[0].Start
		}
		nvSegs := make([]clients.NVSpeakerSegment, len(diar.Segments))
		for i, s := range diar.Segments {
			nvSegs[i] = clients.NVSpeakerSegment{Start: s.Start, End: s.End, Speaker: s.Speaker}
		}
		nvReq := clients.NVBasicMetricsReq{
			Diarization: clients.NVDiarization{Segments: nvSegs, NumSpeakers: diar.NumSpeakers},
			ConvLength:  convLen,
			Percentiles: []int{10, 25, 75, 90},
		}
		nv, err := p.http.NonverbBasicMetrics(ctx, p.cfg.Services.Nonverb.URL, nvReq)
		if err != nil {
			log.Printf("nonverbal metrics error: %v", err)
		} else {
			log.Printf("NV Metrics: speakers=%d overlap_ratio=%.3f silence_ratio=%.3f interruptions=%d",
				nv.Conversation.NumSpeakers,
				nv.Conversation.OverlapRatio,
				nv.Conversation.SilenceRatio,
				nv.Conversation.TotalInterruptions,
			)
		}
	}

	// Feature matrix -> clustering
	features := make([][]float64, 0, len(windows))
	for _, w := range windows {
		features = append(features, w.Vector)
	}
	if len(features) == 0 {
		log.Println("no feature vectors; stopping")
		return nil
	}

	featDim := 0
	if len(features) > 0 && len(features[0]) > 0 {
		featDim = len(features[0])
	}
	log.Printf("feature matrix: %d x %d", len(features), featDim)

	if p.cfg.Services.Clustering.URL == "" {
		return errors.New("clustering service URL not configured")
	}
	if err := ctxErr(ctx); err != nil {
		return err
	}

	// pass "PCA" to use the new dim-reduction arg; omit it to use server default
	clus, err := p.http.Cluster(ctx, p.cfg.Services.Clustering.URL, features, 5, 3, "PCA")
	if err != nil {
		return err
	}
	if err := ctxErr(ctx); err != nil {
		return err
	}
	// Persist
	sid, winJSON, cluJSON, err := persist(p.cfg.Paths.Outputs, wavPath, windows, clus.ClusterLabels, clus.MembershipMatrix)
	if err != nil {
		return err
	}
	log.Printf("saved session %s", sid)
	log.Printf("windows: %s", winJSON)
	log.Printf("clusters: %s", cluJSON)

	outDir := filepath.Dir(winJSON)

	// Visualization: timeline
	if p.cfg.Services.Visualization.URL != "" {
		if err := ctxErr(ctx); err != nil {
			return err
		}
		timestamps := make([]float64, len(windows))
		for i, w := range windows {
			timestamps[i] = w.T0
		}
		_, err = p.http.GenerateTimeline(ctx, p.cfg.Services.Visualization.URL, clients.TimelineReq{
			Timestamps: timestamps,
			Clusters:   clus.ClusterLabels,
			OutputDir:  outDir,
		})
		if err != nil {
			log.Printf("timeline viz error: %v", err)
		}
	}

	// Visualization: radar
	if p.cfg.Services.Visualization.URL != "" {
		if err := ctxErr(ctx); err != nil {
			return err
		}
		var sum [8]float64
		for _, w := range windows {
			if len(w.Vector) >= 8 {
				for i := 0; i < 8; i++ {
					sum[i] += w.Vector[i]
				}
			}
		}
		n := float64(len(windows))
		if n == 0 {
			n = 1
		}
		avg := make([]float64, 8)
		for i := 0; i < 8; i++ {
			avg[i] = sum[i] / n
		}
		cats := []string{"overlap", "mean_utt", "joy", "sadness", "anger", "surprise", "fear", "neutral"}
		_, err = p.http.GenerateRadar(ctx, p.cfg.Services.Visualization.URL, clients.RadarReq{
			Categories:  cats,
			Values:      avg,
			StudentName: "EDMO Session",
			OutputDir:   outDir,
		})
		if err != nil {
			log.Printf("radar viz error: %v", err)
		}
	}

	// Explained-variance chart
	if p.cfg.Services.Visualization.URL != "" {
		if err := ctxErr(ctx); err != nil {
			return err
		}
		_, err = p.http.GenerateVarianceChart(
			ctx,
			p.cfg.Services.Visualization.URL,
			clients.VarianceChartReq{
				TotalVariance:        clus.ExplainedVariance,
				VariancePerDimension: clus.ExplainedVariancePerDimension,
				ReductionUsed:        clus.ReductionUsed, // may be "", that’s OK
				OutputDir:            outDir,
			},
		)
		if err != nil {
			log.Printf("variance viz error: %v", err)
		}
	}

	return nil
}
