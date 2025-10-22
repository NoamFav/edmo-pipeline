package orchestrator

import (
	"math"
	"sort"
)

func (p *Pipeline) window(utts []Utterance) []Window {
	if len(utts) == 0 {
		return nil
	}
	// compute session bounds
	start := utts[0].Start
	end := utts[len(utts)-1].End
	w := float64(p.cfg.Features.TimeWindow)
	o := float64(p.cfg.Features.Overlap)
	step := w - o

	var out []Window
	for t0 := start; t0 < end; t0 += step {
		t1 := math.Min(t0+w, end)
		var slice []Utterance
		for _, u := range utts {
			if u.End <= t0 || u.Start >= t1 {
				continue
			}
			slice = append(slice, u)
		}
		out = append(out, Window{T0: t0, T1: t1, Utts: slice})
	}
	return out
}

func (p *Pipeline) aggregate(w *Window) {
	if len(w.Utts) == 0 {
		return
	}
	total := 0.0
	w.SpeakingShare = map[string]float64{}
	// naive overlap estimate and speaking time
	type edge struct {
		t     float64
		delta int
	}
	var edges []edge
	for _, u := range w.Utts {
		d := math.Max(0, u.End-u.Start)
		total += d
		w.SpeakingShare[u.Spk] += d
		edges = append(edges, edge{t: u.Start, delta: +1}, edge{t: u.End, delta: -1})
	}
	sort.Slice(edges, func(i, j int) bool { return edges[i].t < edges[j].t })
	active := 0
	last := edges[0].t
	overlap := 0.0
	for _, e := range edges {
		if active > 1 {
			overlap += e.t - last
		}
		active += e.delta
		last = e.t
	}
	if total > 0 {
		for k := range w.SpeakingShare {
			w.SpeakingShare[k] /= total
		}
	}
	winDur := w.T1 - w.T0
	if winDur > 0 {
		w.OverlapRate = overlap / winDur
	}
}

func (p *Pipeline) toVector(w Window) []float64 {
	meanLen := 0.0
	for _, u := range w.Utts {
		meanLen += (u.End - u.Start)
	}
	if n := float64(len(w.Utts)); n > 0 {
		meanLen /= n
	}
	// pick a stable order of a few common emotions
	keys := []string{"joy", "sadness", "anger", "surprise", "fear", "neutral"}
	vec := []float64{w.OverlapRate, meanLen}
	for _, k := range keys {
		vec = append(vec, w.Emotions[k])
	}
	return vec
}
