package orchestrator

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type PersistBundle struct {
	SessionID   string      `json:"session_id"`
	AudioPath   string      `json:"audio_path"`
	GeneratedAt time.Time   `json:"generated_at"`
	Windows     []Window    `json:"windows"`
	Clusters    []int       `json:"clusters"`
	Membership  [][]float64 `json:"membership_matrix,omitempty"`
}

func mkSessionDir(outputsRoot string) (string, string, error) {
	ts := time.Now().Format("20060102-150405")
	sid := "session_" + ts
	dir := filepath.Join(outputsRoot, sid)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", fmt.Errorf("make session dir: %w", err)
	}
	return sid, dir, nil
}

func writeJSON(path string, v any) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp for %s: %w", filepath.Base(path), err)
	}
	tmpName := tmp.Name()
	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("encode %s: %w", filepath.Base(path), err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("close temp %s: %w", filepath.Base(path), err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("rename %s: %w", filepath.Base(path), err)
	}
	return nil
}

func persist(outputsRoot, audioPath string, windows []Window, labels []int, membership [][]float64) (sessionID, windowsPath, clustersPath string, err error) {
	sid, outDir, err := mkSessionDir(outputsRoot)
	if err != nil {
		return "", "", "", err
	}

	winPath := filepath.Join(outDir, "windows.json")
	cluPath := filepath.Join(outDir, "clusters.json")

	if err = writeJSON(winPath, windows); err != nil {
		return "", "", "", err
	}

	bundle := PersistBundle{
		SessionID:   sid,
		AudioPath:   audioPath,
		GeneratedAt: time.Now(),
		Windows:     nil,
		Clusters:    labels,
		Membership:  membership,
	}
	if err = writeJSON(cluPath, bundle); err != nil {
		return "", "", "", err
	}

	return sid, winPath, cluPath, nil
}
