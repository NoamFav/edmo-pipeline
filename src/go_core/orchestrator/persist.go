package orchestrator

import (
	"encoding/json"
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
		return "", "", err
	}
	return sid, dir, nil
}

func writeJSON(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
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
		Windows:     nil, // keep windows in windows.json only
		Clusters:    labels,
		Membership:  membership,
	}
	if err = writeJSON(cluPath, bundle); err != nil {
		return "", "", "", err
	}

	return sid, winPath, cluPath, nil
}
