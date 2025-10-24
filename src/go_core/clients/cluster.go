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

// --- Clustering (/cluster) ---
type ClusterReq struct {
	Features    [][]float64 `json:"features"`
	NClusters   int         `json:"n_clusters"`
	NComponents int         `json:"n_components"`
}
type ClusterResp struct {
	ClusterLabels     []int       `json:"cluster_labels"`
	MembershipMatrix  [][]float64 `json:"membership_matrix"`
	ReducedFeatures   [][]float64 `json:"reduced_features"`
	ExplainedVariance float64     `json:"explained_variance"`
}

func (h *HTTP) Cluster(ctx context.Context, url string, features [][]float64, k, ncomp int) (*ClusterResp, error) {
	reqBody, err := json.Marshal(ClusterReq{Features: features, NClusters: k, NComponents: ncomp})
	if err != nil {
		return nil, fmt.Errorf("cluster marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/cluster", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		const maxErr = 4096
		lb := io.LimitReader(resp.Body, maxErr)
		b, _ := io.ReadAll(lb)
		return nil, fmt.Errorf("cluster %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	var out ClusterResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("cluster decode: %w", err)
	}
	return &out, nil
}
