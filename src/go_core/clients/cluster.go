package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
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
	reqBody, _ := json.Marshal(ClusterReq{Features: features, NClusters: k, NComponents: ncomp})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url+"/cluster", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out ClusterResp
	return &out, json.NewDecoder(resp.Body).Decode(&out)
}
