package clients

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

// --- Clustering (/cluster) ---
type ClusterReq struct {
    Features     [][]float64 `json:"features"`
    NClusters    int         `json:"n_clusters"`
    NComponents  int         `json:"n_components"`
    DimRedMethod string      `json:"dim_red_method"`
}

type ClusterResp struct {
    ClusterLabels              []int       `json:"cluster_labels"`
    MembershipMatrix           [][]float64 `json:"membership_matrix"`
    ReducedFeatures            [][]float64 `json:"reduced_features"`
    TotalExplainedVariance     float64     `json:"explained_variance"`
    DimensionExplainedVariance []float64   `json:"dimension_explained_variance"`
    DimensionComponents        [][]float64 `json:"dimension_components"`
    ReductionUsed              string      `json:"reduction_used"`
}

func (h *HTTP) Cluster(ctx context.Context, url string, features [][]float64, k, ncomp int, method string) (*ClusterResp, error) {
    reqBody, _ := json.Marshal(ClusterReq{Features: features, NClusters: k, NComponents: ncomp, DimRedMethod: method})
    req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/cluster", bytes.NewReader(reqBody))
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", "application/json")

    resp, err := h.c.Do(req)
    if err != nil {
        return nil, err
    }
    defer func(Body io.ReadCloser) {
        err := Body.Close()
        if err != nil {

        }
    }(resp.Body)

    if resp.StatusCode != http.StatusOK {
        b, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("cluster %s: %s", resp.Status, string(b))
    }

    var out ClusterResp
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
        return nil, fmt.Errorf("cluster decode: %w", err)
    }
    return &out, nil
}
