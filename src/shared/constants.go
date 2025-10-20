package shared

const (
    // Audio processing
    AudioSampleRate = 16000
    AudioChannels   = 1
    AudioCodec      = "pcm_s16le"
    
    // Video processing
    VideoFrameRate = 1
    
    // Service URLs (from config)
    NLPServiceURL         = "http://localhost:8001"
    ASRServiceURL         = "http://localhost:8002"
    DiarizationServiceURL = "http://localhost:8003"
    EmotionServiceURL     = "http://localhost:8004"
    ClusteringServiceURL  = "http://localhost:8005"
    VisualizationServiceURL = "http://localhost:8006"
    
    // Feature extraction
    TimeWindowSeconds = 30
    OverlapSeconds    = 15
    
    // Version
    Version = "0.1.0"
)
