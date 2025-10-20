# Quick Start Guide

## Processing an Experiment

1. Place your data in the appropriate folders:
```
data/raw/audio/experiment_001.wav
data/raw/robot_logs/experiment_001.json
```

2. Run the pipeline:
```bash
./bin/edmo-pipeline process --experiment experiment_001
```

3. View results:
```
data/outputs/feedback/experiment_001_report.pdf
data/outputs/visualizations/experiment_001_timeline.png
```

## Configuration

Edit `config/dev/config.yaml` to adjust:
- Time window sizes
- Clustering parameters
- Feature extraction settings
- Service endpoints
