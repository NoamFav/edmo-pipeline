# EDMO Pipeline - Project Summary

## Mission Statement

Support teachers in developing students' communication and collaboration skills by providing data-driven, multimodal insights into group dynamics during educational robotics tasks.

## Scientific Objectives

1. **Pattern Discovery**: Extract and identify latent communication patterns from multimodal dialogue features
2. **Success Analysis**: Analyze relationships between communication strategies and task performance
3. **Feedback System**: Develop an explainable, teacher-friendly feedback system grounded in psychological frameworks

## Technical Approach

### Data Sources
- Audio recordings of student conversations
- Video recordings of student interactions
- Robot movement logs and action data
- Teacher observations and assessments

### Processing Pipeline

**Stage 1: Preprocessing**
- Audio normalization → 16kHz mono WAV
- Silence removal
- Robot log standardization
- Timeline synchronization

**Stage 2: Feature Extraction**
- Transcription with timestamps (Whisper)
- Speaker identification (Pyannote)
- Emotion classification (GoEmotions)
- Sentence embeddings (E5/BERT)
- Prosodic features (pitch, energy, rate)
- Robot metrics (distance, actions, coordination)

**Stage 3: Aggregation**
- Time-windowed feature aggregation (30s windows, 15s overlap)
- Per-student and per-group features
- Synchronized audio-robot data points

**Stage 4: Analysis**
- Dimensionality reduction (PCA: 70% variance, t-SNE for visualization)
- Fuzzy C-means clustering (5 clusters based on PISA CPS)
- Pattern interpretation and labeling

**Stage 5: Output**
- Timeline visualizations
- Radar charts (emotion, cooperation)
- Correlation analysis (communication ↔ robot performance)
- Textual feedback reports

## Validation Framework

All discovered patterns are validated against the **PISA 2015 Collaborative Problem Solving (CPS) Framework**:

1. **Establishing and maintaining shared understanding**
2. **Taking appropriate action to solve the problem**
3. **Establishing and maintaining team organization**

## Key Performance Indicators

| Objective | KPI | Target |
|-----------|-----|--------|
| O1: Pattern Discovery | Detection accuracy of PISA CPS skills | >75% |
| O2: Success Analysis | Clear dependency visualizations | Qualitative assessment |
| O3: Feedback System | Alignment with teacher assessments | >75% |

## Milestones

- **MS1** (Nov 3): Data pipeline established
- **MS2** (Nov 24): Feature extraction completed
- **MS3** (Dec 8): Dimensionality reduction ready
- **MS4** (Dec 15): Clustering finalized
- **MS5** (Jan 10): Visualizations completed
- **MS6** (Jan 18): Final deliverables submitted

## Deliverables

1. **Midway Evaluation** (Dec 5): Presentation + demonstration
2. **Final Report** (Jan 18): Research paper format
3. **Processed Dataset** (Jan 18): Audio/robot features + scripts
4. **Insight Software** (Jan 18): Analysis tools + generated insights
5. **Final Presentation** (Jan 19): Project showcase

## Impact

### Educational
- Enhanced soft skills development
- Evidence-based teaching interventions
- Improved collaborative learning outcomes

### Technical
- Reusable multimodal learning analytics framework
- Open-source tools for educational research
- Novel methodology for communication analysis

### Societal
- Reduced teacher workload
- Data-driven educational feedback
- Privacy-preserving student analytics

## Risk Management

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Data quality issues | Medium | High | Pre-screening, normalization, synthetic data |
| Limited teacher access | High | Medium | Efficient meeting preparation |
| Integration complexity | Medium | High | Shared timebase, visual checks |
| Model limitations | Medium | Medium | Fine-tuning, pretrained models, validation |
| Team coordination | Medium | Medium | Clear task leads, regular meetings |

## Related Work

Our approach builds on and extends:
- **Multimodal analytics**: Ma et al. (2022, 2024), Spikol et al. (2017)
- **Communication analysis**: Tsan et al. (2021), D'Angelo & Rajarathnam (2024)
- **Unsupervised methods**: Ezen-Can & Boyer (2015), Li et al. (2016)
- **Educational robotics**: Malinverni et al. (2021), Möckel et al. (2020)

## Future Directions

1. Real-time feedback during tasks
2. Adaptive intervention recommendations
3. Longitudinal skill development tracking
4. Cross-cultural communication pattern analysis
5. Integration with other educational ecosystems
