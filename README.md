# MACAD

Official implementation of **MACAD: Multi-consistency Adaptive Contrastive Domain Adaptation for Time-Series Anomaly Detection**.

MACAD is an unsupervised domain adaptation framework for time-series anomaly detection. It learns domain-invariant and anomaly-discriminative representations by combining three complementary consistency views:

- **Contextual consistency** for global semantic states
- **Temporal consistency** for local temporal continuity
- **Transformational consistency** for structural variations

An adaptive weight network dynamically adjusts the contribution of each consistency objective according to the characteristics of the source-target transfer scenario.

## Overview

The overall framework consists of:

1. Multi-consistency pair construction
2. Shared TCN feature extractor
3. Consistency-aware contrastive learning
4. Adaptive weight network
5. Adversarial domain alignment
6. DeepSVDD-based anomaly detection

For each time-series window, MACAD constructs positive and negative pairs from contextual, temporal, and transformational perspectives. The corresponding contrastive objectives are adaptively weighted and jointly optimized with domain alignment and anomaly detection objectives.

## Datasets

Experiments are conducted on four multivariate time-series benchmark datasets:

- MSL
- SMAP
- SMD
- Boiler

Each source-target pair is treated as an unsupervised domain adaptation scenario, where source labels are available and target labels are not used during training or adaptation.


## License

This repository is provided for academic and research purposes.
