# Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels
This is the training code for our paper "Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels" (NeurIPS-23).

## Abstract
Deep Neural Networks (DNNs) have been criticized because they easily overfit noisy (incorrect) labels. 
To improve the robustness of DNNs, existing methods for image data regard samples with small training losses as correctly labeled data (small-loss criterion).
Nevertheless, time series' discriminative patterns are easily distorted by external noises (i.e., frequency perturbations) during the recording process. This results in training losses of some time series samples that do not meet the small-loss criterion.
Therefore, this paper proposes a deep learning paradigm called Scale-teaching for combating time series noisy labels.
Specifically, we design a fine-to-coarse cross-scale fusion mechanism for learning discriminative patterns by utilizing time series at different scales to train multiple DNNs simultaneously.
Meanwhile, each network is trained in a cross-teaching manner by using complementary information from different scales to select small-loss samples as clean labels.
For unselected large-loss samples, we introduce multi-scale embedding graph learning via label propagation to correct their labels by using selected clean samples.
Experiments on multiple benchmark time series datasets demonstrate the superiority of the proposed Scale-teaching paradigm over state-of-the-art methods in terms of effectiveness and robustness.

## Datasets
### Four individual large time series datasets
* [HAR dataset](https://github.com/emadeldeen24/TS-TCC)
* [UniMiB-SHAR dataset](https://github.com/imics-lab/TSAR)
* [FD-A dataset](https://github.com/emadeldeen24/TS-TCC)
* [Sleep-EDF dataset](https://github.com/emadeldeen24/TS-TCC)
### UCR 128 archive time series datasets
* [UCR 128 archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip)
### UEA 30 archive time series datasets
* [UEA 30 archive](http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip)

