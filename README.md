# InteDisUX: Interpretation-Guided Discriminative User-Centric Explanation for Time Series
This readme file is an outcome of the [CENG7880 (Fall 2026)](https://metu-trai.github.io/) project for reproducing a paper which does not have an implementation. See [CENG7880 (Fall 2026) Project List](https://github.com/metu-trai/Projects2026) for a complete list of all paper reproduction projects.

# 1. Introduction

"InteDisUX: Intepretation-Guided Discriminative User-Centric Explanation for Time Series" [1] was written by Viet-Hung Tran, Zichi Zhang, Tuan Dung Pham, Ngoc Phu Doan, Anh-Tuan Hoang, Peixin Li, Hans Vandierendonck, Ira Assent, and Son T. Mai and was published at AAAI 2025. The study focuses on explaining the decisions of deep learning models for time series classification (TSC) in a way that is both faithful to the model and understandable to humans. Most existing approaches use attribution maps, which assign a separate importance score to every time step, so the resulting explanations are long and hard for users to interpret. LIME-based methods [2, 4, 5] improve on this by splitting the series into segments, but they require a predetermined number of segments for every instance and inherit the shortcomings of LIME, such as instability, poor local fidelity, and sensitivity to imbalanced sampling. To address these issues, the authors propose a **segment-level** explanation method that does not rely on predefined segments. They first introduce Segment-level Integrated Gradients (SIG), which scores each segment by summing the Integrated Gradients [3] of its time steps and satisfies several axioms of model explanation at the segment level. Starting from small equal-length segments, their method then merges two consecutive segments at a time under a greedy strategy guided by two new metrics: discrimination gain, which favors segments that differ clearly from their neighbors, and faithfulness gain, which uses the model's own outputs to keep the explanation faithful. In essence, the number and location of segments are adapted to each instance and to the model being explained. As a result, they report more faithful explanations than LIME-based methods on all 12 datasets and more robust explanations on 8 of the 12 datasets.

The goal of this project is to implement the proposed InteDisUX method and to reproduce the results presented in the paper to evaluate its explanation quality. These results include the faithfulness and robustness scores on 12 UCR datasets averaged over five deep learning architectures (CNN, Transformer, FC, Dense-FCN, and LSTM-FCN), compared to the baseline LIME-based methods (Neves et al. [4], Guillemé et al. [5], and LIMESegment [2]). They also include the ablation studies that measure the separate contributions of SIG, the discrimination gain, and the faithfulness gain, as well as the effect of the initial number of segments on the explanation.

## 1.1. Paper summary

### 1.1.1. Motivation and Aim

Time series classification (TSC) is used in many practical domains such as healthcare, finance, and supply chain management. Deep learning (DL) models have become an effective approach for TSC because they learn relevant feature representations directly from the data. However, DL models are black boxes, which reduces their trustworthiness and makes it difficult to identify and correct their errors. Explainable AI (XAI) methods address this problem by explaining which parts of the input the model relies on.

Most existing XAI methods for time series produce attribution maps, which assign an importance score to every time step. A time series usually contains a large number of time points, so these maps are impractical for users to read. They also ignore the temporal correlations between neighbouring time steps, which is a key difference between time series and tabular data. To obtain explanations that are simpler and closer to human perception, LIME-based methods [2, 4, 5] split the time series into segments and assign an importance score to each segment with a LIME [6] surrogate model. However, these methods have two main problems. First, they require a predetermined number of segments, which is the same for every instance. For example, LIMESegment [2] always produces 3 segments in the authors' comparison. Second, they inherit the weaknesses of LIME:
- **Instability:** sample weights become very small for samples that are far from the explained instance, so the importance scores approach zero.
- **Poor local fidelity:** the generated samples are far from the explained instance, so they are not local.
- **Imbalanced sampling:** all generated samples may fall into a single class, which makes every segment's score zero after ridge regression.

To address these challenges, the authors propose InteDisUX, a segment-level explanation method that does not depend on predefined segments. The importance of each segment is computed with Segment-level Integrated Gradients (SIG), which is deterministic and does not require sampling, so it avoids the problems of LIME. Segments are found with a bottom-up strategy: the series is first divided into small equal segments, and then two consecutive segments are merged greedily. Each merge is guided by two new metrics:
- **Discrimination gain** measures how distinct consecutive segments become after the merge.
- **Faithfulness gain** uses the model's outputs to make sure that the merge does not make the explanation less faithful.

In this way, both the number and the locations of the segments are adapted to each instance and to the model being explained.

### 1.1.2. Datasets

All 12 datasets are univariate binary classification datasets from the UCR Time Series Archive [7]. They are the same datasets that were used in the LIMESegment [2] paper.

**[Coffee](https://www.timeseriesclassification.com/description.php?Dataset=Coffee)** [7]: Contains 56 food spectrograph samples (28 train, 28 test) of length 286. The task is to distinguish Arabica from Robusta coffee beans.

**[Strawberry](https://www.timeseriesclassification.com/description.php?Dataset=Strawberry)** [7]: Contains 983 infrared spectra (613 train, 370 test) of length 235 of fruit purees. The task is to classify purees as strawberry or non-strawberry (adulterated).

**[GunPointOldVersusYoung (GPAge)](https://www.timeseriesclassification.com/description.php?Dataset=GunPointOldVersusYoung)** [7]: Contains 451 samples (136 train, 315 test) of length 150. Each series is the x-coordinate of an actor's hand while drawing a replica gun from a holster. The task is to classify the actor as young or old.

**[GunPointMaleVersusFemale (GPGender)](https://www.timeseriesclassification.com/description.php?Dataset=GunPointMaleVersusFemale)** [7]: Contains 451 samples (135 train, 316 test) of length 150. It uses the same gun-drawing recordings as GPAge, and the task is to classify the actor's gender.

**[HandOutlines (Hand)](https://www.timeseriesclassification.com/description.php?Dataset=HandOutlines)** [7]: Contains 1,370 samples (1,000 train, 370 test) of length 2,709. The outline of a hand, extracted from an X-ray image, is converted into a one-dimensional series.

**[Yoga](https://www.timeseriesclassification.com/description.php?Dataset=Yoga)** [7]: Contains 3,300 samples (300 train, 3,000 test) of length 426. The outline of an actor performing yoga poses is converted into a one-dimensional series, and the task is to classify the actor (male or female).

**[ECG200 (ECG)](https://www.timeseriesclassification.com/description.php?Dataset=ECG200)** [7]: Contains 200 samples (100 train, 100 test) of length 96. Each series records the electrical activity of a single heartbeat, and the task is to distinguish normal heartbeats from myocardial infarction.

**[DodgerLoopGame (Dodger)](https://www.timeseriesclassification.com/description.php?Dataset=DodgerLoopGame)** [7]: Contains 158 samples (20 train, 138 test) of length 288. Each series is one day of traffic counts from a loop sensor near Dodger Stadium (Los Angeles), aggregated every 5 minutes. The task is to classify each day as a game day or a normal day.

**[Chinatown](https://www.timeseriesclassification.com/description.php?Dataset=Chinatown)** [7]: Contains 363 samples (20 train, 343 test) of length 24. Each series is one day of hourly pedestrian counts from an automated counting system in Melbourne's Chinatown. The task is to classify each day as a weekday or a weekend day.

**[FreezerSmallTrain (Freezer)](https://www.timeseriesclassification.com/description.php?Dataset=FreezerSmallTrain)** [7]: Contains 2,878 samples (28 train, 2,850 test) of length 301. Each series is the power demand of a freezer, and the task is to classify the freezer's location (kitchen or garage).

**[HouseTwenty (House20)](https://www.timeseriesclassification.com/description.php?Dataset=HouseTwenty)** [7]: Contains 159 samples (40 train, 119 test) of length 2,000. Each series is an electricity consumption profile of a single house, and there are two classes of load profiles.

**[WormsTwoClass (Worms)](https://www.timeseriesclassification.com/description.php?Dataset=WormsTwoClass)** [7]: Contains 258 samples (181 train, 77 test) of length 900. Each series describes the motion of a *C. elegans* worm, and the task is to distinguish wild-type worms from mutants.

### 1.1.3. Findings and Conclusion of the Paper

The authors propose InteDisUX, a segment-level explanation method for time series classifiers. It does not rely on a predetermined number of segments. Instead, it builds the segments bottom-up by merging consecutive segments under the guidance of the discrimination and faithfulness gains.

They introduce SIG for scoring segments, and they show theoretically that it satisfies important axioms of model explanation at the segment level, such as completeness, sensitivity, and implementation invariance. Because SIG is additive, the score of a merged segment can be computed by adding the scores of its parts, without computing gradients again. As a result, SIG avoids the shortcomings of LIME and also makes the refinement process stable.

The authors evaluate the method on 12 UCR datasets with 5 DL architectures (CNN, Transformer, FC, Dense-FCN, and LSTM-FCN). InteDisUX achieves higher faithfulness than all three LIME-based baselines on all 12 datasets and higher robustness than LIMESegment on 8 of the 12 datasets. Their ablation studies show that:
- SIG mainly improves faithfulness;
- the discrimination gain mainly improves robustness;
- the faithfulness gain further improves faithfulness without reducing robustness.

# 2. The method and our interpretation

Feel free to add more subsections and change subtitles as necessary.

## 2.1. The original method

Explain the original method.

## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

Explain your code & directory structure and how other people can run it.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

[1] V.-H. Tran, Z. Zhang, T. D. Pham, N. P. Doan, A.-T. Hoang, P. Li, H. Vandierendonck, I. Assent, and S. T. Mai, "InteDisUX: Intepretation-Guided Discriminative User-Centric Explanation for Time Series," in *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, vol. 39, pp. 20921–20928, 2025.

[2] T. Sivill and P. Flach, "LIMESegment: Meaningful, Realistic Time Series Explanations," in *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 151, pp. 3418–3433, 2022. Code: https://github.com/TortySivill/LIMESegment

[3] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic Attribution for Deep Networks," in *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3319–3328, 2017.

[4] I. Neves, D. Folgado, S. Santos, M. Barandas, A. Campagner, L. Ronzio, F. Cabitza, and H. Gamboa, "Interpretable Heartbeat Classification Using Local Model-Agnostic Explanations on ECGs," *Computers in Biology and Medicine*, vol. 133, p. 104393, 2021.

[5] M. Guillemé, V. Masson, L. Rozé, and A. Termier, "Agnostic Local Explanation for Time Series Classification," in *IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)*, pp. 432–439, 2019.

[6] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 1135–1144, 2016.

[7] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, "The UCR Time Series Archive," *IEEE/CAA Journal of Automatica Sinica*, vol. 6, no. 6, pp. 1293–1305, 2019. https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

# Contact

- Ferayenur Bozkurt - e279098@metu.edu.tr
- Efe Kaan Fidancı -
