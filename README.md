# Self-Assessed Affect Sub-challenge, INTERSPEECH 2018
+ The Self-Assessed Affect Sub-challenge is one of four challenges in The INTERSPEECH 2018 Computational Paralinguistics Challenge. In this sub-callenge, we try to classified four basic emotions annotated in the speech.
+ We beat the baseline provided by the challenge organizer in [here](https://pdfs.semanticscholar.org/783d/2bd2820b35ce7e398be412569e9d5c6f5880.pdf), 
and won the second place in this challenge.
+ Tensorflow implementation of bidirectional LSTM with attention in [Self-Assessed Affect Recognition using Fusion of Attentional BLSTM and Static Acoustic Features](https://pdfs.semanticscholar.org/5ceb/28f9769b5af8086067b22d71dbb743cc7c13.pdf)

## Reference
S. Mirsamadi, E. Barsoum, and C. Zhang, “Automatic speech emotion recognition using recurrent neural networks with local attention,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, U.S.A., Mar. 2017, IEEE, pp. 2227–2231.

## Data:
Data disciptions of this sub-challenge refer to [The INTERSPEECH 2018 Computational Paralinguistics Challenge](https://pdfs.semanticscholar.org/783d/2bd2820b35ce7e398be412569e9d5c6f5880.pdf).

## Requirements
Some required libraries:
```
python                   >=3.6
numpy                    1.14.5
joblib                   0.13.0
pandas                   0.22.0
scikit-learn             0.19.1
tensorflow               1.4.0
```

## Code:
+ data.py: batch generator
+ model_brnn.py: main codes, bi-directional LSTM with self-attention framework.
