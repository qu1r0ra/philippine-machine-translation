# philippine-machine-translation <!-- omit from toc -->

<!-- ![title](./readme/title.jpg) -->

<!-- Refer to https://shields.io/badges for usage -->

![Year, Term, Course](https://img.shields.io/badge/AY2526--T1-NLP1000-blue) ![JupyterLab](https://img.shields.io/badge/JupyterLab-orange) ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)

An exploration of neural machine translation for Cebuano-Spanish translation. Created for NLP1000 (Introduction to Natural Language Processing).

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Reproducing Results](#2-reproducing-results)
- [3. References](#3-references)

## 1. Introduction

Recently, machine translation (MT) has experienced significant growth in popularity due to its speed and capability to process and deliver large volumes of translated data within a short time frame. This advancement is largely attributed to continuous improvements in machine learning algorithms and hardware. Within the field of natural language processing (NLP), machine translation serves as a major subfield focused on the development and enhancement of computer-based translation systems that automatically convert textual content from one language to another ([Machine Translation, 2025](https://www.sciencedirect.com/topics/computer-science/machine-translation)). Given this context, models must be accurate and generate contextually appropriate translations, preserving the tone, style, and syntactic flexibility of the original text. Therefore, this highlights the need for designing models that can adapt to the nuances of different languages and continuously improve through training and optimization.

Several strategies have been explored to improve the performance of MT systems, including active learning, data augmentation, embedding alignment, and multilingual modeling ([Tafa et al., 2025](https://doi.org/10.1109/ACCESS.2025.3562918)). In this study, a combination of these strategies was applied, with a particular focus on data augmentation and the implementation of an attention-based Gated Recurrent Unit (GRU) Sequence-to-Sequence (Seq2Seq) architecture. Additional optimization techniques such as dropout regularization, gradient clipping, and early stopping were also integrated to enhance model generalization and training stability. The model developed in this work is a Neural Machine Translation (NMT) system designed to map a source sentence to a corresponding target sentence through end-to-end learning. Specifically, the study focuses on two low-resource Philippine languages: Cebuano and Chavacano, and the translation of the former into Spanish.

Following the development and training of the model, a series of experiments was conducted to evaluate translation quality. The system's performance was assessed using standard MT evaluation metrics, namely BLEU, CHRF, and TER, to quantify accuracy, fluency, and error rate, respectively. Overall, this study aims to contribute to the ongoing efforts in building robust neural translation systems for low-resource Philippine languages by demonstrating the effectiveness of attention-based architectures and data-driven augmentation strategies in improving translation quality.

## 2. Reproducing Results

## 3. References

[1] _Machine Translation_ (2025). <https://www.sciencedirect.com/topics/computer-science/machine-translation>.
[2] Tafa, Taofik O. et al. (2025). “Machine Translation Performance for Low-Resource Languages: A Systematic Literature Review”. In: _IEEE Access_ 13, pp. 72486–72505. DOI: [10.1109/ACCESS.2025.3562918](https://doi.org/10.1109/ACCESS.2025.3562918).
