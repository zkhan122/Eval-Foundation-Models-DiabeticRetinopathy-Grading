"Evaluating Generalisation and Transferability in Foundation Models for Ocular Diseases" By Zayaan Khan


Foundation models also known as large X models (LxM) are deep learning models which have been trained on large datasets to be applied across various downstream tasks
Popular examples include ChatGPT, BERT, CLIP, Stable Diffusion.

Aim: To evaluate the performance of retinal-specific foundation models such as RETFound (fundus specific), UrFound (retina specific) & CLIP (general) for the downstream tasks of diabetic retinopathy severity grading, glaucoma detection, multi-disease classification.

Model evaluation metrics:
- per-class recall
- macro-AUC
- weighted-AUC
- F1 (macro included)
- precision
- Quadratic Weighted Kappa
- per-class AUC
- sensitivy
- specificity
- attention maps with entropy (uncertainty) measure

Statistical evaluation tests:
- Jensen-Shannon divergence
- bootstrapped confidence intervals
- McNemar's test

Computational evaluation metrics:
- latency
- throughput
- per epoch time



