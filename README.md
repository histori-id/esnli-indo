# e-SNLI-ID: An Extension of Indonesian Natural Language Inference Dataset with Textual Explanations
## Overview
e-SNLI-ID is a natural language explanation (NLE) resource that extends the existing Indonesian Natural Language Inference (SNLI) dataset [1]. While the baseline Indonesian SNLI dataset provides only text pairs (Premise and Hypothesis) and their corresponding inference labels, this dataset introduces the first collection of translated textual explanations (rationales) in Indonesian for every instance.

By integrating these textual justifications, this dataset effectively elevates the standard Indonesian NLI task into an Explainable AI (XAI) benchmark, enabling researchers to train and evaluate models that not only predict semantic relations (Entailment, Contradiction, Neutral) but also generate human-like reasoning in Indonesian.

## Translation Quality Evaluation
To rigorously evaluate the semantic and structural fidelity of the Indonesian explanations, we employed a back-translation validation pipeline. The localized Indonesian rationales were translated back into English and compared directly against the original English e-SNLI ground truth rationales.

The corpus-wide evaluation scores demonstrating the high fidelity of the text are as follows:

BLEU Score: 34.88 — Reflects the lexical overlap and structural retention between the back-translated text and the original English source.

BERTScore (F1): 89.42 — Measures token-level contextual embedding similarity, demonstrating that the translated rationales successfully preserve critical semantic nuances.

BERT Cosine Similarity: [...]  — Confirms high vector alignment in the shared embedding space, ensuring that no significant logical or conceptual drift occurred during the forward and backward translation processes.

## Methodology & Extension Process
Base Dataset: We utilized the existing translated version of the Indonesian SNLI dataset, which comprises premises, hypotheses, and golden labels.

Explanation Translation: The corresponding textual explanations from the original English e-SNLI dataset were extracted and meticulously translated into Indonesian using advanced machine translation frameworks optimized for contextual and logical consistency.

Alignment & Curation: The translated explanations were systematically paired and aligned baris-demi-baris (line-by-line) with the existing Indonesian SNLI dataset using robust alignment scripts. Structural validation was performed to ensure that the semantic integrity of the explanation still perfectly justifies the localized premise-hypothesis pairs.

## Value of the Data
- Completing the Loop for Indonesian XAI: It transforms a standard classification task (SNLI) into a generative reasoning task (e-SNLI) for the Indonesian NLP community.

- Minimizing Research Redundancy: By building directly upon the previously translated Indonesian SNLI corpus, this dataset maintains consistency for researchers who want to benchmark the impact of adding rationales to their existing NLI models.

- Strategic for Multi-task Learning: Highly valuable for developing pipeline or joint end-to-end architectures (e.g., Predict-Explain-Predict frameworks) in low-to-medium resource language settings.

## Important Notice: Dataset Availability

> ⚠️ **Note on Train Dataset Split:** Please note that due to storage and hosting constraints on this platform, the training subset provided directly within this repository **does not contain the full training dataset**. 
> 
> To download the **complete and uncompressed dataset** (including the full training, validation, and test splits), please access our Google Drive repository here:  
> 🔗 **[Access Full Dataset via Google Drive](https://drive.google.com/drive/folders/1y6th36VS25f2RYpkQFhww6lA18mgzBKq?usp=sharing)**

## Data Format and Structure
The dataset is formatted in JSON Lines (.jsonl), combining the preexisting Indonesian SNLI features with our newly added penjelasan key.

Each JSON object contains the following keys:

``annotator_labels``: The inference label array assigned by annotators.

``label emas``: The golden label for the NLI task (entailment, kontradiksi, or netral).

``kalimat1``: The Premise text in Indonesian (from the pre-existing SNLI-ID).

``kalimat2``: The Hypothesis text in Indonesian (from the pre-existing SNLI-ID).

``penjelasan`` (Our Contribution): The newly added Explanation/Rationale text in Indonesian that logically justifies the golden label.

## Keywords
e-SNLI; Natural Language Inference (NLI); Textual Rationales; Indonesian NLP; Explainable AI (XAI); Dataset Extension.

## References
[1] Putra, I. Made Suwija, Daniel Siahaan, and Ahmad Saikhu. "SNLI Indo: A recognizing textual entailment dataset in Indonesian derived from the Stanford Natural Language Inference dataset." Data in Brief 52 (2024): 109998.