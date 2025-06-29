# 📦 LegitimNarrate Dataset

The **LegitimNarrate** dataset is a structured collection of crowdfunding project descriptions annotated for **legitimation mechanisms**. It is designed for **sequential sentence classification** and **legitimation analysis** in entrepreneurial narratives. The dataset is split into three files located in the `data/` directory:

## 📂 Dataset Files 
The dataset is organized in the `data/` directory as follows: 
``` data/ 
├── legitimacy_train_.json
├── legitimacy_val_.json 
└── legitimacy_test_.json 
```

---

## 📚 Dataset Structure

Each file is in **JSON Lines** format, where each line represents a single **crowdfunding project** with its corresponding metadata and annotated labels.

### 📌 Example

```json
{
  "project_id": 1,
  "sentences": [
    "Introduction : Octav is high quality speaker to assemble by yourself.",
    "It is entirely customizable and it allows you to create a product which suits you.",
    "It is also an upgradeable and repairable product, allowing you to extend its life cycle and consequently to have a controlled and eco-friendly way of buying.",
    ...
  ],
  "labels": [0, 0, 1, 0, ...],
  "dominant_mechanism": [0, 0, 1, 0, ...],
  "second_mechanism": [1, 1, 2, 3, ...],
  "legitim_score": 1.442695041,
  "status": "failed",
  "technology-category": "sound"
}
```

---

## 🏷 Field Descriptions

| Field                 | Type                | Description |
|----------------------|---------------------|-------------|
| `project_id`          | `int`               | Unique identifier for each crowdfunding project. |
| `sentences`           | `List[str]`         | A list of sentences from the project’s description. |
| `labels`              | `List[int]`         | Sentence-level binary labels: `1` if the sentence contains a legitimation mechanism, `0` otherwise. |
| `dominant_mechanism`  | `List[int]`         | Integer-encoded dominant legitimation mechanism per sentence. Values: `0` (none), `1` (identity), `2` (associative), `3` (organizational). |
| `second_mechanism`    | `List[int or null]` | Second legitimation mechanism per sentence, if present. Otherwise `null`. |
| `legitim_score`       | `float`             | Overall legitimacy score of the project, calculated from the frequency of mechanisms normalized by sentence count. |
| `status`              | `str`               | Outcome of the campaign: `successful` , `failed` or `canceled`. |
| `technology-category` | `str`               | The technological domain or category of the project (e.g., `"sound"`, `"health"`, `"software"`). |

---

## 🧪 Dataset Splits

The dataset is divided into three subsets and stored in the `data/` directory:

| File Name                 | Description                         |
|---------------------------|-------------------------------------|
| `legitimacy_train_.json` | Training set (~80% of the dataset)  |
| `legitimacy_val_.json`   | Validation set (~10%)               |
| `legitimacy_test_.json`  | Test set (~10%)                     |

---

## 🆕 Augmented Data

In addition to the original dataset files, the `data/` directory also includes an **augmentation file** generated using Bing Chat with varied tones (e.g., professional, casual, informative). This file is intended to support research in **data augmentation**, **style variation**, and **robustness evaluation**.

### 📄 File: `augmentation.csv`

| Column     | Description |
|------------|-------------|
| `augmented`| The reformulated sentence generated in a different tone or style. |
| `original` | The original sentence from the dataset. |
| `category` | Label indicating whether the original sentence expresses legitimation mechanism (e.g., `1`) or not (`0`). |

### 🧪 Example Row

`augmented:` The product lifecycle has been extended to support an environmentally responsible approach.
`original:` The product life cycle is extended, allowing you to enter into an eco-responsible approach.
`category:` 1

## 🔍 Applications

The **LegitimNarrate** dataset supports a wide range of research and development applications, including:

- **Entrepreneurial Legitimation Analysis**  
  Study how legitimation mechanisms are used in crowdfunding narratives and how they relate to campaign success.

- **Sequential Sentence Classification**  
  Train models that account for the order and context of sentences to detect legitimation mechanisms.

- **Discourse-level Narrative Understanding**  
  Explore how coherent and persuasive storytelling unfolds across multiple sentences in project descriptions.

- **Imbalanced & Few-shot Learning**  
  Benchmark and develop methods to handle class imbalance, especially in detecting underrepresented mechanisms.

- **AI for Social Impact and Communication Studies**  
  Analyze how entrepreneurs construct legitimacy through language, identity, and associative strategies in digital platforms.

- **Style-based Data Augmentation**  
  Use the `augmentation.csv` file to expand training data with reformulated sentences across diverse tones (e.g., professional, casual, informative), improving generalization and robustness.

- **Tone-aware Sentence Classification**  
  Investigate how different writing styles affect the perception or detection of legitimation strategies in narratives.

- **Paraphrase Detection and Semantic Similarity**  
  Use aligned `original` and `augmented` pairs for developing and testing models that detect meaning preservation across stylistic shifts.

- **Generated Text Detection**  
  Train models to distinguish between human-written and machine-generated sentences, using the `augmentation.csv` file as a labeled dataset for generation detection tasks.
