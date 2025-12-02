Project C
# Place Conflation for Overture Maps

This project is a **place conflation / entity resolution** pipeline for the Overture Maps Foundation.  
Given two place records (name, address, website, phone), the goal is to predict whether they refer to the **same real-world place**.

I built:
- Strong **embedding + ML baselines**
- A **feature-rich ensemble model** (30+ features)
- An optional **XGBoost** variant
- A lightweight **Streamlit demo app** for interactive exploration

---

## 1. Dataset & Problem

- **Source:** Sample of ~2.7k place pairs from Overture Maps  
- **Task:** Binary classification – `MATCH` vs `NO_MATCH`  
- **Split:** 3-fold **stratified cross-validation**  
- **Primary metric:** **F1 score** (balance between precision and recall)  
- **Class balance:** ~60% matches / ~40% non-matches  

Files:

- `data/raw/samples_3k_project_c_updated.parquet` – original sample  
- `data/processed/places_cleaned.parquet` – cleaned version used for modeling

Each row contains:
- `name`, `address`, `website`, `phone` – candidate place
- `base_name`, `base_address`, `base_website`, `base_phone` – reference place
- `label` – 1.0 = match, 0.0 = non-match

---

## 2. Baseline Models

I started with **simple similarity features** built from a single embedding model:

**Embedding models tested:**

1. `all-MiniLM-L6-v2` (384-dim, fast)
2. `BAAI/bge-base-en-v1.5` (768-dim, stronger semantics)
3. `intfloat/e5-small-v2` (384-dim, balanced)

**Baseline feature set (4 features):**

- Cosine similarity of **name embeddings**
- Cosine similarity of **name+address embeddings**
- Exact name match (boolean)
- Fuzzy name similarity (`rapidfuzz` ratio)

**Classifier:** GradientBoostingClassifier (sklearn), 3-fold CV.

| Model        | F1   | Accuracy | Precision | Recall | AUC   | Features |
|-------------|------|----------|-----------|--------|-------|----------|
| MiniLM      | 0.827 | 0.793 | 0.831 | 0.824 | 0.875 | 4 |
| E5-small    | 0.857 | 0.827 | 0.852 | 0.864 | 0.905 | 4 |
| BGE-base    | 0.865 | 0.837 | 0.863 | 0.867 | 0.912 | 4 |

**Takeaway:**  
Even with only 4 features, semantic embeddings already give **F1 ≈ 0.86**.  
BGE-base consistently performs best among single-model baselines.

---

## 3. Enhanced Model (30 Features, Gradient Boosting)

To push performance further, I built an **ensemble feature set** combining:
- **3 embedding models** (MiniLM, BGE, E5)
- **Advanced string matching**
- **Contact and interaction features**

### 3.1 Feature engineering

**Embedding-based (name + address):**

- Cosine sims for each model:
  - `MiniLM_Name`, `MiniLM_Combined`
  - `BGE_Name`, `BGE_Combined`
  - `E5_Name`
- Ensemble aggregations:
  - Average and max name similarity
  - Average combined similarity

**String similarity:**

- Exact name match
- `rapidfuzz` scores:
  - `Fuzz_Ratio`, `Fuzz_Partial`, `Fuzz_Token_Sort`, `Fuzz_Token_Set`
- Address fuzzy ratio
- Levenshtein ratio (`difflib.SequenceMatcher`)

**Contact features:**

- Normalized phone comparison → `Same_Phone`
- Domain extracted from URLs → `Same_Domain`
- `Both_Contacts` (phone & domain agree)
- `Any_Contact` (at least one agrees)

**Interaction features:**

- Products of high-signal features  
  (`Name_Combined_Product`, `Ensemble_Product`, `Fuzz_BGE_Product`)
- High-similarity flags (`High_Name_Sim`, `High_Combined_Sim`)
- Interactions with contacts (e.g. `Phone_High_Sim`)

(Confidence-related features are placeholders set to zero in this dataset.)

### 3.2 Results (Gradient Boosting, 3-fold CV)

Best model: **GradientBoostingClassifier** on the 30-feature set.

- **F1:** ~**0.891**
- **Accuracy:** ~0.863  
- **Precision:** ~0.86  
- **Recall:** ~0.92  
- **AUC:** ~0.93  
- **Avg threshold:** ~0.41 (tuned per fold for F1)

Compared to the worst baseline, this is about a **+7–8% relative F1 improvement**.

### 3.3 Feature importance (top drivers)

Top 10 most important features:

1. `Fuzz_Token_Set` (name)  
2. `BGE_Combined` (name+address)  
3. `Name_Combined_Product`  
4. `Addr_Fuzz`  
5. `Fuzz_Partial`  
6. `Ensemble_Combined_Avg`  
7. `MiniLM_Combined`  
8. `BGE_Name`  
9. `Fuzz_BGE_Product`  
10. `Fuzz_Token_Sort`

**Interpretation:**  
The model relies heavily on **string structure** *and* **semantic similarity**, especially for combined name+address text.

---

## 4. Advanced Improvements (XGBoost + Geo features)

I explored a more advanced pipeline to try and reach the internal target of **F1 = 0.93**.

### Extra feature families

1. **Geographic distance** (using cached geocodes):
   - `Geo_Distance_KM`
   - `Geo_Very_Close` (<0.1 km)
   - `Geo_Same_Neighborhood` (<1 km)
   - `Geo_Same_City` (<50 km)
   - `Geo_Different_Region` (>50 km)

2. **Category & brand**  
   - In this sample there are no explicit category/brand columns, so these are safely handled as **dummy zero features** and do not break the pipeline.

3. **Text statistics**
   - Name length diff + ratio
   - Address token count diff
   - Jaccard overlap for names and addresses

4. **Email & cross-field features**
   - Email domain match (dummy zeros here – no email columns)
   - Whether the name appears inside the address
   - Whether both names contain numbers (e.g. “7-Eleven”)

Total: **~47 features** (30 original + 17 new).

### XGBoost results

Model: `xgboost.XGBClassifier` (300 trees, depth 5, subsample 0.9, colsample_bytree 0.8).

3-fold CV:

- **F1:** ~**0.894**  
- **Accuracy:** ~0.870  
- **Precision:** ~0.88  
- **Recall:** ~0.91  
- **AUC:** ~0.94  

This is very close to the Gradient Boosting model; **the dataset may be near its performance ceiling** without richer business features (brand, full categories, better labels).

New features that show up in the top 15:

- `Geo_Same_City`, `Geo_Very_Close`, `Geo_Distance_KM`
- `Name_Len_Diff`

So geographic proximity clearly helps when geocodes are available.

---

## 5. Error Analysis

I inspected misclassified pairs using the best Gradient Boosting model:

- **Total errors:** ~5.4% of pairs  
  - **False positives (~107):** model predicts **MATCH** but label is **NO_MATCH**  
    - Often **chain stores** sharing an address but different business units  
      - e.g. “Walmart Fuel Station” vs “Walmart” at the same address  
  - **False negatives (~40):** true matches that are missed  
    - Typically **language variations, partial names, or noisy international addresses**

This suggests two improvement directions:
1. **Chain-aware logic / brand rules** for large retailers and malls  
2. Better **normalization for multilingual names + addresses**

---

## 6. Streamlit Demo

There is a small Streamlit app to demo the matcher interactively.

### Run locally

```bash
git clone https://github.com/<your-username>/Tisha-Place-Conflation.git
cd Tisha-Place-Conflation

# Create a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Launch the demo
cd demo
streamlit run app.py
