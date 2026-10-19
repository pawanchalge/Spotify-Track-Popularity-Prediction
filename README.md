# 🎵 Spotify Track Popularity Prediction: Classical Machine Learning Project

> An end-to-end, pedagogical machine learning project predicting a Spotify track's popularity score (0–100) from acoustic audio waveform features and musical metadata. Built with **scikit-learn**, **XGBoost**, and an interactive **Streamlit** dashboard.

---

## 📌 Project Overview
Spotify catalogs over 100 million tracks. What acoustic properties differentiate viral hits from obscure catalog uploads?
This project formulates song popularity prediction as a **supervised regression task** using classical machine learning algorithms:

- **Target Variable ($y$):** `popularity` (continuous integer score from 0 to 100, reflecting recent stream velocity and overall play counts).
- **Features ($X$):** 15 features encompassing acoustic metrics (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`), structural metadata (`key`, `mode`, `time_signature`, `explicit`), and `track_genre` (114 unique categories).
- **Leakage Prevention:** Deduplication on `['track_name', 'artists']` followed by `GroupShuffleSplit` on `track_id` ensures zero data leakage between training and testing splits.
- **Pipeline Architecture:** All one-hot encoding and feature scaling transformations are enclosed inside scikit-learn `Pipeline` and `ColumnTransformer` blocks.

---

## 📂 Project Structure

```text
d:/ML_project/
├── dataset.csv                          # Spotify Tracks Dataset (114,000 rows × 21 columns)
├── spotify_popularity_prediction.ipynb   # Complete, executed end-to-end Jupyter Notebook
├── spotify_popularity_model.joblib      # Serialized production pipeline (ColumnTransformer + Tuned XGBoost)
├── app.py                               # Interactive Streamlit Web Application (Port 8501)
├── index.html                           # Spotify SoundLab Web Interface (Semantic HTML5 - Port 3000)
├── style.css                            # SoundLab Design System & Glassmorphism (Separated CSS)
├── app.js                               # Real-Time Telemetry & Radar Engine (Separated JavaScript)
├── build_notebook.py                    # Programmatic notebook generator script
├── requirements.txt                     # Python library dependencies
└── README.md                            # Comprehensive project documentation

```

---

## 📊 Experimental Results & Model Comparison

All models were evaluated using **5-Fold GroupKFold Cross-Validation** on the training set (~65,000 tracks), followed by a single final evaluation of the best model on the held-out test set (~16,000 tracks).

| Model Pipeline | Preprocessing | 5-Fold CV RMSE | 5-Fold CV MAE | 5-Fold CV $R^2$ | Test Set RMSE | Test Set MAE | Test Set $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Dummy Baseline** | Passthrough | $19.379 \pm 0.128$ | $16.147 \pm 0.088$ | $-0.000 \pm 0.000$ | — | — | — |
| **2. Ridge Regression** | StandardScaler + OneHot | $15.241 \pm 0.085$ | $11.082 \pm 0.045$ | $0.383 \pm 0.004$ | — | — | — |
| **3. Random Forest (50 trees)** | Passthrough + OneHot | $16.529 \pm 0.080$ | $12.115 \pm 0.046$ | $0.274 \pm 0.004$ | — | — | — |
| **4. Default XGBoost** | Passthrough + OneHot | $15.496 \pm 0.081$ | $11.196 \pm 0.045$ | $0.362 \pm 0.003$ | — | — | — |
| **5. Tuned XGBoost (Winner)** | Passthrough + OneHot | **$15.084 \pm 0.078$** | **$10.612 \pm 0.042$** | **$0.406 \pm 0.004$** | **$15.124$** | **$10.630$** | **$0.403$ (40.3%)** |

### 💡 Plain-English Metric Interpretation:
1. **$R^2 = 0.403$ (40.3%):** Acoustic audio properties and musical genre explain **~40% of the variance** in song popularity. The remaining 60% is driven by external contextual factors (artist star power, marketing, playlist placement).
2. **$\text{MAE} = 10.63$ points:** On a 0–100 scale, the model's prediction is, on average, within ~10.6 points of the song's actual popularity.
3. **$\text{RMSE} = 15.12$ points:** Slightly higher than MAE because it penalizes extreme outliers (e.g. unpredicted viral hits).

---

## 🔍 Key Findings & Acoustic Insights

1. **Genre Dominates Acoustic Nuance:** Permutation importance shows that `track_genre` accounts for the vast majority of predictive power. Mainstream genres (`pop-film`, `k-pop`, `chill`, `grunge`) have average popularities above 50, whereas niche genres (`chicago-house`, `detroit-techno`) average below 20.
2. **Vocal Presence is Essential:** `instrumentalness` is the strongest negative audio feature. Songs with high instrumentalness (>0.7) are rarely popular on mainstream Spotify charts.
3. **The Commercial Loudness Sweet Spot:** Popular tracks cluster tightly in the **-8 dB to -4 dB** range. Extremely quiet tracks (< -20 dB) almost never register high streaming counts.
4. **Danceability & Tempo:** Mainstream hits concentrate in the **0.6 to 0.8 danceability** range with tempos between **100 and 130 BPM**.
5. **Pitch Class Irrelevance:** Features like `key` and `mode` (major vs. minor) had near-zero permutation importance, showing that listeners and algorithms do not discriminate based on musical key signature.

---

## ⚠️ Error Analysis & Model Limitations (The "Hit Song" Paradox)

Why does classical machine learning cap out around $R^2 \approx 0.40$ on acoustic audio features?

- **Failure Mode 1: Underpredicting Superstar Hits:** Tracks by Bad Bunny, Taylor Swift, or Drake achieve 80–100 popularity regardless of whether their audio features resemble an indie demo. **Audio features cannot capture star power.**
- **Failure Mode 2: Overpredicting High-Energy Obscure Tracks:** An unreleased track produced with pop-grade danceability, energy, and loudness will still have 0 streams if it lacks marketing and playlist curation.
- **Failure Mode 3: Zero-Inflation:** ~16% of the catalog has `popularity == 0` (unindexed or dormant tracks).
- **Regression to the Mean:** Because extreme values (0 and 100) are rare, the model tends to pull predictions toward the global mean (30–50).

---

## 🚀 How to Run the Project

### 1. Prerequisites & Environment Setup
Clone or navigate to the project directory:
```bash
cd d:\ML_project
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

### 2. Running the Jupyter Notebook
You can open and step through the notebook in JupyterLab or VS Code:
```bash
python -m jupyter notebook spotify_popularity_prediction.ipynb
```
Or run the entire notebook from top to bottom non-interactively:
```bash
python -m nbconvert --to notebook --execute --inplace spotify_popularity_prediction.ipynb
```

#### What You Will See in the Notebook Output:
- **Section 1–4:** Automated CSV detection, column dtypes, missing value breakdown, duplicate song deduplication, and zero-popularity analysis.
- **Section 5:** 5 publication-ready visualizations:
  1. Popularity distribution with median and mean markers.
  2. Histograms of all continuous audio features.
  3. Pearson correlation heatmap and ranked correlation table.
  4. Top 15 genres ranked by mean popularity.
  5. Hexbin density plots showing popularity vs. `instrumentalness`, `loudness`, and `danceability`.
- **Section 6–8:** Leakage-free `ColumnTransformer` pipelines and 5-fold cross-validation comparison table with error bars.
- **Section 9–10:** Randomized hyperparameter tuning results, held-out test evaluation, Predicted vs. Actual scatter plot, and Residual plot.
- **Section 11–12:** Side-by-side built-in vs. permutation feature importance, top 10 over/under-predicted song tables, and MAE across popularity tiers.
- **Section 13–14:** Serialization of `spotify_popularity_model.joblib` and real-time custom song prediction.

---

### 3. Launching the Interactive Streamlit Web App
To run the interactive web dashboard:
```bash
python -m streamlit run app.py --server.port 8501
```

The application opens at **`http://localhost:8501`** and provides 6 primary tabs:
- **🔍 114k Catalogue & Player:** Search across all 114,000 Spotify tracks by title, artist, or genre. Displays album artwork, official Spotify embed player, audio preview streaming, and 1-click feature ingestion into the predictor.
- **🎯 Popularity Predictor & Uncertainty:** Instantaneous XGBoost pipeline inference with 95% Confidence Interval bounds (`±29.64` pts), MAE error bounds (`±10.63` pts), and 15/15 Feature Completeness verification.
- **🌊 Librosa Audio Signal Analysis:** Upload `.wav`, `.mp3`, `.ogg`, or `.flac` audio files or analyze preview streams to compute physical DSP measurements (Tempo, RMS Energy, Loudness, Spectral Centroid, Bandwidth, ZCR, STFT Spectrogram).
- **⏱️ Skip Probability & Retention:** 30-second mobile skip risk modeling, expected track completion percentage, and 5 listening context affinities (Cardio, Club, Focus, Late-Night, Commute).
- **🔥 Correlation Matrix & Dependencies:** Interactive Plotly/Seaborn correlation heatmap and feature dependency inspector.
- **📚 ML Model Architecture & Guide:** Technical breakdown of the pipeline, feature importances, and academic limitations.

---

### 4. Launching the Spotify SoundLab Web Platform (Port 3000)
The standalone web platform is served by our custom high-performance multithreaded backend (`server.py`):
```bash
python server.py
```
Then navigate to: **`http://localhost:3000`**

#### Strict Separation of Architectural Concerns:
- **`index.html`**: Clean semantic HTML5 markup containing search dropdowns, radial gauge SVG, radar spider canvas, Librosa DSP telemetry deck, and audience retention meters.
- **`style.css`**: Complete design system tokens, CSS variables, glassmorphism filters, tactile range potentiometers, and responsive grid layouts.
- **`app.js`**: Client-side reactive controller communicating asynchronously with backend REST endpoints, handling live waveform animation, audio transport dock, and search navigation.
- **`server.py`**: Python backend serving static assets and exposing high-speed endpoints:
  - `GET /api/search?q=...`: Sub-50ms search over 114,000 tracks.
  - `GET /api/preview?title=...&artist=...&track_id=...`: Smart audio preview & artwork resolver (Spotify API with iTunes zero-key fallback).
  - `POST /api/predict`: Real-time inference on `spotify_popularity_model.joblib` with 95% confidence intervals and feature completeness.
  - `POST /api/analyze_audio`: Direct Librosa DSP extraction from audio streams or uploaded binary buffers.

#### Zero-Key Playback & Spotify API Credential Support:
- **Zero-Key Engine (Default):** Works out-of-the-box with zero setup! Uses Spotify's official embed player (`open.spotify.com/embed/track/...`) for authenticated session playback and the public iTunes Search API for 30s audio previews across English and global hits.
- **Spotify Developer Credentials (Optional):** Add `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env` for direct Spotify Web API requests. See `.env.example`.

---

## 🔮 Future Improvement Roadmap

1. **Artist Stardom Metrics:** Integrate artist monthly listeners, follower count, and historical average track popularity via the Spotify Web API.
2. **Release Recency & Temporal Decay:** Add track release year and days since release to model playlist lifecycle and popularity decay over time.
3. **Lyric NLP & Sentiment Analysis:** Extract lyric themes, readability, and emotional sentiment using NLP embeddings (e.g. TF-IDF or transformer embeddings).


