# Spotify-Project
## Proposed Project Workflow

Raw data → Clean → Harmonize → Engineer → Explore → Infer → Model → Validate → Interpret → Communicate

### Workflow Phases

1. **Acquire and inspect data (Weeks 1–2)**
   - Collect 2023, 2024, 2025, and Wrapped 2025 datasets
   - Inspect schemas and produce data-quality checks
2. **Clean and harmonize data (Weeks 1–3)**
   - Resolve missing values, duplicates, type issues, and inconsistencies
   - Build a comparable longitudinal dataset and a richer cross-platform dataset
3. **Feature engineering (Week 3)**
   - Create analysis features such as song age, log-transformed streams, and collaboration indicators
4. **Exploratory analysis (Weeks 4–5)**
   - Perform descriptive statistics and visualization
   - Compare distributions across years, artists, and release context
5. **Inference and hypothesis testing (Week 6)**
   - Apply ANOVA/Kruskal-Wallis and correlation analyses
   - Report effect sizes, confidence intervals, and adjusted significance where needed
6. **Modeling and diagnostics (Weeks 7–8)**
   - Build progressively richer regression models for streaming success
   - Run diagnostics for multicollinearity, residual behavior, and heteroskedasticity
7. **Cross-platform and concentration analysis (Week 9)**
   - Analyze relationships between Spotify and other platform metrics
   - Measure concentration of success using Pareto-style and Gini-based analysis
8. **Multivariate structure discovery (Week 10)**
   - Use PCA for dimensionality reduction
   - Cluster songs into data-supported profile groups
9. **Prediction and final integration (Week 11)**
   - Classify high-streaming songs with logistic and ML methods
   - Evaluate with held-out performance metrics and synthesize final conclusions

### Reproducibility Principles

- Programmatic data acquisition
- Version-controlled analysis
- Documented preprocessing and statistical decisions
- Shared data dictionary and auditable outputs by phase

## Collaborative task board (website)

The `/home/runner/work/Spotify-Project/Spotify-Project/index.html` page now supports live multi-user collaboration with Firebase Firestore.

### One-time setup

1. Create a Firebase project.
2. Enable:
   - **Authentication** → Anonymous sign-in
   - **Firestore Database** (production or test mode)
3. Copy `/home/runner/work/Spotify-Project/Spotify-Project/firebase-config.example.json` to `firebase-config.json` in the repo root and fill in your Firebase web config values.
4. Deploy/publish the site (for example, GitHub Pages).

After setup, everyone using the page URL will see task/note updates in real time.  
If `firebase-config.json` is missing or invalid, the page falls back to browser-local storage for single-user use.
