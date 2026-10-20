# AI BrowserGuard

AI BrowserGuard is an explainable, traditional machine-learning application that estimates phishing risk from URL structure. It uses the UCI [PhiUSIIL Phishing URL dataset](https://archive.ics.uci.edu/dataset/898/phiusiil+phishing+url+website), licensed under CC BY 4.0.

## Design

Only the dataset's `URL` and `label` columns are used. Every model input is calculated locally by `src/feature_extraction.py`; webpage HTML, redirects, reputation services, and all other dataset columns are excluded. Label `0` is treated as phishing and label `1` as legitimate.

The application trains a balanced Logistic Regression model, saves it with Joblib, and explains predictions with SHAP when available (with the equivalent linear contribution fallback). Metrics are calculated on a stratified holdout set.

The Home hero uses the original lightweight SVG illustration at
`assets\browserguard_hero.svg`; it is a project-created local asset and
does not depend on an external image service.

## Windows setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place `PhiUSIIL_Phishing_URL_Dataset.csv` in `data\raw\`. The loader also accepts the first CSV in that directory if the filename differs.

## Train and run

```powershell
python train.py
python -m streamlit run app.py
```

Training writes `models\url_model.joblib` and `models\url_model.metrics.json`. Training may take several minutes because the source data contains more than 200,000 records.

Run tests with:

```powershell
python -m pytest
```

## Streamlit Community Cloud deployment

The application uses `models\url_model.joblib` when that local artifact is
present. If it is absent, `src\prediction.py` downloads the model from the
private Hugging Face repository `kritika2005/browserguard-model` through
`huggingface_hub`. Hugging Face's local cache prevents repeated downloads.
For anonymous downloads, the repository must be Public in Hugging Face
settings; this application does not change repository visibility.

Public downloads do not require a token. If the repository remains private,
add a secret named `HF_TOKEN` with a read-enabled token in Streamlit Cloud,
or set `HF_TOKEN` for local fallback testing. Never commit the token or place
it in source code.

The runtime reads metrics from the `metrics` field inside the Joblib bundle;
the separately generated `url_model.metrics.json` is a training-side report
and is not required for application startup.

## Limitations and responsible use

The result is an estimated statistical risk, not a safety guarantee. False positives and false negatives are possible, especially for new attack patterns. HTTPS alone does not establish legitimacy. The application never opens, resolves, downloads, or sends submitted URLs to an external service.
