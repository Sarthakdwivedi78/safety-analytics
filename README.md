# Nightshift Lab

Streamlit capstone for nighttime vision sensor attenuation in dense urban platoons. It includes a reproducible research model, a telemetry audit, STPA, and a simulated emergency braking governor.

## Run locally

Use Python 3.12. In the repository root:

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m streamlit run streamlit_app.py

On macOS or Linux, replace .venv\Scripts\python with .venv/bin/python.

    python -m unittest discover -s tests -v
    python train.py
    python train.py --csv path/to/telemetry.csv --target 0.05

## Deploy to Streamlit Community Cloud

1. Create a private GitHub repository named nightshift-lab and push this package to branch main. Keep uploaded telemetry, credentials, local environments, and trained binary artifacts outside Git.
2. Open https://share.streamlit.io and sign in. Link the GitHub account and authorize access to the repository if prompted.
3. Create an app from the repository. Select branch main and entry point streamlit_app.py. Choose Python 3.12 in Advanced settings. Verify the app's viewing permissions in its sharing settings.
4. After the app opens, inspect the synthetic model results, run the Low camera with obstacle preset, and switch to the NHTSA archive to verify observed-only mode.

The model trains from the selected telemetry, so the app requires no external model server, secrets, or downloaded weights.

## Data and reproduction

- Synthetic mode generates 4,200 observations in 150 independent sessions with seed 24021. Synthetic relationships are assumptions of the generator.
- The ADS context file is a projection of the official NHTSA SGO archive through June 2025. It contains lighting, pre-crash subject speed converted from mph to km/h, roadway type, and report year. It retains the highest report version per report ID (1,899 report IDs). Multiple reports can represent the same incident.
- The exact cohort is Dark-No-Lights AND Camera_Blind. NHTSA records do not supply the second field or a usable outcome denominator. The app will not train from that archive.
- An uploaded CSV can be mapped interactively. For the CLI use the canonical column names from the telemetry template. Use stable session IDs shared by dependent observations. For deployment validation, use independent vehicles or organizations as groups when correlations persist across sessions.
- The attenuation fit, camera-only baseline, and fusion classifier fit on training groups. Threshold selection uses separate calibration groups. ROC, average precision, Brier score, risk intervals and false negatives are evaluated on held-out groups. The model gate is a research candidate only.

Sources: https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting and the archived ADS data dictionary linked there.
