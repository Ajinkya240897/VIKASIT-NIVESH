VIKASIT NIVESH - Updated app bundle.

Changes made:
- UI title changed to 'VIKASIT NIVESH'.
- Company description now displayed in full (no trimming). If extremely large, a fallback shows first 4000 chars.
- Recommendation paragraphs formatted to be beginner-friendly and clearer with line breaks.
- requirements.txt updated to prefer numpy >= 1.26.0 to be compatible with latest Python versions (3.12+).
- App core logic preserved from your uploaded NIVESH app; only the changes above were applied.

Deployment notes:
- For Streamlit Cloud use Python 3.10/3.11 or ensure environment supports numpy>=1.26 wheels.
- If you run into build errors, run: pip install --upgrade pip setuptools wheel before deploying.