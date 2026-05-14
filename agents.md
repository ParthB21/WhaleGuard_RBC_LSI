# Role: Whale Conservation Data Engineer

## Project Context
Building a geospatial risk-assessment tool for the Gulf of St. Lawrence.
- **Models:** 3 variants (details in ML_Walkthrough.md).
- **Stack:** Python, Streamlit, Pydeck (for 3D heatmaps) or Folium.

## Standards
- **Modularity:** Keep model inference logic in `inference.py` and UI logic in `app.py`.
- **Date Handling:** Use `pandas.to_datetime` for all slider logic to avoid format errors.