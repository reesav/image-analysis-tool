# Image Measures (prototype)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

First run downloads the YOLO model (~20 MB).

Deploy: push this folder to a GitHub repo / Hugging Face Space (SDK: Streamlit, set private).
Add a new measure: write a function in measures.py and merge its dict in measure_image().
