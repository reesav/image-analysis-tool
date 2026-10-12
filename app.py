import io
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from measures import measure_image

st.set_page_config(page_title="Image Measures", layout="wide")
st.title("Image Measures")
st.write("Drag in one or more images, click **Run**, and download the results as a spreadsheet. "
         "Cyan outlines show where people were found; pink boxes show faces.")

files = st.file_uploader("Drop images here (JPG or PNG)", type=["jpg", "jpeg", "png"],
                         accept_multiple_files=True)

LABELS = {
    "filename": "Image",
    "n_people": "Number of people",
    "pct_people": "% of image covered by people",
    "face_detected": "Face found?",
    "n_faces": "Number of faces",
    "largest_face_pct": "Largest face (% of image)",
}

if st.button("Run", type="primary", disabled=not files):
    rows, overlays = [], {}
    bar = st.progress(0.0)
    for i, f in enumerate(files):
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.warning(f"{f.name} could not be read as an image, skipped.")
            continue
        vals, overlay = measure_image(img)
        rows.append({"filename": f.name, **vals})
        overlays[f.name] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        bar.progress((i + 1) / len(files))

    if rows:
        df = pd.DataFrame(rows).rename(columns=LABELS)
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.download_button("Download spreadsheet (CSV)", df.to_csv(index=False),
                           "image_measures.csv", "text/csv")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, rgb in overlays.items():
                ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                z.writestr(f"overlay_{name.rsplit('.', 1)[0]}.png", png.tobytes())
        c2.download_button("Download overlay images (ZIP)", buf.getvalue(),
                           "overlays.zip", "application/zip")

        st.subheader("Check the results")
        cols = st.columns(3)
        for k, (name, rgb) in enumerate(overlays.items()):
            cols[k % 3].image(rgb, caption=name, use_container_width=True)

with st.expander("How to read the columns"):
    st.markdown("""
- **% of image covered by people**: share of the picture's pixels inside a person outline (clothing and headwear included; overlapping people counted once).
- **Largest face (% of image)**: size of the biggest face box relative to the whole image.
- Always glance at the overlays: automatic detection can miss stylised or unusual figures.
""")
