from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = YOLO("version2.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file.stream).convert("RGB")
            img_np = np.array(image)

            results = model.predict(img_np, conf=0.5)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_img)

            # Convert ảnh sang base64 để hiển thị trên HTML
            buffer = io.BytesIO()
            result_pil.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return render_template("index.html", img_data=img_str)

    return render_template("index.html", img_data=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)
