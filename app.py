
# app.py

from flask import Flask, render_template, request, send_from_directory
import os
from model_utils import remove_background

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            img_path = os.path.join(UPLOAD_FOLDER, img.filename)
            img.save(img_path)

            result_path = os.path.join(RESULT_FOLDER, 'transparent_' + img.filename)
            remove_background(img_path, result_path)

            from pathlib import Path

            relative_result_path = Path(result_path).as_posix().split('static/')[-1]
            return render_template('index.html', result_image=relative_result_path)


    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
