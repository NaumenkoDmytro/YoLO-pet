from flask import Flask, request, jsonify, send_from_directory, url_for
from ultralytics import YOLO
import os
import uuid

# Конфігурація Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.url_map.strict_slashes = False

# Створюємо папки для зберігання файлів, якщо таких не існує
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Завантажуємо YOLOv8 модель
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

# Роут для процессінгу відео
@app.route('/process-video', methods=['POST'])
def process_video():
    # Перевіряємо наявність файлу у запиті
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Перевіряємо навність імені файлу
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Генеруємо UUID для створення унікальності в імені файлу, робимо шлях для збереження і зберігаємо файл
    file_id = str(uuid.uuid4())
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id + '_' + file.filename)
    file.save(input_path)

    # Створення шляху для зберігання оброблених даних
    save_dir = os.path.join(app.config['PROCESSED_FOLDER'], file_id)
    os.makedirs(save_dir, exist_ok=True)

    # Процесінг відео за допомогою YOLOv8
    model.predict(source=input_path, save=True, save_dir=save_dir, project=save_dir, exist_ok=True)

    # Створення посилання для завантаження обробленого файлу
    output_dir = os.path.join(app.config['PROCESSED_FOLDER'], file_id, "predict")
    file_name = [f for f in os.listdir(output_dir) if f.startswith(file_id)][0]
    download_link = url_for('download_file', filename=os.path.join(output_dir, file_name), _external=True)

    # Поверння JSON файлу за допомогою функції Flask
    return jsonify({"download_link": download_link}), 200

# Роут для завантаження відео
@app.route('/downloads/<path:filename>')
def download_file(filename):
    # Поверенення файлу за допомогою функції send_from_directory
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

if __name__ == "__main__":
    # Запуск застостунку Flask
    app.run(host="0.0.0.0", port=5000)
