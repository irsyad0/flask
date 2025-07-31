# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model, Model
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import traceback

app = Flask(__name__)

tf.random.set_seed(42)
np.random.seed(42)

# --- Konfigurasi (Tidak Berubah) ---
MODEL_PATH = 'model/model_deteksi_adam_0.0002.h5'
UNET_MODEL_PATH = 'model/unet_lip_segmentation_model_crop.h5'

# --- Variabel Global (Tidak Berubah) ---
main_model = None
unet_segmentation_model = None
label_encoder = None
feature_extractor_model = None
TARGET_SIZE_CNN = (224, 224)
UNET_IMG_WIDTH = 256
UNET_IMG_HEIGHT = 256

# --- Fungsi Bantuan ---

def load_all_models_and_encoder():
    """Memuat semua model dan encoder saat aplikasi dimulai."""
    global main_model, unet_segmentation_model, label_encoder, feature_extractor_model
    print("Loading U-Net Lip Segmentation Model...")
    try:
        unet_segmentation_model = load_model(UNET_MODEL_PATH)
        print("✅ U-Net model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading U-Net model: {e}")
    
    print("Loading Main CNN Model...")
    try:
        main_model = load_model(MODEL_PATH)
        # MODIFIKASI KUNCI: Ambil output dari layer GlobalAveragePooling2D
        # Gunakan nama yang sesuai dari model.summary() Anda, yaitu 'global_average_pooling2d_7'
        feature_extractor_model = Model(inputs=main_model.input, outputs=main_model.get_layer('global_average_pooling2d_7').output)
        print("✅ Main CNN model and feature extractor created from 'global_average_pooling2d_7'.")
    except Exception as e:
        print(f"❌ Error loading Main CNN model: {e}")

# --- FUNGSI INTI YANG BARU ---

def process_image_pipeline(img_original_array, unet_model):
    """
    Fungsi ini adalah gabungan dari semua langkah di preprocess_dataset.py.
    Ia mengambil gambar asli dan mengembalikan DUA hal:
    1. Gambar pra-pemrosesan akhir (hitam-putih) untuk disimpan/ditampilkan.
    2. Gambar yang siap untuk dimasukkan ke model CNN (3 channel, dinormalisasi).
    """
    # --- Langkah 1: Segmentasi dan Cropping dengan U-Net ---
    img_rgb_for_unet = cv2.cvtColor(img_original_array, cv2.COLOR_BGR2RGB)
    img_resized_for_unet = cv2.resize(img_rgb_for_unet, (UNET_IMG_WIDTH, UNET_IMG_HEIGHT))
    img_normalized_for_unet = img_resized_for_unet / 255.0
    img_input_for_unet = np.expand_dims(img_normalized_for_unet, axis=0)
    prediction_mask = unet_model.predict(img_input_for_unet, verbose=0)[0]
    predicted_mask_unet = (prediction_mask > 0.5).astype(np.uint8) * 255
    predicted_mask_display = cv2.resize(predicted_mask_unet, (img_original_array.shape[1], img_original_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(predicted_mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Peringatan: U-Net tidak menemukan kontur bibir.")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    padding = 15
    x_start, y_start = max(0, x - padding), max(0, y - padding)
    x_end, y_end = min(img_original_array.shape[1], x + w + padding), min(img_original_array.shape[0], y + h + padding)
    roi_lip_color = img_original_array[y_start:y_end, x_start:x_end]
    mask_for_roi = predicted_mask_display[y_start:y_end, x_start:x_end]
    if roi_lip_color.size == 0: return None, None
    roi_lip_color = cv2.bitwise_and(roi_lip_color, roi_lip_color, mask=mask_for_roi)

    # --- Langkah 2: Preprocessing Lanjutan pada Hasil Crop ---
    img_gray = cv2.cvtColor(roi_lip_color, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_lip_roi_clahe = clahe.apply(img_gray)

    kernels, gabor_filtered_images = [], []
    for freq in [0.2, 0.3, 0.4]:
        for sigma in [2, 3, 4]:
            for theta in np.arange(0, np.pi, np.pi / 8):
                try: kernels.append(gabor_kernel(frequency=freq, theta=theta, sigma_x=sigma, sigma_y=sigma))
                except: continue
    
    if kernels:
        for k in kernels: gabor_filtered_images.append(ndi.convolve(processed_lip_roi_clahe.astype(float), k.real, mode='wrap'))
        gabor_output = np.max(gabor_filtered_images, axis=0)
        gabor_output = cv2.normalize(gabor_output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        gabor_output = processed_lip_roi_clahe

    binarized_image = cv2.adaptiveThreshold(gabor_output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5)
    kernel_opening = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel_opening)
    kernel_closing = np.ones((2, 2), np.uint8)
    final_bw_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel_closing)
    
    # --- Persiapan untuk Model CNN ---
    image_for_cnn = cv2.cvtColor(final_bw_image, cv2.COLOR_GRAY2RGB)
    image_for_cnn = cv2.resize(image_for_cnn, TARGET_SIZE_CNN)
    image_for_cnn_normalized = (image_for_cnn.astype(np.float32) / 127.5) - 1.0
    
    return final_bw_image, image_for_cnn_normalized


def identify_person_with_multiple_embeddings(query_embedding, database_embeddings_map, threshold=0.90):
    """
    Fungsi ini membandingkan embedding yang di-scan dengan daftar embedding dari database.
    Meskipun namanya 'multiple', ini akan bekerja dengan benar meskipun hanya ada satu pengguna di map.
    threshold=0.97 berarti skor kemiripan harus 97% atau lebih untuk dianggap cocok.
    Nilai ini lebih ketat dan cocok untuk verifikasi.
    """
    max_similarity_for_user = -1.0
    identified_id = "Unknown"
    
    query_embedding_reshaped = query_embedding.reshape(1, -1)
    
    # Loop melalui setiap pengguna dalam data yang dikirim (dalam kasus ini, hanya ada satu)
    for username, list_of_db_embeddings in database_embeddings_map.items():
        # Cari kemiripan tertinggi untuk pengguna ini
        user_max_sim = -1.0
        for db_embedding in list_of_db_embeddings:
            db_embedding_reshaped = np.array(db_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embedding_reshaped, db_embedding_reshaped)[0][0]
            if similarity > user_max_sim:
                user_max_sim = similarity
        
        # Jika kemiripan tertinggi untuk pengguna ini melebihi ambang batas, anggap teridentifikasi
        if user_max_sim >= threshold:
            # Hanya set ID jika benar-benar cocok di atas threshold yang ketat
            max_similarity_for_user = user_max_sim
            identified_id = username
        else:
            # Jika tidak, pastikan untuk mengembalikan skor tertinggi yang ditemukan, meskipun di bawah threshold
            max_similarity_for_user = user_max_sim

            
    return identified_id, float(max_similarity_for_user)


# --- Rute API Flask ---

@app.route('/generate_embedding', methods=['POST'])
def generate_embedding_endpoint():
    """Endpoint untuk pendaftaran. Mengembalikan embedding dan gambar pra-pemrosesan akhir."""
    if feature_extractor_model is None or unet_segmentation_model is None:
        return jsonify({'status': 'error', 'error': 'Model tidak dimuat di server.'}), 503

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'error': 'Tidak ada data gambar.'}), 400

        image_data_b64 = data['image']
        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',', 1)[1]
        
        # Baca gambar sebagai BGR, karena fungsi pipeline kita menggunakannya
        image_binary = base64.b64decode(image_data_b64)
        img_np = np.frombuffer(image_binary, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Jalankan seluruh pipeline
        image_to_save, image_for_cnn = process_image_pipeline(img_bgr, unet_segmentation_model)
        
        if image_to_save is None:
            return jsonify({'status': 'error', 'error': 'Gagal memproses gambar (segmentasi atau langkah lain gagal).'}), 400

        # Ekstrak embedding dari gambar yang sudah siap untuk CNN
        input_for_embedding = np.expand_dims(image_for_cnn, axis=0)
        embedding = feature_extractor_model.predict(input_for_embedding, verbose=0)[0]
        embedding_b64 = base64.b64encode(','.join(map(str, embedding.tolist())).encode('utf-8')).decode('utf-8')

        # Encode gambar pra-pemrosesan akhir (hitam-putih) untuk dikirim ke PHP
        _, buffer = cv2.imencode('.png', image_to_save)
        image_to_save_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'embedding': embedding_b64,
            'preprocessed_image_b64': image_to_save_b64
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': f'Kesalahan internal server: {str(e)}'}), 500
    
@app.route('/generate_and_check_embedding', methods=['POST'])
def generate_and_check_embedding_endpoint():
    """
    Endpoint untuk pendaftaran yang menghasilkan embedding DAN memeriksa duplikat
    dengan embedding yang sudah ada untuk pengguna tersebut.
    """
    if feature_extractor_model is None or unet_segmentation_model is None:
        return jsonify({'status': 'error', 'error': 'Model tidak dimuat di server.'}), 503

    try:
        data = request.get_json()
        # Validasi input
        if not data or 'image' not in data or 'existing_embeddings' not in data:
            return jsonify({'status': 'error', 'error': 'Request tidak valid. `image` atau `existing_embeddings` tidak ada.'}), 400

        image_data_b64 = data['image']
        existing_embeddings_b64 = data['existing_embeddings'] # Ini adalah list of Base64 strings

        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',', 1)[1]
        
        image_binary = base64.b64decode(image_data_b64)
        img_np = np.frombuffer(image_binary, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Jalankan pipeline untuk mendapatkan gambar yang akan disimpan dan dianalisis
        image_to_save, image_for_cnn = process_image_pipeline(img_bgr, unet_segmentation_model)
        
        if image_to_save is None:
            return jsonify({'status': 'error', 'error': 'Gagal memproses gambar.'}), 400

        # Hasilkan embedding baru
        input_for_embedding = np.expand_dims(image_for_cnn, axis=0)
        new_embedding = feature_extractor_model.predict(input_for_embedding, verbose=0)[0]

        # --- Logika Pengecekan Duplikat ---
        # Definisikan ambang batas kemiripan yang sangat tinggi untuk dianggap duplikat
        DUPLICATE_THRESHOLD = 0.99 
        
        for old_embedding_b64 in existing_embeddings_b64:
            try:
                old_embedding_str = base64.b64decode(old_embedding_b64).decode('utf-8')
                old_embedding = np.array([float(x) for x in old_embedding_str.split(',')])
                
                # Hitung kemiripan
                similarity = cosine_similarity(new_embedding.reshape(1, -1), old_embedding.reshape(1, -1))[0][0]
                
                if similarity >= DUPLICATE_THRESHOLD:
                    print(f"DUPLICATE DETECTED! Similarity: {similarity:.4f} >= {DUPLICATE_THRESHOLD}")
                    # Jika ditemukan duplikat, langsung hentikan dan kirim respons khusus
                    return jsonify({
                        'status': 'duplicate_found',
                        'error': 'Gambar ini terlalu mirip dengan yang sudah Anda daftarkan sebelumnya. Silakan gunakan gambar yang berbeda.'
                    }), 200 # Kirim 200 OK karena ini bukan error server, tapi validasi bisnis
            except Exception as e:
                print(f"Warning: Gagal memproses salah satu embedding lama - {e}")
                continue # Lanjutkan ke embedding berikutnya

        # Jika loop selesai tanpa menemukan duplikat, lanjutkan seperti biasa
        new_embedding_b64 = base64.b64encode(','.join(map(str, new_embedding.tolist())).encode('utf-8')).decode('utf-8')

        _, buffer = cv2.imencode('.png', image_to_save)
        image_to_save_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'embedding': new_embedding_b64,
            'preprocessed_image_b64': image_to_save_b64
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': f'Kesalahan internal server: {str(e)}'}), 500

@app.route('/identify', methods=['POST'])
def identify_endpoint():
    """Endpoint untuk identifikasi. Logika tidak perlu banyak berubah."""
    if feature_extractor_model is None or unet_segmentation_model is None:
        return jsonify({'status': 'error', 'error': 'Model tidak dimuat di server.'}), 503

    try:
        data = request.get_json()
        if not data or 'image' not in data or 'database_embeddings' not in data:
            return jsonify({'status': 'error', 'error': 'Request tidak valid.'}), 400

        scanned_image_b64 = data['image']
        if ',' in scanned_image_b64:
            scanned_image_b64 = scanned_image_b64.split(',', 1)[1]
        
        image_binary = base64.b64decode(scanned_image_b64)
        img_np = np.frombuffer(image_binary, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        image_to_save, image_for_cnn = process_image_pipeline(img_bgr, unet_segmentation_model)

        if image_to_save is None:
            return jsonify({'status': 'error', 'error': 'Gagal memproses gambar scan.'}), 400

        input_for_embedding = np.expand_dims(image_for_cnn, axis=0)
        scanned_embedding = feature_extractor_model.predict(input_for_embedding, verbose=0)[0]

        decoded_db_embeddings_map = {}
        for username, list_of_b64_embeddings in data['database_embeddings'].items():
            user_embeddings_list = []
            for b64_embedding_str in list_of_b64_embeddings:
                try:
                    embedding_str = base64.b64decode(b64_embedding_str).decode('utf-8')
                    user_embeddings_list.append([float(x) for x in embedding_str.split(',')])
                except: continue
            if user_embeddings_list:
                decoded_db_embeddings_map[username] = user_embeddings_list
        
        if not decoded_db_embeddings_map:
            return jsonify({'status': 'error', 'error': 'Tidak ada embedding terdaftar yang valid.'}), 500

        # Lakukan identifikasi. Fungsi ini sekarang akan mengembalikan username jika cocok, atau "Unknown" jika tidak.
        identified_as, similarity_score = identify_person_with_multiple_embeddings(scanned_embedding, decoded_db_embeddings_map)

        _, buffer = cv2.imencode('.png', image_to_save)
        image_to_save_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'identified_as': identified_as, # Akan berisi username atau "Unknown"
            'similarity_score': similarity_score, # Tetap dikirim, tapi tidak ditampilkan di UI
            'preprocessed_image': image_to_save_b64
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': f'Kesalahan internal server: {str(e)}'}), 500

# --- Blok Eksekusi Utama ---
if __name__ == '__main__':
    load_all_models_and_encoder()
    app.run(host='0.0.0.0', port=5000, debug=True)