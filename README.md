# **Web Model Deployment Using Flask Menggunakan Data Tabular**
---
## Deskripsi
Projek Machine Learning kali ini melibatkan penerapan model yang telah dikembangkan dari data tabular. selain itu Proyek ini bertujuan untuk mendeploy model machine learning yang telah dibangun sebelumnya menggunakan Flask. Model tersebut dirancang untuk melakukan prediksi pendapatan seseorang berdasarkan input yang telah ditentukan, dengan menggunakan dataset income.csv, model tersebut berhasil memprediksi label yang benar berdasarkan kondisi "income_>50k" dengan tingkat akurasi sebesar 0.80. Oleh karena itu, model yang telah disusun pada modul sebelumnya akan diterapkan pada projek ini untuk mendukung analisis dan prediksi pada dataset yang sama.
Berikut atribut-atribut yang harus diinput ke dalam web predict:
- Age: data umur
- Education: data edukasi
- Occupation: data pekerjaan
- Hours per Week: jumlah jam yang dihabiskan untuk bekerja dalam seminggu
- Native Country: asal daerah

## Dataset
Dataset yang digunakan dalam proyek ini adalah dataset income dalam format CSV yang terdiri dari 43.957 data dengan 15 atribut, yaitu age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country, dan income_>50K. Dataset ini memiliki dua kelas, di mana kelas 1 mewakili individu yang memiliki pemasukan lebih dari 50K, sedangkan kelas 0 mewakili individu yang tidak memiliki pemasukan lebih dari 50K.

## **Struktur Project**
* Model Machine Learning: Model TabNet yang telah dilatih digunakan untuk memprediksi penghasilan berdasarkan data tabular.
* Aplikasi Flask: Terdapat dua halaman, satu untuk input data (index.html) dan satu untuk menampilkan hasil prediksi (predict.html).
* Styling: Terdapat file styles.css untuk memberikan tampilan yang menarik dan responsif.

## Langkah-Langkah 
### 1. Menyimpan model dalam format .h5 dan json dalam bentuk zip 
```bash
df.to_hdf('/content/drive/MyDrive/praktikumsmt7/modul6/income_.h5', key='data', mode='w')
```
```bash
filesaved = clf.save_model('content/drive/MyDrive/praktikumsmt7/modul6/modeltabular')
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the PyTorch TabNet library
* Perintah diatas digunakan untuk menginstal pustaka PyTorch TabNet. PyTorch TabNet adalah implementasi dari algoritma TabNet yang digunakan untuk pembelajaran mesin pada data tabular.
```bash
pip install pytorch_tabnet
```

### 4. Jalankan Flask
* menjalankan perintah ini akan memulai server pengembangan dan membuat aplikasi web yang dapat diakses pada alamat http://127.0.0.1:5000.

```bash
flask run
```
## Stuktur Code
* app.py: file utama Flask yang menangani routing dan integrasi model TabNet.
* templates/index.html: Template untuk halaman input data.
* templates/predict.html: Template untuk halaman hasil prediksi.
* static/css/styles.css: Berkas CSS untuk styling antarmuka web.

## Requirements.txt

```bash
Flask==3.0.0
Werkzeug==3.0.1
datetime==4.3.0
Numpy==1.23.5
Pandas==1.5.3
```
