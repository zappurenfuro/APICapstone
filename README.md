# CVScan - Resume Scanner and Job Matcher

CVScan adalah alat untuk memindai resume dan mencocokkannya dengan judul pekerjaan menggunakan pemrosesan bahasa alami dan kesamaan semantik.

## Features

- Ekstraksi teks dari file resume PDF dan DOCX
- Pembersihan dan pemrosesan data teks
- Pembuatan embedding menggunakan model bahasa terkini
- Pencocokan resume terhadap job title
- Optimalisasi memori untuk menangani dataset besar

## Requirements

- Python 3.8+
- Docker (opsional)

## Installation

### Menggunakan Docker (Direkomendasikan)

1. Clone repositori:
   ```bash
   git clone https://github.com/yourusername/cvscan.git
   cd cvscan

2. Build dan jalankan dengan Docker Compose:
   ```bash
    docker-compose up --build

### Instalasi Manual

1. Clone repositori:
   ```bash
    git clone https://github.com/yourusername/cvscan.git
    cd cvscan

2. Buat virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate

3. Instal dependensi:
   ```bash
    pip install -r requirements.txt

4. Jalankan aplikasi:
   ```bash
    python main.py

## Usage
1. Tempatkan file data CSV Anda di folder input:
    - 01_people.csv
    - 02_abilities.csv
    - 03_education.csv
    - 04_experience.csv
    - 05_person_skills.csv

2. Tempatkan file resume (PDF atau DOCX) di direktori root atau tentukan path-nya di file .env.

3. Jalankan aplikasi:
   ```bash
    python main.py

4. Lihat hasilnya di folder output.

## Configuration
Anda dapat mengatur konfigurasi aplikasi melalui file .env:
   ```
    # Input dan output folder
    INPUT_FOLDER=input
    OUTPUT_FOLDER=output

    # Pengaturan model
    MODEL_NAME=BAAI/bge-large-en-v1.5
    BATCH_SIZE=32
    TOP_N=5

    # Daftar file CV yang diproses
    CV_FILES=cv_hilda.pdf,cv_rakha.pdf

    # Pengaturan sistem
    RAM_DISK_SIZE=1024
   ```

## How to Use This Structure
1. Buat Struktur Direktori
   ```bash
    mkdir -p cvscan/src/utils

2. Salin Semua File
Pastikan setiap file berada di lokasi yang sesuai di dalam struktur direktori.
3. Build dan Jalankan dengan Docker
   ```bash
    cd cvscan
    docker-compose up --build

4. Untuk Pengembangan Lokal
   ```bash
    cd cvscan
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python main.py

#   A I M o d e l - A P I  
 #   A P I C a p s t o n e  
 