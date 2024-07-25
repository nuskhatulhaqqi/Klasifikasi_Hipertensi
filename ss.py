# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import random



# Load dataset
# @st.cache_data
def load_data():
    # df = pd.read_csv("https://raw.githubusercontent.com/nuskhatulhaqqi/dataset/main/data_hipertensi.csv")
    df = pd.read_csv('dataset_fix.csv')
    return df

df = load_data()

# Preprocessing functions
def preprocessing(df):
    # Ensure columns are strings before using .str accessor
    df['Umur Tahun'] = df['Umur Tahun'].astype(str).str.replace('Tahun', '').str.replace('tahun', '').astype(int)
    df['Tinggi'] = df['Tinggi'].astype(str).str.replace('cm', '').astype(float)
    df['Berat Badan'] = df['Berat Badan'].astype(str).str.replace('kg', '').astype(float)
    df['Sistole'] = df['Sistole'].astype(str).str.replace('mm', '').astype(int)
    df['Diastole'] = df['Diastole'].astype(str).str.replace('Hg', '').astype(int)
    df['Nafas'] = df['Nafas'].astype(str).str.replace('/menit', '').astype(int)
    df['Detak Nadi'] = df['Detak Nadi'].astype(str).str.replace('/menit', '').astype(int)

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[['Jenis Kelamin']]).toarray()
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Jenis Kelamin']))
    df_one_hot = pd.concat([df[['Umur Tahun', 'Tinggi', 'Berat Badan', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']], df_encoded, df[['Diagnosa 1']]], axis=1)

    df_akhir = pd.concat([df[['Umur Tahun', 'Tinggi', 'Berat Badan', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']], df_encoded, df[['Diagnosa 1']]], axis=1)

    scaler = MinMaxScaler()
    fitur_numerik = ['Umur Tahun', 'Tinggi', 'Berat Badan', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi', 'Jenis Kelamin_L', 'Jenis Kelamin_P']
    df_akhir[fitur_numerik] = scaler.fit_transform(df_akhir[fitur_numerik])
    min_max =df_akhir[fitur_numerik]

    return df_akhir, scaler, encoder, df_one_hot,min_max, df

df_akhir, scaler, encoder, df_one_hot,min_max,df= preprocessing(df)

# Splitting data
X = df_akhir.drop('Diagnosa 1', axis=1)
y = df_akhir['Diagnosa 1']
# X_train_resampled, X_test, y_train_resampled,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Applying SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)


def validitas(kk, data_train, data_train_y, distance_metric='euclidean', p=3):
    # Mengonversi data train dan label ke numpy array
    data_train_np = data_train.to_numpy()
    data_train_y_np = data_train_y.to_numpy()
    validitas, distance, terdekat, label, cek, proses = [], [], [], [], [], []

    for i in range(len(data_train_np)):
        if distance_metric == 'euclidean':
            # Hitung jarak Euclidean
            jarak = np.sqrt(np.sum((data_train_np[i] - data_train_np) ** 2, axis=1))
        elif distance_metric == 'manhattan':
            # Hitung jarak Manhattan
            jarak = np.sum(np.abs(data_train_np[i] - data_train_np), axis=1)
        elif distance_metric == 'minkowski':
            # Hitung jarak Minkowski dengan parameter p
            jarak = np.sum(np.abs(data_train_np[i] - data_train_np) ** p, axis=1) ** (1 / p)
        elif distance_metric == 'hamming':
            # Hitung jarak Hamming
            jarak = np.sum(data_train_np[i] != data_train_np, axis=1) / data_train_np.shape[1]
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        jarak[i] = np.inf  # Abaikan jarak ke dirinya sendiri dengan mengaturnya ke tak hingga
        # Dapatkan indeks dari k-tetangga terdekat
        distance.append(jarak)
        indeks_terdekat = np.argsort(jarak)[:kk]
        terdekat.append(indeks_terdekat)
        # Hitung skor validitas
        label_terdekat = data_train_y_np[indeks_terdekat]
        label.append(label_terdekat)
        label_cek = label_terdekat == data_train_y_np[i]
        cek.append(label_cek)
        valid = np.mean(label_terdekat == data_train_y_np[i])
        validitas.append(valid)
      
    proses.append((distance, terdekat, label, cek, validitas))
    return validitas, proses

def distance_metric_func(data_train_np, data_test_np, distance_metric, p=3):
    if distance_metric == 'euclidean':
        return np.sqrt(((data_test_np[:, np.newaxis, :] - data_train_np[np.newaxis, :, :]) ** 2).sum(axis=2))
    elif distance_metric == 'manhattan':
        return np.abs(data_test_np[:, np.newaxis, :] - data_train_np[np.newaxis, :, :]).sum(axis=2)
    elif distance_metric == 'minkowski':
        return np.power(np.abs(data_test_np[:, np.newaxis, :] - data_train_np[np.newaxis, :, :]), p).sum(axis=2) ** (1 / p)
    elif distance_metric == 'hamming':
        return (data_test_np[:, np.newaxis, :] != data_train_np[np.newaxis, :, :]).mean(axis=2)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def calculate_distances(data_train, data_test, distance_metric, p=3):
    data_train_np = data_train.to_numpy()
    data_test_np = data_test.to_numpy()
    jarak = distance_metric_func(data_train_np, data_test_np, distance_metric, p)
    distance_df = pd.DataFrame(jarak.T, columns=[f'Dist{i + 1}' for i in range(len(data_test))])
    return distance_df

def WeighVoting(data_train, data_test, data_train_y, kk, smoothing_regulator, distance_metric='euclidean', p=3):
    sort,top_k,top_class,hitung,sort_class,proses_wf = [],[],[],[],[],[]
    # Hitung jarak antara data_train dan data_test 
    ED = calculate_distances(data_train, data_test, distance_metric, p)
    ED_np = ED.to_numpy()
    validi,proses = validitas(kk, data_train, data_train_y, distance_metric, p)
    validitas_np = np.array(validi)
    data_train_y_np = data_train_y.to_numpy()

    # Hitung bobot W
    W_np = validitas_np[:, np.newaxis] / (ED_np + smoothing_regulator)
    W = pd.DataFrame(W_np, columns=[f'W{i+1}' for i in range(ED_np.shape[1])])
    W['Y'] = data_train_y.tolist()

    Final_predic = []
    for i in range(W_np.shape[1]):
        # Sortir berdasarkan bobot dan ambil kk tetangga terdekat
        sorted_indices = np.argsort(-W_np[:, i])
        sort.append(sorted_indices)
        top_k_indices = sorted_indices[:kk]
        top_k.append(top_k_indices)

        # Hitung frekuensi kelas dari tetangga terdekat
        top_k_classes = data_train_y_np[top_k_indices]
        top_class.append(top_k_classes)
        unique, counts = np.unique(top_k_classes, return_counts=True)
        hitung.append([unique,counts])
        sorted_classes = unique[np.argsort(-counts)]
        sort_class.append(sorted_classes)

        # Tambahkan prediksi kelas dengan frekuensi tertinggi
        Final_predic.append(sorted_classes[0])
    valid = validitas_np.T
    proses_wf.append((valid,ED_np,W_np,sort,top_k,top_class,hitung,sort_class))
    return W, Final_predic,proses,proses_wf



def initialize_population(pop_size, k_range):
    max_bits = len(bin(k_range[1])) - 2  # Tentukan jumlah bit yang diperlukan untuk merepresentasikan nilai maksimum k
    population = []
    for _ in range(pop_size):
        random_num = random.randint(*k_range)  # Ambil angka acak dari 1 hingga k_range
        binary_representation = bin(random_num)[2:].zfill(max_bits)  # Ubah angka acak menjadi biner
        population.append(binary_representation)
    return population

# Konversi string biner ke integer
def binary_to_int(binary_str):
    return int(binary_str, 2)

# Fungsi fitness untuk mengevaluasi setiap individu dalam populasi
def fitness_function(k_binary, data_train, data_train_y, distance_metric):
    k = binary_to_int(k_binary)  # Konversi string biner ke integer
    if k > len(data_train):
        return 0  # Kembalikan 0 jika nilai k melebihi jumlah data training
    valid,proses= validitas(k, data_train, data_train_y, distance_metric)
    rata = np.mean(valid)
    return rata  # Hitung rata-rata validitas

# Seleksi individu berdasarkan fitness
def selection(population, fitnesses):
    total_fitness = sum(fitnesses)  # Hitung total fitness
    if total_fitness == 0 or np.isnan(total_fitness):  # Jika total fitness 0 atau NaN
        return random.choices(population, k=len(population))  # Pilih individu acak dari populasi
    probabilities = [f / total_fitness for f in fitnesses]  # Hitung probabilitas seleksi untuk setiap individu
    probabilities = [p if not np.isnan(p) else 1/len(probabilities) for p in probabilities]  # Tangani nilai NaN dalam probabilitas
    selected = np.random.choice(population, size=len(population), p=probabilities)  # Pilih individu berdasarkan probabilitas
    return selected.tolist()

# Proses crossover untuk menghasilkan keturunan
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)  # Tentukan titik crossover secara acak
    child1 = parent1[:crossover_point] + parent2[crossover_point:]  # Gabungkan bagian dari kedua orang tua untuk menghasilkan anak pertama
    child2 = parent2[:crossover_point] + parent1[crossover_point:]  # Gabungkan bagian dari kedua orang tua untuk menghasilkan anak kedua
    return child1, child2

# Proses mutasi untuk menghasilkan variasi dalam populasi
def mutate(k_binary, mutation_rate):
    k_list = list(k_binary)  # Ubah string biner menjadi list karakter
    for i in range(len(k_list)):
        if random.random() < mutation_rate:  # Jika angka acak lebih kecil dari laju mutasi
            k_list[i] = '0' if k_list[i] == '1' else '1'  # Balik nilai bit (0 menjadi 1, 1 menjadi 0)
    return ''.join(k_list)  # Gabungkan kembali list karakter menjadi string

# Algoritma genetika utama
def genetic_algorithm(data_train, data_train_y, distance_metric, pop_size, generations=50, mutation_rate=0.1, elitism_rate=0.1):
    k_range = (1, len(data_train))  # Tentukan rentang nilai k
    population = initialize_population(pop_size, k_range)  # Inisialisasi populasi awal
    max_bits = len(population[0])  # Hitung jumlah bit yang diperlukan
    num_elites = int(pop_size * elitism_rate)  # Jumlah individu elit yang akan disimpan

    for generation in range(generations):
        fitnesses = [fitness_function(k, data_train, data_train_y, distance_metric) for k in population]  # Hitung fitness untuk setiap individu

        # Simpan individu terbaik (Elitism sebelum crossover)
        elite_indices = np.argsort(fitnesses)[-num_elites:]
        elites = [population[i] for i in elite_indices]

        if all(f == 0 or np.isnan(f) for f in fitnesses):  # Jika semua fitness 0 atau NaN
            fitnesses = [f + 1e-6 for f in fitnesses]  # Tambahkan nilai kecil untuk menghindari masalah

        selected_population = selection(population, fitnesses)  # Seleksi individu berdasarkan fitness
        new_population = []

        # Lakukan crossover
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1 if i + 1 < len(selected_population) else 0]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        print(new_population,'new_population')
        # Elitism setelah crossover, sebelum mutasi
        crossover_fitnesses = [fitness_function(k, data_train, data_train_y, distance_metric) for k in new_population]
        elite_crossover_indices = np.argsort(crossover_fitnesses)[-num_elites:]
        elite_crossovers = [new_population[i] for i in elite_crossover_indices]

        # Mutasi individu yang tidak termasuk dalam elitism
        mutated_population = [mutate(individual, mutation_rate) for individual in new_population[num_elites:]]

        # Elitism setelah mutasi
        mutated_fitnesses = [fitness_function(k, data_train, data_train_y, distance_metric) for k in mutated_population]
        elite_mutated_indices = np.argsort(mutated_fitnesses)[-num_elites:]
        elite_mutated = [mutated_population[i] for i in elite_mutated_indices]

        # Gabungkan elit dengan populasi baru yang sudah dimutasi
        population = elite_crossovers + elite_mutated

        # Jika populasi tidak cukup, tambahkan kembali beberapa individu terpilih
        if len(population) < pop_size:
            population += selected_population[:pop_size - len(population)]

    best_k_binary = max(population, key=lambda k: fitness_function(k, data_train, data_train_y, distance_metric))  # Temukan individu dengan fitness terbaik
    best_k = binary_to_int(best_k_binary)  # Konversi nilai biner terbaik ke integer
    return best_k  # Kembalikan nilai k terbaik


# Saving the model and preprocessing objects
with open('model.pkl', 'wb') as file:
    pickle.dump((scaler, encoder, X_train_resampled, y_train_resampled), file)

# Saving the model and preprocessing objects
with open('model_no_smote.pkl', 'wb') as file:
    pickle.dump((scaler, encoder, X_train, y_train), file)

# Load model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        scaler, encoder, X_train_resampled, y_train_resampled = pickle.load(file)
    return scaler, encoder, X_train_resampled, y_train_resampled

def model(X_train_resampled, X_test, y_train_resampled, kk=151, sm=1e-5, jarak='manhattan', metode='knn'):
    if metode == 'knn':
        knn = KNeighborsClassifier(n_neighbors=kk, metric=jarak)
        knn.fit(X_train_resampled, y_train_resampled)  # Use y_train_resampled
        y_pred = knn.predict(X_test)
    else:
        _, y_pred,proses,proses_wv  = WeighVoting(X_train_resampled, X_test, y_train_resampled, kk=kk, smoothing_regulator=sm, distance_metric=jarak)

    cm = confusion_matrix(y_test, y_pred)
    accuracy_gmknn_sm = accuracy_score(y_test, y_pred)
    akurasi = accuracy_gmknn_sm * 100
    st.write(f'Akurasi: {accuracy_gmknn_sm * 100:.2f}%')
    st.write("Confusion Matrix Heatmap:")
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)
    return akurasi



def gabungan(akurasi1, akurasi2, akurasi3, akurasi4, akurasi5, akurasi6):
    skenarios = [
        "KNN", "SMOTE ENN + KNN",
        "MKNN", "SMOTE ENN + MKNN",
        "GMKNN", "SMOTE ENN + GMKNN"
    ]
    akurasi_test = [
        akurasi1,
        akurasi2,
        akurasi3,
        akurasi4,
        akurasi5,
        akurasi6
    ]
    plt.figure(figsize=(10, 6))

    # Plot the test accuracies
    plt.plot(skenarios, akurasi_test, label="Akurasi Test", marker='o')

    # Adding titles and labels
    plt.title("Perbandingan Akurasi Test Model Terbaik Antar Skenario")
    plt.xlabel("Skenario")
    plt.ylabel("Akurasi")
    plt.ylim(80, 100)  # Set the y-axis limit for better visualization
    plt.legend()
    plt.grid(True)

    # Adding annotation for each point
    for i, txt in enumerate(akurasi_test):
        plt.annotate(f'{txt:.4f}', (skenarios[i], akurasi_test[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Display the plot
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Use Streamlit to display the plot
    st.pyplot(plt)

# Fungsi untuk menampilkan hasil klasifikasi dengan warna latar belakang
def hasil(result):
    if result == 'BUKAN':
        st.success('Hasil Klasifikasi: Tidak Hipertensi')
    else:
        st.error('Hasil Klasifikasi: Hipertensi')



# Load model
scaler, encoder, X_train_resampled, y_train_resampled = load_model('model.pkl')

# Load model
scaler, encoder, X_train, y_train = load_model('model_no_smote.pkl')

# Menu
menu = ["Home", "Tentang Dataset", "Preprocessing Data", "Model", "Implementasi"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("IMPLEMENTASI ALGORITMA GENETIC MODIFIED K-NEAREST NEIGHBOR DALAM KASUS KLASIFIKASI PENYAKIT HIPERTENSI")
    st.subheader("Nama : Nuskhatul Haqqi")
    st.subheader("Nim : 200411100034")

elif choice == "Tentang Dataset":
    st.title("Tentang Dataset")
    st.write('Pada Penelitian ini akan dilakukan klasifikasi hipertensi menggunakan ALGORITMA GENETIC MODIFIED K-NEAREST NEIGHBOR.')
    st.title('Klasifikasi data inputan berupa:')
    st.write("1. Jenis Kelamin: Laki-laki/Perempuan.")
    st.write("2. Umur: Umur dalam tahun.")
    st.write("3. IMT: Indeks Massa Tubuh.")
    st.write("4. Tinggi: Tinggi badan dalam (cm).")
    st.write("5. Berat: Berat badan dalam (Kg).")
    st.write("6. Tekanan Sistolik: Tekanan sistolik adalah tekanan darah pada saat jantung memompa darah atau saat berkontraksi  (mm/Hg).")
    st.write("7. Tekanan Diastolik: Tekanan diastolik adalah tekanan darah pada saat jantung relaksasi dalam (mm/Hg).")
    st.write("8. Tingkat Nafas: Tingkat nafas dalam (per menit).")
    st.write("9. Denyut Nadi: Denyut nadi dalam (per menit).")
    st.write("10. Kelas: Menunjukkan apakah masuk dalam Hipertensi/Bukan Hipertensi.")
    st.title("Asal Data")
    st.write("Dalam Klasifikasi ini data yang digunakan berasal dari UPT Puskesmas Modopuro, Kecamatan Mojosari, Kabupaten Mojokerto, Provinsi Jawa Timur.")
    st.write("Total datanya yang digunakan ada 1750 data dengan 10 inputan terdiri dari 9 fitur dengan 1 label")

elif choice == "Preprocessing Data":
    st.subheader("Preprocessing Data")
    # Define tabs
    a, b,c,d= st.tabs(['Dataset Asal', 'Cleaning data', 'Encoder', 'Normalisasi'])
    # Home tab
    with a:
        "### Dataset"
        df_a = pd.read_csv("dataset_fix.csv")
        st.write(df_a)
    with b:
        "### Cleaning data"
        st.write(df)
    with c:
        "### Mengubah data kategorikal menjadi numerik menggunakan One Hot Encoding"
        st.write(df_one_hot)
    with d:
        "### Normalisasi menggunakan Min-Max Scaler"
        st.write(min_max)

elif choice == "Model":
    st.subheader("Model dan Confusion Matrix")
    # Define tabs
    a, b, c, d, e, f , g= st.tabs(['KNN', 'KNN + SMOTE ENN', 'MKNN', 'MKNN + SMOTE ENN', 'GMKNN', 'GMKNN + SMOTE ENN','Perbandingan Akurasi'])
    kumpulan_akurasi = []
    # Home tab
    with a:
        st.markdown("### Metode : K Nearest Neighbor (KNN)")
        st.write("Model terbaik KNN: \n K = 7 , Jarak = euclidean ")
        akurasi1 = model(X_train, X_test, y_train, kk=7, sm=1e-5, jarak='euclidean', metode='knn')
        kumpulan_akurasi.append(akurasi1)
    with b:
        st.markdown("### Metode : K Nearest Neighbor (KNN) + SMOTE ENN")
        st.write("Model terbaik KNN + SMOTE ENN: \n K = 9 , Jarak = euclidean ")
        akurasi2 = model(X_train_resampled, X_test, y_train_resampled, kk=9, sm=1e-5, jarak='euclidean', metode='knn')
        kumpulan_akurasi.append(akurasi2)
    with c:
        st.markdown("### Metode : Modified K Nearest Neighbor (MKNN)")
        st.write("Model terbaik MKNN: \n K = 3, Jarak = minkowski, smoothing_regulator = 0.1 ")
        akurasi3 = model(X_train, X_test, y_train, kk=3, sm=0.1, jarak='minkowski', metode='mknn')
        kumpulan_akurasi.append(akurasi3)
    with d:
        st.markdown("### Metode : Modified K Nearest Neighbor (MKNN) + SMOTE ENN")
        st.write("Model terbaik MKNN + SMOTE ENN: \n K = 5 , Jarak = minkowski, smoothing_regulator = 0")
        akurasi4 = model(X_train_resampled, X_test, y_train_resampled, kk=5, sm=0, jarak='minkowski', metode='mknn')
        kumpulan_akurasi.append(akurasi4)
    with e:
        st.markdown("### Metode : Genetic Modified K Nearest Neighbor (GMKNN)")
        st.write("Model terbaik GMKNN: \n Populasi = 300, K = 13 , Jarak = manhattan, smoothing_regulator = 0")
        akurasi5 = model(X_train, X_test, y_train, kk=13, sm=0, jarak='manhattan', metode='gmknn')
        kumpulan_akurasi.append(akurasi5)
    with f:
        st.markdown("### Metode : Genetic Modified K Nearest Neighbor (GMKNN) + SMOTE ENN")
        st.write("Model terbaik GMKNN + SMOTE ENN: \n Populasi = 200, K = 8 , Jarak = minkowski, smoothing_regulator = 0.1")
        akurasi6 = model(X_train_resampled, X_test, y_train_resampled, kk=8 ,sm=1e-5, jarak='minkowski', metode='gmknn')
        kumpulan_akurasi.append(akurasi6)
    with g:
        st.markdown("### Perbandingan Akurasi Untuk Berbagai Metode Klasifikasi")
        ak1,ak2,ak3,ak4,ak5,ak6 = kumpulan_akurasi[0],kumpulan_akurasi[1],kumpulan_akurasi[2],kumpulan_akurasi[3],kumpulan_akurasi[4],kumpulan_akurasi[5]

        # gabungan(0.9429, 0.9429, 0.9429, 0.9371, 0.9400, 0.9400)
        gabungan(ak1,ak2,ak3,ak4,ak5,ak6)



elif choice == "Implementasi":
    st.title("Implementasi")
    # Custom or default parameters
    # Home tab
    metode = st.selectbox("Pilih Metode", ['GMKNN', 'MKNN', 'KNN'])
    if metode == 'KNN':
        metode = 'KNN'
        SMOTE = st.checkbox("Menggunakan SMOTE ENN", value=False)
        custom_params = st.checkbox("Custom Parameter", value=False)
        if custom_params:
            k = st.selectbox("Pilih Nilai k", [3, 5, 7, 9, 11, 13, 15])
            distance = st.selectbox("Select distance metric", ['euclidean', 'manhattan', 'minkowski','hamming'])       
            if SMOTE == True:
                st.subheader("Metode KNN + SMOTE ENN")
            else:
                st.subheader("Metode KNN")
        else:
            if SMOTE == True:
                st.subheader("Metode KNN + SMOTE ENN")
                k = 9
                distance = 'euclidean'
            else:
                st.subheader("Metode KNN")
                k = 7
                distance = 'euclidean'

        
    elif metode == 'MKNN':
        metode = 'MKNN'
        SMOTE = st.checkbox("Menggunakan SMOTE ENN", value=False)
        custom_params = st.checkbox("Custom Parameter", value=False)
        if custom_params:
            k = st.selectbox("Pilih Nilai k", [3, 5, 7, 9, 11, 13, 15])
            distance = st.selectbox("Select distance metric", ['euclidean', 'manhattan', 'minkowski','hamming'])
            sm = st.selectbox("Pilih Smoothing regulator", [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])      
            if SMOTE == True:
                st.subheader("Metode MKNN + SMOTE ENN")
            else:
                st.subheader("Metode MKNN")
        else:
            if SMOTE == True:
                st.subheader("Metode MKNN + SMOTE ENN")
                k = 5
                distance = 'minkowski'
                sm = 0
            else:
                st.subheader("Metode MKNN")
                k = 3
                distance = 'minkowski'
                sm = 0.1
        
    else:
        metode = 'GMKNN'
        st.write(metode)
        SMOTE = st.checkbox("Menggunakan SMOTE ENN", value=False)
        custom_params = st.checkbox("Custom Parameter", value=False)
        if custom_params:
            k=7
            pop_size = st.number_input("Populasi", min_value=0)
            gen = st.number_input("Generasi (defauld = 10):", min_value=0)
            distance = st.selectbox("Select distance metric", ['euclidean', 'manhattan', 'minkowski','hamming'])
            sm = st.selectbox("Pilih Nilai Smoothing Regulator", [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])    
            if SMOTE == True:
                st.subheader("Metode GMKNN + SMOTE ENN")
            else:
                st.subheader("Metode GMKNN")
        else:
            if SMOTE == True:
                st.subheader("Metode GMKNN + SMOTE ENN")
                best_k=8
                pop_size = 10
                distance = 'minkowski'
                sm = 0.1
                gen= 10
                k = 8
            else:
                st.subheader("Metode GMKNN")
                best_k=13
                pop_size = 10
                distance = 'manhattan'
                sm = 0
                gen= 10
                k=13
        
    metode = metode

    # inputan data

    umur = st.number_input("Umur Tahun", min_value=0)
    tinggi = st.number_input("Masukkan tinggi badan (dalam cm):", min_value=0.0, format="%.2f")
    berat = st.number_input("Masukkan berat badan (dalam kg):", min_value=0.0, format="%.1f")
    # Konversi tinggi badan ke meter
    tinggi_m = tinggi / 100
    # Hitung IMT secara otomatis
    if tinggi_m > 0 and berat > 0:
        imt = berat / (tinggi_m ** 2)
        st.write(f"Indeks Massa Tubuh (IMT) Anda adalah: {imt:.2f}")
    else:
        imt = 0
        st.write("Masukkan tinggi badan dan berat badan yang valid untuk menghitung IMT.")
    sistole = st.number_input("Masukkan sistole (mmHg):", min_value=0)
    diastole = st.number_input("Masukkan diastole (mmHg):", min_value=0)
    nafas = st.number_input("Masukkan Nafas (/menit)", min_value=0)
    detak_nadi = st.number_input("Masukkan Detak Nadi (/menit)", min_value=0)
    jenis_kelamin = st.selectbox("Pilih Jenis Kelamin", ["L", "P"])

    if st.button("Submit"):
        # Preprocess input data
        custom_params = custom_params 
        SMOTE = SMOTE
        k = k
        distance = distance
        input_data = pd.DataFrame({
            # 'SMOTE' : [SMOTE],
            'Umur Tahun': [umur],
            'Tinggi': [tinggi],
            'Berat Badan': [berat],
            'IMT': [imt],
            'Sistole': [sistole],
            'Diastole': [diastole],
            'Nafas': [nafas],
            'Detak Nadi': [detak_nadi],
            'Jenis Kelamin_L': [1 if jenis_kelamin == 'L' else 0],
            'Jenis Kelamin_P': [1 if jenis_kelamin == 'P' else 0]
        })
        fitur_numerik = ['Umur Tahun', 'Tinggi', 'Berat Badan', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi', 'Jenis Kelamin_L', 'Jenis Kelamin_P']
        input_data[fitur_numerik] = scaler.transform(input_data[fitur_numerik])
        # Perform classification
        if metode == 'KNN':
            if SMOTE == True:
                knn = KNeighborsClassifier(n_neighbors=k, metric=distance)
                knn.fit(X_train_resampled,y_train_resampled)
                predictions = knn.predict(input_data)
            else:
                knn = KNeighborsClassifier(n_neighbors=k, metric=distance)
                knn.fit(X_train_resampled,y_train_resampled)
                predictions = knn.predict(input_data)
            hasil(predictions[0])
            st.write('Menggunakan Parameter:')
            data = {
                'Parameter': ['SMOTE ENN', 'K Terbaik', 'Matriks Jarak'],
                'Nilai': [SMOTE, k, distance]
            }
            # Membuat DataFrame
            df = pd.DataFrame(data)
            # Mentrasposisi DataFrame agar parameter menjadi kolom
            df_transposed = df.set_index('Parameter').T
            # Menampilkan DataFrame sebagai tabel
            st.table(df_transposed)
        elif metode == 'MKNN':
            sm = sm
            if SMOTE == True:
                _, prediction,proses,proses_wv = WeighVoting(X_train_resampled, input_data, y_train_resampled, kk=k, smoothing_regulator=sm, distance_metric=distance)
            else:
                _, prediction,proses,proses_wv= WeighVoting(X_train, input_data, y_train, kk=k, smoothing_regulator=sm, distance_metric=distance)
            hasil(prediction[0])
            st.write('Menggunakan Parameter:')
            data = {
                'Parameter': ['SMOTE ENN', 'K Terbaik', 'Matriks Jarak', 'Smoothing Regulator'],
                'Nilai': [SMOTE, k, distance, sm]
            }
            # Membuat DataFrame
            df = pd.DataFrame(data)
            # Mentrasposisi DataFrame agar parameter menjadi kolom
            df_transposed = df.set_index('Parameter').T
            # Menampilkan DataFrame sebagai tabel
            st.table(df_transposed)
        else:
            pop_size = pop_size
            sm = sm
            if custom_params == True:
                # best_k = genetic_algorithm(X_train, y_train, distance_metric=distance, pop_size=pop_size, generations=gen, mutation_rate=0.1)
                best_k = genetic_algorithm(X_train, y_train, distance_metric='euclidean', pop_size=10, generations=1, mutation_rate=0.1, elitism_rate=0.1)
            if SMOTE == True:
                _, prediction,proses,proses_wv = WeighVoting(X_train_resampled, input_data, y_train_resampled, kk=best_k, smoothing_regulator=sm, distance_metric=distance)
            else:
                _, prediction,proses,proses_wv = WeighVoting(X_train, input_data, y_train, kk=best_k, smoothing_regulator=sm, distance_metric=distance)
            # st.markdown(f'<p style="color:blue;">Hasil Prediksi Diagnosa: {prediction[0]} Hipertensi</p>', unsafe_allow_html=True)
            hasil(prediction[0])
            st.write('Menggunakan Parameter:')
            data = {
                'Parameter': ['SMOTE ENN', 'Populasi', 'Generasi', 'K Terbaik', 'Matriks Jarak', 'Smoothing Regulator'],
                'Nilai': [SMOTE, pop_size, gen, k, distance, sm]
            }
            # Membuat DataFrame
            df = pd.DataFrame(data)
            # Mentrasposisi DataFrame agar parameter menjadi kolom
            df_transposed = df.set_index('Parameter').T
            # Menampilkan DataFrame sebagai tabel
            st.table(df_transposed)

        if metode in ['GMKNN', 'MKNN']:
            y, z= st.tabs(['Proses Validitas Data', 'Proses Weighted Voting'])
            with y:
                '### Proses Validitas Data'
                # Displaying the results in a single row grid
                with st.expander('Keterangan'):
                    data = {
                        "Kolom": [
                            "Jarak",
                            "Terdekat",
                            "Label",
                            "Cek",
                            "Validitas"
                        ],
                        "Deskripsi": [
                            "Menghitung jarak antar data latih",
                            "Setelah jarak di urutkan berdasarkan terkecil ke terbesar di ambil nilai index sejumlah nilai k",
                            "Mengambil nilai label sesuai index terdekat",
                            "Dilakukan pengecekan jika label sama nilainya dengan label dari data yang validitas di cari maka di beri nilai 1 (diumpamakan True), jika salah 0 (diumpamakan False)",
                            "hasil perhitungan validitas"
                        ]
                    }
                    # Membuat DataFrame
                    df = pd.DataFrame(data)
                    # Menampilkan DataFrame sebagai tabel di Streamlit
                    st.table(df)
                    st.write('NB: Data yang di tampilkan hanya 5 data sebagai contoh')

                columns = st.columns(5)
                columns[0].write('Jarak')
                columns[1].write('Terdekat')
                columns[2].write('Label')
                columns[3].write('Cek')
                columns[4].write('Validitas')

                distance, terdekat, label, cek, valid= proses[0]
                for i in range(5):
                    col = st.columns(5)
                    col[0].write(distance[i])
                    col[1].write(terdekat[i])
                    col[2].write(label[i])
                    col[3].write(cek[i])
                    col[4].write(valid[i])
            with z:
                '### Proses Weighted Voting'
                with st.expander('Keterangan'):
                    data = {
                        "Kolom": [
                            "Validitas",
                            "Jarak",
                            "Hitung Bobot",
                            "Urutkan",
                            "Top-K Index",
                            "Top-K Kelas",
                            "Voting",
                            "Penentuan Kelas"
                        ],
                        "Deskripsi": [
                            "Nilai Validitas data latih",
                            "Perhitungan Jarak antara data latih dan data uji",
                            "Menghitung nilai bobot",
                            "Hasil perhitungan nilai bobot di urutkan berdasarkan terbesar ke terkecil (yang di tampilkan adalah index datanya)",
                            "Menampilkan nilai index sesuai jumlah nilai k yang di tentukan dari terbesar ke terkecil",
                            "Menampilkan kelas dari masing-masing index terdekat",
                            "Hasil voting untuk menentukan kelas mayoritas dan minoritas",
                            "Menampilkan kelas mayoritas yang dijadikan sebagai kelas dari data baru"
                        ]
                    }
                    # Membuat DataFrame
                    df = pd.DataFrame(data)
                    # Menampilkan DataFrame sebagai tabel di Streamlit
                    st.table(df)
                
                validi,ED,W_np,sort,top_k,top_class,hitung,sort_class= proses_wv[0]
                columns = st.columns(8)
                columns[0].write('Validitas')
                columns[1].write('Jarak')
                columns[2].write('Hitung Bobot')
                columns[3].write('Urutkan')
                columns[4].write('Top-K Index')
                columns[5].write('Top-K Kelas')
                columns[6].write('Voting')
                columns[7].write('Penentuan Kelas')

                for i in range(1):
                    col = st.columns(8)
                    col[0].write(validi)
                    col[1].write(ED)
                    col[2].write(W_np)
                    col[3].write(sort[i])
                    col[4].write(top_k[i])
                    col[5].write(top_class[i])
                    if (len(hitung[i][0]) != 1):
                        col[6].write(f'{hitung[i][0][0]} : {hitung[i][1][0]} \n {hitung[i][0][1]} : {hitung[i][1][1]} ')
                    else:
                        col[6].write(f'{hitung[i][0][0]} : {hitung[i][1][0]}')
                    col[7].write(sort_class[i][0])
            
            
        
