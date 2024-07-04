# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN


# Load dataset
# @st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/nuskhatulhaqqi/dataset/main/data_hipertensi.csv")
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

    # encoder = OneHotEncoder()
    # encoded_data = encoder.fit_transform(df[['Jenis Kelamin']]).toarray()
    # df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Jenis Kelamin']))
    # OneHotEncoding
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

def validitas(kk, data_train, data_train_y, distance_metric, p=3):
    data_train_np = data_train.to_numpy()
    data_train_y_np = data_train_y.to_numpy()
    validitas = []

    for i in range(len(data_train_np)):
        if distance_metric == 'euclidean':
            jarak = np.sqrt(np.sum((data_train_np[i] - data_train_np) ** 2, axis=1))
        elif distance_metric == 'manhattan':
            jarak = np.sum(np.abs(data_train_np[i] - data_train_np), axis=1)
        elif distance_metric == 'minkowski':
            jarak = np.sum(np.abs(data_train_np[i] - data_train_np) ** p, axis=1) ** (1 / p)
        elif distance_metric == 'hamming':
            jarak = np.sum(data_train_np[i] != data_train_np, axis=1) / data_train_np.shape[1]
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        jarak[i] = np.inf  # Ignore distance to itself by setting it to infinity

        indeks_terdekat = np.argsort(jarak)[:kk]
        label_terdekat = data_train_y_np[indeks_terdekat]
        valid = np.mean(label_terdekat == data_train_y_np[i])
        validitas.append(valid)

    return np.mean(validitas)

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

def WeighVoting(data_train, data_test, data_train_y, kk, smoothing_regulator, distance_metric, p=3):
    ED = calculate_distances(data_train, data_test, distance_metric, p)
    ED_np = ED.to_numpy()
    validi = validitas(kk, data_train, data_train_y, distance_metric, p)
    validitas_np = np.array(validi)
    data_train_y_np = data_train_y.to_numpy()
    W_np = validitas_np / (ED_np + smoothing_regulator)  # Ensure validitas_np is a scalar or array with appropriate dimension
    W = pd.DataFrame(W_np, columns=[f'W{i + 1}' for i in range(ED_np.shape[1])])
    W['Y'] = data_train_y.tolist()

    Final_predic = []
    for i in range(W_np.shape[1]):
        sorted_indices = np.argsort(-W_np[:, i])
        top_k_indices = sorted_indices[:kk]
        top_k_classes = data_train_y_np[top_k_indices]
        unique, counts = np.unique(top_k_classes, return_counts=True)
        sorted_classes = unique[np.argsort(-counts)]
        Final_predic.append(sorted_classes[0])

    return W, Final_predic

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

def model(X_train_resampled, X_test, y_train_resampled, kk=151, sm=1e-5, jarak='manhattan'):
    _, y_pred_gmknn_sm = WeighVoting(X_train_resampled, X_test, y_train_resampled, kk=kk, smoothing_regulator=sm, distance_metric=jarak)
    cm = confusion_matrix(y_test, y_pred_gmknn_sm)
    accuracy_gmknn_sm = accuracy_score(y_test, y_pred_gmknn_sm)
    akurasi = accuracy_gmknn_sm * 100
    st.write(f'Akurasi: {accuracy_gmknn_sm * 100:.2f}%')   
    st.write("Confusion Matrix Heatmap:")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)
    return akurasi

# Load model
scaler, encoder, X_train_resampled, y_train_resampled = load_model('model.pkl')

# Load model
scaler, encoder, X_train, y_train = load_model('model_no_smote.pkl')

# Menu
menu = ["Home", "Tentang Dataset", "Preprocessing Data", "Model", "Implementasi"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("IMPLEMENTASI ALGORITMA GENETIC MODIFIED K-NEAREST NEIGHBOR DALAM KASUS KLASIFIKASI PENYAKIT HIPERTENSI")
    # st.title("IMPLEMENTASI ALGORITMA GENETIC MODIFIED K-NEAREST NEIGHBOR DALAM KASUS KLASIFIKASI PENYAKIT HIPERTENSI")
    st.subheader("Nama : Nuskhatul Haqqi")
    st.subheader("Nim : 200411100034")

elif choice == "Tentang Dataset":
    st.title("Tentang Dataset")
    # st.title("IMPLEMENTASI ALGORITMA GENETIC MODIFIED K-NEAREST NEIGHBOR DALAM KASUS KLASIFIKASI PENYAKIT HIPERTENSI")
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
    a, b,c,d= st.tabs(['Dataset Asal', 'Clining data', 'Encoder', 'Normalisasi'])
    # Home tab
    with a:
        "### Dataset"
        df_a = pd.read_csv("dataset_fix.csv")
        st.write(df_a)
    with b:
        "### Clining data"
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
    a, b,c,d,e,f= st.tabs(['KNN', 'KK + SMOTE ENN', 'MKNN', 'MKNN + SMOTE ENN', 'GMKNN', 'GMKNN + SMOTE ENN'])
    # Home tab
    with a:
        "### Metode : K Nearest Neighbor (KNN)"
        st.write("Model terbaik KNN: \n K = 7 , Jarak = euclidean ")
        mknn = model(X_train, X_test, y_train, kk=7, sm=1e-5, jarak='euclidean')
    with b:
        "### Metode : K Nearest Neighbor (KNN) + SMOTE ENN"
        st.write("Model terbaik KNN + SMOTE ENN: \n K = 11 , Jarak = minkowski ")
        mknn_sm = model(X_train_resampled, X_test, y_train_resampled, kk=11, sm=1e-5, jarak='euclidean')
    with c:
        "### Metode : Modified K Nearest Neighbor (MKNN)"
        st.write("Model terbaik MKNN: \n K = 7 , Jarak = euclidean, smoothing_regulator = 0.5 ")
        mknn = model(X_train, X_test, y_train, kk=7, sm=1e-5, jarak='euclidean')
    with d:
        "### Metode : Modified K Nearest Neighbor (MKNN) + SMOTE ENN"
        st.write("Model terbaik MKNN + SMOTE ENN: \n K = 11 , Jarak = minkowski, smoothing_regulator = 0.5 ")
        mknn_sm = model(X_train_resampled, X_test, y_train_resampled, kk=11, sm=1e-5, jarak='euclidean')
    with e :
        "### Metode : Genetic Modified K Nearest Neighbor (MKNN)"
        st.write("Model terbaik GMKNN: \n K = 143 , Jarak = mahattan, smoothing_regulator = 0.5 ")
        gmknn = model(X_train, X_test, y_train, kk=143, sm=1e-5, jarak='manhattan')
    with f:
        "### Metode : Genetic Modified K Nearest Neighbor (MKNN) + SMOTE ENN"
        st.write("Model terbaik GMKNN + SMOTE ENN: \n K = 151 , Jarak = mahattan, smoothing_regulator = 0.5 ")
        gmknn_sm = model(X_train_resampled, X_test, y_train_resampled, kk=151, sm=1e-5, jarak='manhattan')



elif choice == "Implementasi":
    st.subheader("Implementasi")
    # Input fields for new data
    umur = st.number_input("Umur Tahun", min_value=0)
    tinggi = st.number_input("Tinggi (cm)", min_value=0.0)
    berat = st.number_input("Berat Badan (kg)", min_value=0.0)
    imt = st.number_input("IMT", min_value=0.0)
    sistole = st.number_input("Sistole (mm)", min_value=0)
    diastole = st.number_input("Diastole (Hg)", min_value=0)
    nafas = st.number_input("Nafas (/menit)", min_value=0)
    detak_nadi = st.number_input("Detak Nadi (/menit)", min_value=0)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["L", "P"])

    if st.button("Submit"):
        # Preprocess input data
        input_data = pd.DataFrame({
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
        _, prediction = WeighVoting(X_train_resampled, input_data, y_train_resampled, kk=7, smoothing_regulator=1e-6, distance_metric='euclidean')

        st.write("Hasil Prediksi Diagnosa:", prediction[0])
