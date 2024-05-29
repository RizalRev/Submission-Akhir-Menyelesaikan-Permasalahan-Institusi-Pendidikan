import streamlit as st 
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Fungsi untuk preprocessing dataset
def dataset_preprocessing(main_df):
    kolom_dihapus_df = ['Application_mode', 'Application_order', 'Mothers_qualification', 'Fathers_qualification', 'Gender',
                    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate', 'Inflation_rate', 'GDP']
    dataset_ml = main_df.drop(kolom_dihapus_df, axis=1)

    fitur_normalisasi, fitur_encoding = [], []

    for fitur in dataset_ml:
        if dataset_ml[fitur].dtype == "object":
            fitur_encoding.append(fitur)
        else:
            fitur_normalisasi.append(fitur)

    # Melakukan Label Encoder pada fitur
    LE = LabelEncoder() #Mendefenisikan LabelEncoder sebagai LE
    dataset_ml_main = dataset_ml.copy() #Mencegah SettingWithCopyWarning pandas

    for col in fitur_encoding:
        dataset_ml_main[col] = LE.fit_transform(dataset_ml_main[col])

        # mapping encoder
        label_mapping = {index: label for index, label in enumerate(LE.classes_)}
        print(f'Label mapping for {col}:', label_mapping)

    scaler = MinMaxScaler()  #Mendefinisikan MinMaxScaler
    dataset_ml_main[fitur_normalisasi] = scaler.fit_transform(dataset_ml_main[fitur_normalisasi]) #Menerapkan fit_transform untuk normalisasi fitur terpilih

    dataset_ml_main.drop("Status", axis=1, inplace=True)

    return dataset_ml_main


# Fungsi untuk prediksi attrition
def predict_attrition(fix_main_df):

    best_model = joblib.load("random_forest.joblib")

    prediction = best_model.predict(fix_main_df)

    return prediction


# Fungsi untuk menampilkan hasil prediksi
def result_attrition(main_df, prediction_array):

    result_df_1 = main_df[["Marital_status", "Previous_qualification", "Nacionality", "Gender", "Age_at_enrollment"]]
    result_df_2 = pd.DataFrame(data=prediction_array)
    result_df = pd.merge(result_df_1,result_df_2, how='left', left_index = True, right_index = True)
    result_df.columns = ["Status Perkawinan", "Pendidikan Terakhir", "Kewarganegaraan", "Jenis Kelamin", "Umur (saat mendaftar)", "Prediksi Status Siswa"]

    result_df_fix = result_df.copy()
    result_df_fix["Prediksi Status Siswa"] = result_df_fix["Prediksi Status Siswa"].apply(lambda x: "Keluar" if x == 0 else "Aktif" if x == 1 else "Lulus")

    return result_df_fix



# Fungsi utama website  
def main():

    st.title('Jaya Jaya Institut')

    st.header('Dasbor Siswa')

    st.text('Halo, Selamat datang di halaman dasbor prediksi status siswa di Jaya Jaya Institut')

    with st.expander("Kenalan Dulu dengan Jaya Jaya Institut ..."):
        st.write(
            """
                Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. 
                Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik.
            """
        )

    with st.expander("Mengapa dasbor ini dibuat ya?"):
        st.write(
            """
                Permasalahan pada Jaya Jaya Institut adalah terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout. 
                Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. 
                Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.
            """
        )

    with st.expander("Bagaimana sih cara menggunakan dasbor ini?"):
        st.write(
            """
                Sederhana kok, kamu bisa ngikutin tahapan-tahapan dibawah ini:
                1. Masukan dulu dataset yang sesuai pada menu "Masukan Dataset" dibawah. Dataset-nya dimana? Bisa didownload dari github/hasil output notebook.ipynb dengan nama file "df_student_for_ML.csv"
                2. Kalo datasetnya sudah muncul, mantapss...
                3. Tahan duiu, kamu bisa set berapa banyak data yang mau di prediksi melalui menu slider ya...
                4. Yuk lanjut!!! kalau udah cocok berapa banyak datanya, gaskun klik tombol "Prediksi Status Siswa"
                5. Eits, yang sabar ya tunggu proses prediksi-nya...
                6. Kalau sudah muncul hasil prediksinya, yeeey Selamat ya Kamu Berhasil...
            """
        )

    st.write("##")

    uploaded_files = st.file_uploader("Masukan Dataset", accept_multiple_files=True, key="fileuploader1")
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("filename:", uploaded_file.name)
        try:
            row_value_df = st.slider('Tampilkan Data', 0, df.shape[0], 5) 
            df = df.iloc[:row_value_df]
            st.dataframe(df)
        except NameError:
            st.write("")

    # when 'Predict' is clicked, make the prediction and store it
    try:
        if uploaded_files is not None:
            try:
                if st.button("Prediksi Status siswa"): 
                    main_new_dataset = dataset_preprocessing(df) 
                    
                    result = predict_attrition(main_new_dataset)

                    result_dataset = result_attrition(df, result)

                    st.success(f"Hasil Prediksi Berhasil untuk {result_dataset.shape[0]} siswa Jaya Jaya Institut")
                    row_value_result_dataset = st.slider('Tampilkan Data Hasil Prediksi', 0, result_dataset.shape[0], result_dataset.shape[0])
                    result_dataset = result_dataset.iloc[:row_value_result_dataset]
                    st.dataframe(result_dataset)
                    
            except ValueError:
                    st.error('Dataset Kosong! Pastikan Dataset memiliki isi sebelum di prediksi', icon="🔥")
                    
    except UnboundLocalError:
        st.error('Anda belum menginput dataset', icon="🚨")

        


if __name__=='__main__': 
    main()


st.caption('Copyright (c) 2024')