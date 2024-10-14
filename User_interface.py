import numpy as np
import streamlit as st
import joblib as jb
import time

# Các biến liên quan 
features = ['fixed acidity', 'chlorides', 'free sulfur dioxide', 'density',
            'pH', 'sulphates', 'sulfur_ratio', 'not_free_sulfur',
            'fixed_volatile_acidity_interaction', 'citric_residual_sugar_interaction',
            'alcohol_bins']

# tải model và scaler lên giao diện 
model = jb.load("wine_quality_model.pkl")
scaler = jb.load("scaler.pkl")

st.title("MÔ HÌNH DỰ ĐOÁN CHẤT LƯỢNG RƯỢU")

st.divider()

# chức năng nhập thông tin mới 
user_input = {}
for feature in features:
    if feature == 'alcohol_bins':
        user_input[feature] = st.selectbox(
            f"Chọn nồng độ cồn {feature.replace('_', ' ')}",
            options=['rất thấp : 0', 'thấp : 1', 'trung bình : 2', 'cao : 3', 'rất cao : 4']
        )
    else:
        user_input[feature] = st.number_input(
            f"Nhập giá trị cho {feature.replace('_', ' ')}",
            step=0.01,
            format="%.2f"
        )

st.divider()

predict_button = st.button("DỰ ĐOÁN")
st.divider()

# chức năng dự đoán 
if predict_button:
    with st.spinner("Đang phân tích mẫu rượu..."):
        time.sleep(1.5)  # giả lập tgian xử lý dữ liệu 

        # gán nhãn cho biến 'alcohol' và 'quality'
        alcohol_mapping = {'rất thấp : 0': 0, 'thấp : 1': 1, 'trung bình : 2': 2, 'cao : 3': 3, 'rất cao : 4': 4}
        X = [user_input[feature] if feature != 'alcohol_bins' else alcohol_mapping[user_input[feature]] for feature in features]
        X_arr = np.array(X).reshape(1, -1)
        X_scaled = scaler.transform(X_arr)
        prediction = model.predict(X_scaled)[0] # lấy giá trị đầu tiên 

        quality_mapping = {0: 'tệ', 1: 'trung bình', 2: 'tốt'}
        result = quality_mapping[prediction]

        st.success(f"Kết quả dự đoán chất lượng rượu: {result.upper()}")

        st.info("Thông tin chi tiết:")
        st.write(f"- Giá trị số: {prediction}")
        st.write(f"- Ý nghĩa: 0 (tệ), 1 (trung bình), 2 (tốt)")
# streamlit run User_interface.py