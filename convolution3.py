# convolution3.py

'''
## pythonは3.11です。
streamlit==1.24.0
opencv-python==4.7.0.72
pillow==9.5.0
numpy==1.24.3
'''

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Streamlitアプリの設定
st.title("畳み込みの仕組みを体験しよう！")
st.write("画像にカーネルを適用して、畳み込みの効果を確認できます。")

# 画像のアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください（JPGまたはPNG形式）", type=["jpg", "png"])

# カーネルの選択
st.sidebar.title("カーネルの選択")
kernel_options = {
    "エッジ検出": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    "ぼかし（平滑化）": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    "シャープ化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}
kernel_name = st.sidebar.selectbox("カーネルを選んでください", list(kernel_options.keys()))
kernel = kernel_options[kernel_name]

# カーネルのカスタマイズ
st.sidebar.write("カスタムカーネルを有効にするには値を編集してください。")
custom_kernel_enabled = st.sidebar.checkbox("カスタムカーネルを使用")
if custom_kernel_enabled:
    kernel = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            kernel[i, j] = st.sidebar.number_input(f"カーネル[{i},{j}]", value=float(kernel[i, j]))

# 画像処理と結果表示
if uploaded_file is not None:
    # 入力画像を読み込み
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # RGB形式に変換
    image_array = np.array(image)

    # 畳み込みを適用
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    convolved_image = cv2.filter2D(gray_image, -1, kernel)

    # 結果を1つのボックスにまとめて表示
    st.image([image, convolved_image], caption=["元の画像", f"カーネル適用後 ({kernel_name})"], use_column_width=True)

    # サイドバーにカーネルを表示
    st.sidebar.subheader("現在のカーネル")
    st.sidebar.table(kernel)
else:
    st.warning("画像をアップロードしてください。")
