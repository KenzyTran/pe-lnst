import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Đọc và chuẩn bị dữ liệu
dataset = pd.read_csv('VCB.csv')
dataset = dataset.sort_values('thoigian')
X = np.arange(len(dataset)).reshape(-1, 1)
y = dataset['pe'].values

# Chuẩn hóa dữ liệu
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)

# Huấn luyện mô hình SVR
model = SVR(kernel='rbf')
model.fit(X_scaled, y)

# Dự đoán P/E cho 12 quý tiếp theo
future_quarters_index = np.arange(len(dataset), len(dataset) + 12).reshape(-1, 1)
future_quarters_scaled = sc_X.transform(future_quarters_index)
future_pe = model.predict(future_quarters_scaled)

# Lưu kết quả
results = {f"Quý {i}": round(pe_pred, 2) for i, pe_pred in enumerate(future_pe, 1)}