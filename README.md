# PE Predictor - Dự đoán PE cổ phiếu sử dụng SVR

## Mô tả
Module Python để dự đoán PE (Price-to-Earnings ratio) của cổ phiếu trong tương lai sử dụng Support Vector Regression (SVR). Module có thể dự đoán PE cho bất kỳ khoảng thời gian nào (ví dụ: 3, 5 năm) và tính toán PE trung bình dự đoán.

## Tính năng chính
- 🔮 Dự đoán PE cho n năm tới (có thể tùy chỉnh)
- 📊 Tính toán PE trung bình trong khoảng thời gian dự đoán
- 🎯 Tối ưu hóa tham số mô hình tự động
- 📈 Vẽ biểu đồ trực quan hóa kết quả
- 📋 Thống kê chi tiết về dự đoán
- 🔧 Xử lý dữ liệu thời gian định dạng "YYYY Qx"

## Cài đặt

### 1. Clone repository hoặc tải về các file
```bash
git clone <repository-url>
cd pe-lnst
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Cấu trúc file
```
pe-lnst/
├── pe_predictor.py          # Module chính
├── VCB.csv                  # Dữ liệu mẫu (VCB Bank)
├── requirements.txt         # Dependencies
└── README.md               # Hướng dẫn này
```

## Cách sử dụng

### 1. Sử dụng cơ bản
```python
from pe_predictor import PEPredictor

# Khởi tạo predictor
predictor = PEPredictor(kernel='rbf', optimize_params=True)

# Tải dữ liệu
predictor.load_data('VCB.csv')

# Chuẩn bị dữ liệu
predictor.prepare_data()

# Huấn luyện mô hình
predictor.train_model()

# Dự đoán PE 3 năm tới
results = predictor.predict_future_pe(years=3)

print(f"PE trung bình 3 năm tới: {results['average_pe']:.2f}")
```

### 2. Chạy demo đầy đủ
```bash
python demo_pe_predictor.py
```

### 3. Dự đoán cho các khoảng thời gian khác nhau
```python
# Dự đoán 3 năm
results_3y = predictor.predict_future_pe(years=3)

```

## Định dạng dữ liệu đầu vào

File CSV cần có 2 cột:
- `thoigian`: Thời gian theo định dạng "YYYY Qx" (ví dụ: "2024 Q1")
- `pe`: Giá trị PE ratio

Ví dụ:
```csv
thoigian,pe
2024 Q1,15.61
2023 Q4,12.47
2023 Q3,12.74
...
```

## Kết quả trả về

Hàm `predict_future_pe()` trả về dictionary chứa:

```python
{
    'predictions': [
        {
            'period': '2024 Q3',
            'time_numeric': 2024.5,
            'predicted_pe': 14.25
        },
        # ... more predictions
    ],
    'average_pe': 14.85,          # PE trung bình dự đoán
    'years_predicted': 10,        # Số năm được dự đoán
    'total_periods': 40          # Tổng số kỳ dự đoán
}
```

## Tùy chỉnh mô hình

### 1. Thay đổi kernel
```python
# Sử dụng kernel khác
predictor = PEPredictor(kernel='linear')  # 'rbf', 'linear', 'poly'
```

### 2. Tắt tối ưu hóa tham số (chạy nhanh hơn)
```python
predictor = PEPredictor(optimize_params=False)
```

### 3. Vẽ biểu đồ kết quả
```python
# Vẽ biểu đồ với vùng tin cậy
predictor.plot_predictions(results, show_confidence=True)
```

## Ví dụ thực tế

### Dự đoán PE của VCB trong 5 năm tới:
```python
from pe_predictor import PEPredictor

# Khởi tạo và huấn luyện
predictor = PEPredictor()
predictor.load_data('VCB.csv')
predictor.prepare_data()
predictor.train_model()

# Dự đoán 5 năm
results = predictor.predict_future_pe(years=5)

print(f"PE trung bình VCB trong 5 năm tới: {results['average_pe']:.2f}")
print(f"Khoảng dự đoán: {results['min_predicted_pe']:.2f} - {results['max_predicted_pe']:.2f}")

# Hiển thị chi tiết 4 quý đầu tiên
for pred in results['predictions'][:4]:
    print(f"{pred['period']}: PE = {pred['predicted_pe']:.2f}")
```

Output:
```
PE trung bình VCB trong 5 năm tới: 14.85
Khoảng dự đoán: 12.45 - 17.20
2024 Q3: PE = 14.25
2024 Q4: PE = 13.89
2025 Q1: PE = 15.12
2025 Q2: PE = 14.67
```

## Thông tin kỹ thuật

### Dependencies chính:
- `numpy`: Tính toán số học
- `pandas`: Xử lý dữ liệu
- `matplotlib`: Vẽ biểu đồ
- `scikit-learn`: Machine learning (SVR)
- `seaborn`: Cải thiện biểu đồ

### Thuật toán:
- **Support Vector Regression (SVR)** với kernel RBF (mặc định)
- **Grid Search** để tối ưu hóa tham số
- **StandardScaler** để chuẩn hóa dữ liệu

### Hiệu suất:
- Phù hợp với dữ liệu từ 10-100 mẫu
- Thời gian huấn luyện: < 30 giây (với tối ưu hóa)
- Độ chính xác: Phụ thuộc vào chất lượng dữ liệu lịch sử

## Lưu ý quan trọng

1. **Chất lượng dữ liệu**: Kết quả dự đoán phụ thuộc nhiều vào chất lượng và độ dài chuỗi dữ liệu lịch sử
2. **Giới hạn dự đoán**: Dự đoán dài hạn (>10 năm) có thể kém chính xác
3. **Kiểm tra kết quả**: Luôn kiểm tra và phân tích kết quả trước khi sử dụng cho quyết định đầu tư
4. **Rủi ro**: Đây chỉ là công cụ hỗ trợ, không thay thế cho phân tích đầu tư chuyên sâu

## Troubleshooting

### Lỗi thường gặp:

1. **"Không thể phân tích thời gian"**: Kiểm tra định dạng cột `thoigian` phải là "YYYY Qx"
2. **"Chưa tải dữ liệu"**: Gọi `load_data()` trước khi sử dụng các hàm khác
3. **"Mô hình chưa được huấn luyện"**: Gọi `train_model()` trước khi dự đoán

### Tối ưu hóa hiệu suất:
- Sử dụng `optimize_params=False` nếu muốn chạy nhanh
- Giảm số năm dự đoán nếu cần kết quả nhanh
- Sử dụng kernel 'linear' cho dữ liệu đơn giản

