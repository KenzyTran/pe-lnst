# -*- coding: utf-8 -*-
"""
Sử dụng đơn giản PE Predictor - chỉ dự đoán và trả kết quả
"""

from pe_predictor import PEPredictor

def predict_pe(years):
    """
    Dự đoán PE cho n năm tới
    
    Args:
        years (int): Số năm cần dự đoán
        
    Returns:
        float: PE trung bình dự đoán
    """
    # Khởi tạo predictor
    predictor = PEPredictor()
    
    # Tải và chuẩn bị dữ liệu
    predictor.load_data('VCB.csv')
    predictor.prepare_data()
    
    # Huấn luyện mô hình
    predictor.train_model()
    
    # Dự đoán
    results = predictor.predict_future_pe(years=years)
    
    return results['average_pe']

# Ví dụ sử dụng
if __name__ == "__main__":
    
    # Dự đoán 3 năm
    pe_3_years = predict_pe(3)
    print(f"PE trung bình dự đoán 3 năm tới: {pe_3_years:.2f}")
