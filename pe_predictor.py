# -*- coding: utf-8 -*-
"""
PE Predictor Module using Support Vector Regression (SVR)
Dự đoán PE của cổ phiếu trong n năm tới và tính trung bình
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import warnings
from datetime import datetime, timedelta
import re

warnings.filterwarnings('ignore')

class PEPredictor:
    """
    Lớp dự đoán PE sử dụng Support Vector Regression
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1):
        """
        Khởi tạo PE Predictor
        
        Args:
            kernel (str): Loại kernel cho SVR ('rbf', 'linear', 'poly')
            C (float): Tham số regularization
            gamma (str or float): Tham số kernel coefficient
            epsilon (float): Epsilon trong epsilon-SVR model
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.regressor = None
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.data = None
        self.X_train = None
        self.y_train = None
        self.is_trained = False
        
    def _parse_time_column(self, time_str):
        """
        Chuyển đổi chuỗi thời gian từ format 'YYYY Qx' thành số thứ tự
        
        Args:
            time_str (str): Chuỗi thời gian dạng '2024 Q1'
            
        Returns:
            float: Số thứ tự tương ứng với thời gian
        """
        match = re.match(r'(\d{4})\s*Q(\d)', time_str)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            # Chuyển đổi thành số thứ tự: năm + (quý-1)/4
            return year + (quarter - 1) / 4
        else:
            raise ValueError(f"Không thể phân tích thời gian: {time_str}")
    
    def load_data(self, csv_file_path):
        """
        Tải dữ liệu từ file CSV
        
        Args:
            csv_file_path (str): Đường dẫn đến file CSV
        """
        try:
            self.data = pd.read_csv(csv_file_path)
            
            # Kiểm tra cột cần thiết
            required_columns = ['thoigian', 'pe']
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Không tìm thấy cột '{col}' trong dữ liệu")
            
            # Chuyển đổi cột thời gian
            self.data['time_numeric'] = self.data['thoigian'].apply(self._parse_time_column)
            
            # Sắp xếp theo thời gian
            self.data = self.data.sort_values('time_numeric').reset_index(drop=True)
            
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {str(e)}")
            raise
    
    def prepare_data(self):
        """
        Chuẩn bị dữ liệu cho training
        """
        if self.data is None:
            raise ValueError("Chưa tải dữ liệu. Hãy gọi load_data() trước.")
        
        # Chuẩn bị X (thời gian) và y (PE)
        X = self.data[['time_numeric']].values
        y = self.data['pe'].values.reshape(-1, 1)
        
        # Feature scaling
        self.X_train = self.sc_X.fit_transform(X)
        self.y_train = self.sc_y.fit_transform(y)
        
    
    def train_model(self):
        """
        Huấn luyện mô hình SVR
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
        
        # Tạo và huấn luyện SVR với tham số đã chỉ định
        self.regressor = SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            epsilon=self.epsilon
        )
        self.regressor.fit(self.X_train, self.y_train.ravel())
        
        self.is_trained = True
        print(f"Mô hình đã được huấn luyện thành công!")
    
    def predict_future_pe(self, years=3):
        """
        Dự đoán PE trong n năm tới
        
        Args:
            years (int): Số năm cần dự đoán
            quarters_per_year (int): Số quý mỗi năm (mặc định 4)
            
        Returns:
            dict: Kết quả dự đoán bao gồm các giá trị PE và trung bình
        """
        if not self.is_trained:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi train_model() trước.")
        
        # Lấy thời gian cuối cùng trong dữ liệu
        last_time = self.data['time_numeric'].max()
        
        # Tạo các thời điểm tương lai
        future_times = []
        future_periods = []
        
        for i in range(1, years * 4 + 1):
            future_time = last_time + i / 4
            future_times.append(future_time)
            
            # Tạo nhãn thời gian
            year = int(future_time)
            quarter = int((future_time - year) * 4) + 1
            future_periods.append(f"{year} Q{quarter}")
        
        # Chuẩn hóa thời gian tương lai
        future_times_array = np.array(future_times).reshape(-1, 1)
        future_times_scaled = self.sc_X.transform(future_times_array)
        
        # Dự đoán
        future_pe_scaled = self.regressor.predict(future_times_scaled)
        future_pe = self.sc_y.inverse_transform(future_pe_scaled.reshape(-1, 1)).flatten()
        
        # Tính trung bình
        average_pe = np.mean(future_pe)
        
        # Tạo kết quả
        results = {
            'predictions': [
                {
                    'period': period,
                    'time_numeric': time,
                    'predicted_pe': pe
                }
                for period, time, pe in zip(future_periods, future_times, future_pe)
            ],
            'average_pe': average_pe,
            'years_predicted': years
        }
        
        return results



def main():
    """
    Hàm chính để demo module
    """
    # Tạo instance của PE Predictor
    predictor = PEPredictor(kernel='rbf')
    
    try:
        # Tải và chuẩn bị dữ liệu
        predictor.load_data('VCB.csv')
        predictor.prepare_data()
        
        # Huấn luyện mô hình
        predictor.train_model()
        
        # Dự đoán 3 năm tới
        results_3y = predictor.predict_future_pe(years=3)
        print(f"PE trung bình dự đoán 3 năm tới: {results_3y['average_pe']:.2f}")
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()

