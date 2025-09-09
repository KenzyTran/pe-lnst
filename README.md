# README — Dự đoán P/E (TTM) bằng SVR (DEMO)

**Phạm vi**: Đây là bản *demo* chỉ chạy **cho 1 mã** dưới dạng **file CSV**. Gói gọn trong phạm vi này.

## Files

* `support_vector_regression.ipynb` — notebook tham khảo.
* `pe_predictor.py` — script demo.
* `requirements.txt` — dependency.

## Input (Single CSV)

* File CSV duy nhất cho 1 mã.
* Columns bắt buộc: `thoigian` (ví dụ `2019-Q1`), `pe` (số).
* Chỉ sử dụng dữ liệu P/E trong **vòng 5 năm gần nhất** của mã VCB.

## Process (tóm tắt)

1. Đọc file và chuẩn hoá thứ tự theo `thoigian`.
2. Lấy chỉ các điểm thuộc 5 năm gần nhất.
3. Nếu số điểm >= 4: tạo X = 0..N-1 → scale → fit SVR → predict 12 quý tiếp theo.
4. Xuất kết dự đoán ra màn hình.

## Run (ví dụ)

```bash
python pe_predictor.py --input data/VCB.csv
```

## Notes ngắn

* Không thực hiện imputation dữ liệu trong demo.
* Bắt lỗi khi train/predict và log rõ lý do.
* Đối với những mã lọc dữ liệu trong 5 năm mà **không đủ 4 điểm hợp lệ** → skip (không tính).

