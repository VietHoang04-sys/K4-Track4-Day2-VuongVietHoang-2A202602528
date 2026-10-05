# Lab Day 2 — DeepWeeds

> This report is a submission template. The Colab notebook fills measured metrics and tables after the experiments complete. Do not replace missing results with estimates or reference-paper numbers.

## 1. Tóm tắt

- Dataset: DeepWeeds, fold 0.
- Cấu hình tốt nhất theo validation: **chưa chạy**.
- Kết quả test cuối (mean ± std, tối thiểu 3 seed): **chưa chạy**.
- Kết luận chính: **chưa chạy**.

## 2. Dữ liệu và thiết lập

Ghi số lượng từng split/lớp, kiểm tra giao rỗng, phiên bản thư viện, GPU, seed, công thức nền và cách chọn checkpoint. Chỉ dùng validation để chọn; test chỉ chạy một lần cho mỗi seed ở giai đoạn chung kết.

## 3. So sánh backbone

Chèn bảng `Backbones` trong `results.xlsx`. Nhận xét macro-F1 validation cùng tham số, GMAC, thời gian train và độ trễ đo thật.

## 4. Công thức huấn luyện

Chèn bảng `Training`. Nêu rõ mỗi thí nghiệm khác `T00` ở một yếu tố nào; báo cáo Δ và không khẳng định chênh lệch nhỏ hơn độ nhiễu là cải thiện.

## 5. Phương pháp suy luận

Chèn bảng `Inference` và `Latency`. Báo cáo macro-F1, top-1, ECE và p50/p95/p99 đo bằng warmup, đồng bộ GPU và ít nhất 50 lần lặp.

## 6. Cấu hình cuối và phân tích lỗi

Chèn kết quả `Final`, `PerClass` và ma trận nhầm lẫn từ kết quả đã chạy. So sánh với baseline cùng số seed; xem các trường hợp nhầm lẫn, đặc biệt Chinee Apple ↔ Snake Weed.

## 7. Kết luận, khuyến nghị và hạn chế

Trả lời cấu hình nào tốt nhất, backbone/recipe/inference đóng góp thế nào, và cấu hình nào phù hợp triển khai theo ngân sách độ trễ. Nêu giới hạn một fold, số seed, dữ liệu chia ngẫu nhiên không theo địa điểm và các thí nghiệm chưa thực hiện.

## Phụ lục

Liệt kê `exp_id`, cấu hình đầy đủ, link notebook Colab và link chạy lại được.
