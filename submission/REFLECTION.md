# Reflection — Lab 21

**Ngô Thế Khanh — 2A202602503 — 07/10/2026**

## 1. Điều làm tôi ngạc nhiên

Fine-tune tăng target từ 0.765 lên 0.960 nhưng verdict vẫn FAILED do regression giảm quá ngưỡng. Kết quả này giúp tôi phân biệt cải thiện một nhiệm vụ với cải thiện tổng thể. Attention-only khi khớp ngân sách cũng đạt 0.970, cao hơn correct 0.960, nên không thể mặc định cấu hình mang tên correct sẽ đứng đầu mọi phép đo.

## 2. Phần tôi phải chờ và khó khăn thao tác

Tôi phải chờ quá trình train và đánh giá trên Colab, sau đó bổ sung bước sinh câu trả lời để so sánh từng ticket. Khi phiên Colab mất thư mục làm việc, tôi đã lo phải chạy lại toàn bộ; ZIP lưu trên máy giúp khôi phục báo cáo, kết quả và adapter. Trong lượt chạy được ghi log, NB5 mất 627 giây, là stage dài nhất; tổng lượt đó là 1945 giây và có tái sử dụng hai adapter ở NB4.

## 3. Điều tôi hiểu rõ hơn sau bài

Tôi đã hỏi liệu bài có chỉ cần viết báo cáo; sau khi thực hiện, tôi hiểu báo cáo phải có bằng chứng từ thí nghiệm. Loss giảm hoặc unit test đạt không thay thế đánh giá target, regression, format và latency. Cần so với baseline đã prompt tốt và đọc cổng hồi quy trước khi kết luận nên triển khai.

## 4. Tôi dùng AI assistant vào việc gì và chỗ nào cần kiểm tra

AI hỗ trợ giải thích rubric, hướng dẫn chạy Colab, đọc log, chuẩn bị script so sánh, tổng hợp báo cáo và kiểm tra ZIP. Assistant ban đầu bỏ sót EVAL_LIMIT=8; khi nhìn ảnh cấu hình mới chỉ ra đây là smoke mode và hướng dẫn chạy toàn bộ tập. Tôi vì vậy cần kiểm tra số mẫu và cấu hình trong output thực tế. Các lượt bổ sung được ghi đúng nhóm: target có một ca FT thua, còn regression có ba ca thua theo keyword recall. Tôi cũng lưu ý mẫu trái cây cho thấy so chuỗi bỏ dấu có thể cho điểm dù nội dung chưa đúng.

## 5. Nếu làm cho khách hàng thật

Tôi sẽ xác định yêu cầu chất lượng của bài toán, xây tập đánh giá và đo baseline với prompt tối ưu trước khi quyết định fine-tune. Tôi sẽ đặt ngưỡng regression, format và latency từ yêu cầu vận hành, lưu cấu hình cùng artifacts và kiểm tra lỗi từng mẫu. Kết quả FAILED ở lab này cho thấy chưa nên dùng adapter correct như một model đa nhiệm đã vượt kiểm tra chất lượng.
