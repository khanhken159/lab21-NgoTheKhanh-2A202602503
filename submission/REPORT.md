# Lab 21 — LoRA cải thiện phân loại ticket nhưng chưa vượt cổng hồi quy

**Họ tên:** Ngô Thế Khanh

**MSSV:** 2A202602503

**Ngày viết báo cáo:** 07/10/2026

**Commit chạy trên Colab:** `d27c1c0` (theo output Setup đã cung cấp)

**Model:** `unsloth/Qwen3.5-4B` · **Tier:** T4

**GPU thực tế:** Tesla T4, 14.6 GB VRAM được runtime nhận diện · **Precision:** fp16

## 1. Bài toán, dữ liệu và thiết kế

Thí nghiệm phân loại ticket chăm sóc khách hàng tiếng Việt thành JSON với bốn trường: `intent`, `urgency`, `product`, `sentiment`. Sử dụng corpus mặc định gồm 250 mẫu vì bài toán có nhãn để chấm khách quan và kiểm tra được cả tính đúng của nội dung lẫn định dạng. Model được chọn theo cấu hình T4 mặc định của lab để chạy trên Colab và dùng cùng base model cho baseline và adapter. Không thay model giữa bước đo baseline và bước đánh giá fine-tune.

| Thành phần | Cấu hình hoặc số đo |
|---|---|
| Train / validation | 225 / 25 mẫu, seed 42 |
| Target evaluation | 50 mẫu |
| Regression evaluation | 15 mẫu |
| Chế độ đánh giá | Toàn bộ tập; `eval_limit=null`, `smoke_mode=false` |
| Mask | `assistant-only` |
| Độ dài trung bình / p95 / tối đa | 93.1 / 98 / 101 token |
| Độ dài gợi ý bởi NB1 | 256 token |
| `max_length` thực tế | 1024 token |
| Epochs của correct | 2 |
| Optimizer steps của correct | 30 |
| Batch mỗi thiết bị / gradient accumulation | 1 / 16, batch hiệu dụng 16 |
| Scheduler / warmup | cosine / 3 bước |
| Gradient checkpointing / packing | bật / tắt |

Giữ `max_length=1024` theo tier T4 để chạy cấu hình mặc định, mặc dù NB1 gợi ý 256 từ thống kê độ dài. Đây là lựa chọn bảo thủ về giới hạn chiều dài, không phải bằng chứng rằng dữ liệu cần đến 1024 token. Với mẫu dài nhất chỉ 101 token, không thấy nhu cầu cắt câu trả lời ở giới hạn này; lượt tối ưu tiếp theo nên thử 256 với cấu hình còn lại cố định và đo lại bộ nhớ, tốc độ. Không suy ra mức padding thực tế chỉ từ giới hạn `max_length`.

T4 được chạy ở fp16, với `bf16=False`. Log ghi nhận phần cứng này không hỗ trợ bfloat16 như GPU Ampere. Model có 32 hidden layers, gồm 24 lớp linear attention và 8 lớp full attention. Phần text-linear nhắm vào 12 tên module, gồm các projection của attention và MLP; không dùng nhãn “all-linear” để mặc nhiên suy ra adapter đã gắn vào mọi thành phần của model.

## 2. Bằng chứng mask và chat template

| Kiểm tra mẫu NB1 | Kết quả |
|---|---|
| Tổng token | 94 |
| Token được tính loss | 39 |
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi được che khỏi loss | `true` |

Đoạn được tính loss trong output thực tế:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Hai assert xác nhận câu trả lời được giám sát và câu hỏi không nằm trong phần loss. Trên toàn bộ 225 mẫu train ở NB3, có 9014/20951 token được giám sát, khoảng 43.0%. Ngược lại, thử `everything` đưa cả 94/94 token của mẫu NB1 vào loss, bao gồm câu hỏi và prompt. Vì vậy, giảm loss trong chế độ đó không chứng minh model đã học đúng nhiệm vụ chỉ sinh câu trả lời.

Kiểm tra template trả về `reasoning preserved — safe to train on traces`: nội dung suy luận thử nghiệm trong `<think>...</think>` vẫn xuất hiện sau render. Tuy nhiên, câu trả lời của corpus mặc định là JSON, không chứa trace suy luận có nội dung. `valid_trace_rate=0.0` ở NB5 do đó không đủ để kết luận model bị reasoning-trace collapse. Output còn cho thấy token đóng `</think>` nằm trong phần được giám sát; cần đọc chuỗi render thực tế thay vì mặc định phần loss chỉ gồm JSON thuần.

## 3. Baseline được đóng băng trước lượt train correct

NB2 đo hai baseline trước NB3 trong lượt chạy được cung cấp. Baseline (a) dùng prompt đơn giản; baseline (b) dùng prompt tối ưu có schema và ví dụ. SHA rút gọn của prompt (b) được lưu là `719e74d3b6232053`. Báo cáo dùng đúng mốc đã đo, không hạ điểm baseline hoặc sửa ngưỡng sau khi nhìn kết quả fine-tune.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3260.0 |
| (b) Base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1027.6 |
| (c) LoRA correct + naive prompt | 0.9600 | 0.7444 | 1.0000 | 1364.0 |

Target là độ chính xác trung bình của các trường, không phải tỷ lệ ticket đúng toàn bộ bốn trường. Regression được chấm bằng keyword recall trên câu hỏi phổ thông; format là tỷ lệ JSON đáp ứng kiểm tra khóa của bộ chấm. Latency là số đo của hàm sinh batch trong lab, không phải cam kết độ trễ của hệ thống phục vụ từng người dùng.

Baseline (b) thực sự mạnh hơn (a): target tăng 0.765 và format tăng từ 0 lên 1. Log không ghi nhận việc chỉnh sửa prompt. Chỉ đổi cách mô tả nhiệm vụ đã tạo một mốc cạnh tranh rõ ràng mà chưa cần train. Fine-tune tiếp tục tăng target thêm 0.195, tức 19.5 điểm phần trăm, giữ format 100%. Tuy nhiên, latency của correct cao hơn baseline (b) khoảng 32.7%; prompt ngắn hơn không đồng nghĩa với suy luận nhanh hơn trong phép đo này.

## 4. Đối chứng vị trí, learning rate và lượng tử hóa

| Run | Vị trí | r | Trainable params | LR | Training loss được báo cáo | Target NB5 | Train (s) | Peak VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32464896 | 0.0001 | 0.6312 | 0.9600 | 391.6 | 8.78 |
| attn_only | q,v | 283 | 32456704 | 0.0001 | 0.5373 | 0.9700 | 268.3 | 8.79 |
| wrong_lr | text-linear | 16 | 32464896 | 0.00001 | 1.5702 | 0.0000 | 402.0 | 8.78 |
| qlora | text-linear | 16 | 32464896 | 0.0001 | 0.7068 | 0.9400 | 460.6 | 3.85 |

**Kiểm tra tính công bằng:** output gatekeeper và `runs.csv` người thực hiện cung cấp xác nhận correct, attn_only, wrong_lr và qlora đều dùng 30 optimizer steps, cùng tier T4 và base model. Ngân sách tham số đối chứng cũng được gatekeeper xác nhận khớp. NB4 tái sử dụng attn_only và wrong_lr đã train trong phiên trước; thời gian NB4 của lượt hiện tại không bao gồm train mới hai adapter này. CSV có hai dòng correct: lượt cũ ghi loss 0.6268, thời gian 412.9 giây; lượt mới ghi loss 0.6312, thời gian 391.6 giây. Bảng báo cáo dùng dòng correct mới nhất, khớp với log NB3 ngay trước NB5. CSV không có timestamp để kiểm chứng độc lập thứ tự đóng băng baseline so với các lượt đối chứng cũ; báo cáo không khẳng định điều vượt quá bằng chứng hiện có.

### 4.1. Vị trí so với rank

Hai ngân sách tham số chênh 8192, khoảng 0.0252%, nhỏ hơn ngưỡng 5% của rubric. Nếu giữ q/v ở r=16 thì chỉ có 1835008 tham số, không tương đương với 32464896 tham số của correct; dùng r=283 tránh phép so sánh ngân sách không công bằng đó. Theo bảng NB5, attn_only đạt 0.97, cao hơn correct 0.96 một điểm phần trăm. Training loss cũng thấp hơn, 0.5373 so với 0.6312, nên thứ tự hai run nhất quán trong lượt đo này.

Kết quả không ủng hộ khẳng định text-linear luôn tốt hơn attention-only trên tác vụ hẹp này. Nó cũng không chứng minh rank cao tự nó là nguyên nhân: rank của đối chứng được thay để giữ ngân sách khi đổi vị trí. Chênh lệch một điểm phần trăm trên 50 mẫu tương ứng hai trường đúng thêm trong tổng 200 trường, chưa có lặp nhiều seed hoặc khoảng tin cậy để khẳng định ưu thế ổn định. Chưa đo regression của attn_only, nên không thể kết luận đây là adapter nên triển khai chỉ từ điểm target cao nhất.

### 4.2. Learning rate

wrong_lr dùng LR=1e-5, thấp hơn correct mười lần, với cùng vị trí và rank theo bảng cấu hình. Training loss được báo cáo là 1.5702, cao hơn correct 0.6312; đồng thời target và format ở NB5 đều bằng 0. Với correct, các log loss theo đoạn giảm từ 2.163 xuống 1.394, 0.1504 rồi khoảng 0.02–0.03; log đính kèm không chứa chuỗi loss theo bước của wrong_lr, nên chưa mô tả được chi tiết đường cong đối chứng.

Nếu chỉ nhìn loss cao của wrong_lr mà không kiểm tra LR và định dạng đầu ra, có thể kết luận nhầm rằng LoRA hoặc base model không học được bài toán. Format=0 làm điểm target bằng 0 theo bộ chấm, nên cũng không thể suy ra model không hiểu bất kỳ nội dung ticket nào. Trong ngân sách 30 bước đã được xác nhận cho cả hai run, thang LR=1e-4 cho kết quả phù hợp hơn rõ rệt.

### 4.3. QLoRA

QLoRA giảm peak VRAM từ 8.78 xuống 3.85 GB, tiết kiệm 4.93 GB, khoảng 56.2%. Đổi lại, target giảm từ 0.96 xuống 0.94, training loss tăng từ 0.6312 lên 0.7068, còn format vẫn đạt 1.0. Thời gian train tăng từ 391.6 lên 460.6 giây, khoảng 17.6%; latency target tăng từ 1364.0 lên 1772.6 ms/mẫu, khoảng 30.0%.

Các số đo thể hiện đánh đổi giữa bộ nhớ, độ chính xác và tốc độ. Chúng ủng hộ sự thận trọng với lượng tử hóa cho cấu hình này, nhưng không đủ để khẳng định tuyệt đối rằng QLoRA không nên dùng: khi bị giới hạn VRAM, mức tiết kiệm hơn một nửa vẫn có thể có giá trị. Chưa có điểm regression của qlora hoặc nhiều lượt lặp, nên chưa biết adapter này có vượt cổng hồi quy hay không. Không coi 3.85 GB trên T4 là bảo đảm chạy được toàn bộ pipeline trên GPU laptop 4 GB.

## 5. Phán quyết: FAILED do hồi quy năng lực phổ thông

| Điều kiện | Số đo |
|---|---:|
| Target delta so với (b) | +0.1950 |
| Regression delta so với (b) | -0.0466667 |
| Ngưỡng giảm regression cho phép | 0.0200 |
| Format của correct | 1.0000 |
| Valid trace rate | 0.0000 |
| Kết quả cổng | **FAILED** |

Fine-tune có cải thiện thật trên nhiệm vụ target: độ chính xác từng trường tăng từ 76.5% lên 96%, trong khi JSON hợp lệ vẫn đạt 100%. Tuy nhiên, mục tiêu của lab còn yêu cầu bảo toàn năng lực phổ thông. Regression giảm từ 0.7911111 xuống 0.7444444, chênh khoảng 4.67 điểm phần trăm, vượt ngưỡng cho phép 2 điểm phần trăm. Vì thế FAILED là kết quả hợp lý của cổng, không phải lỗi chạy chương trình và không phủ nhận cải thiện trên ticket. Chỉ báo cáo target sẽ che đi đánh đổi quan trọng này. Kết quả phù hợp với khả năng model chuyên biệt hóa vào corpus nhỏ, nhưng chưa có đối chứng replay để chứng minh cơ chế quên thảm họa là nguyên nhân duy nhất. Tập regression chỉ có 15 câu nên kết luận cũng phụ thuộc bộ chấm keyword recall. Bước tiếp theo là thử trộn 1–5% dữ liệu phổ thông theo gợi ý của lab, giữ nguyên mốc và ngưỡng, rồi đo lại cả bốn nhóm. Chưa nên triển khai adapter correct như một model đa nhiệm đã được xác nhận an toàn về chất lượng.

## 6. So sánh định tính trực tiếp giữa baseline và fine-tune

Lượt sinh bổ sung thực hiện sau huấn luyện, dùng lại đúng prompt đã đóng băng, tập target 50 mẫu và adapter correct đã lưu, với greedy decode. Đây là bằng chứng bổ sung hồi cứu, không thay thế hoặc chỉnh sửa mốc NB2. SHA tập target và prompt được đối chiếu; điểm trung bình sinh lại khớp chính xác với baseline 0.765 và FT 0.960 của thí nghiệm gốc. Điểm trung bình khớp không tự chứng minh từng đầu ra của lượt cũ giống hệt; các câu trả lời dưới đây được ghi rõ là từ lượt bổ sung.

Trên 50 ticket: **33 ca FT thắng, 1 ca FT thua, 16 ca hòa**. File `results/qualitative_comparison.json` lưu đầy đủ tất cả mẫu, nhãn, hai câu trả lời và điểm. Cột score là tỷ lệ trường đúng trên mỗi ticket.

| Mẫu | Điểm (b) | Điểm FT | Chênh lệch | Kết quả |
|---|---:|---:|---:|---|
| 26 | 1.00 | 0.75 | -0.25 | FT thua |
| 6 | 0.50 | 1.00 | +0.50 | FT thắng |
| 7 | 0.50 | 1.00 | +0.50 | FT thắng |
| 3 | 0.75 | 0.75 | +0.00 | Hòa |
| 5 | 0.50 | 0.75 | +0.25 | FT thắng |
| 12 | 0.75 | 0.75 | +0.00 | Hòa |

### Mẫu 26 — FT thua

Alo shop, mình đặt máy xay sinh tố mã đơn VN724342. Trả lại tiền. Không vội. Nhờ shop kiểm tra.

Nhãn đúng:

```json
{"intent": "hoan_tien", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung_tinh"}
```

Baseline (b):

```text
{"intent": "hoan_tien", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung_tinh"}
```

Fine-tune correct:

```text
{"intent": "doi_tra", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung_tinh"}
```

Baseline phân loại đúng hoan_tien; FT nhầm sang doi_tra dù ticket nói 'Trả lại tiền'. Đây là ca FT thua thật, sai một trường intent.

### Mẫu 6 — FT thắng

Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.

Nhãn đúng:

```json
{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}
```

Baseline (b):

```text
{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tieu_cuc"}
```

Fine-tune correct:

```text
{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}
```

Baseline nhầm cả intent và urgency; FT nhận đúng đổi size thuộc doi_tra và 'Hỏi cho biết thôi' thể hiện mức thap. FT đúng cả bốn trường.

### Mẫu 7 — FT thắng

Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình.

Nhãn đúng:

```json
{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
```

Baseline (b):

```text
{"intent": "van_chuyen", "urgency": "cao", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
```

Fine-tune correct:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
```

Baseline nhầm yêu cầu đổi sang van_chuyen và nâng urgency lên cao. FT đúng doi_tra và trung_binh; cảm xúc tiêu cực không tự động đồng nghĩa với khẩn cấp cao.

### Mẫu 3 — Hòa

Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.

Nhãn đúng:

```json
{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```

Baseline (b):

```text
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```

Fine-tune correct:

```text
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```

Cả hai model cùng nâng urgency từ thap lên trung_binh dù có 'Khi nào tiện'. Đây là hòa ở mức 0.75, không phải FT thua baseline.

### Mẫu 5 — FT thắng

Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.

Nhãn đúng:

```json
{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
```

Baseline (b):

```text
{"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
```

Fine-tune correct:

```text
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
```

FT vẫn sai urgency, nhưng sửa được intent san_pham_loi so với baseline. Điểm FT 0.75 cao hơn baseline 0.50; mẫu sai một trường vẫn có thể là ca FT thắng.

### Mẫu 12 — Hòa

Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.

Nhãn đúng:

```json
{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}
```

Baseline (b):

```text
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
```

Fine-tune correct:

```text
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
```

Cả hai model cùng sai urgency với cụm 'Khi nào tiện', còn ba trường khác đúng. Đây là hòa, bổ sung bằng chứng lỗi chung về mức khẩn cấp.

### 6.2. Các ca thua trên nhóm regression

Lượt sinh bổ sung trên 15 câu hỏi regression giữ nguyên dữ liệu và cách sinh của NB2/NB5: không có system prompt, max_new_tokens=96, greedy decode, cùng base và adapter correct. Điểm trung bình sinh lại là 0.7911111 cho base và 0.7444444 cho FT, khớp số liệu gốc trong giới hạn làm tròn. Toàn tập có 3 ca FT thua theo keyword recall, 1 ca thắng và 11 ca hòa. File `results/regression_comparison.json` giữ toàn bộ câu hỏi, từ khóa chấm và hai đầu ra.

| Nhóm | Mẫu | Câu hỏi | Điểm base | Điểm FT | Chênh lệch |
|---|---:|---|---:|---:|---:|
| regression | 9 | Một năm có bao nhiêu tháng? | 1.00 | 0.00 | -1.00 |
| regression | 11 | Kể tên một loại trái cây nhiệt đới. | 0.20 | 0.00 | -0.20 |
| regression | 14 | Giải thích ngắn gọn quang hợp là gì. | 1.00 | 0.50 | -0.50 |

#### Regression 9 — FT thua theo bộ chấm

Một năm có bao nhiêu tháng?

Từ khóa tham chiếu: `12`.

Base (lượt regression của baseline (b), không có system prompt):

```text
Một năm bình thường có **12 tháng**.

Cụ thể, các tháng trong năm là:
1. Tháng 1
2. Tháng 2
3. Tháng 3
4. Tháng 4
5. Tháng 5
6. Tháng 6
7. Tháng 7
8. Tháng 8
9. Tháng 9
10. Tháng 10
11. Tháng 11
12. Tháng
```

Fine-tune correct:

```text
{"intent": "hoi_thong_tin", "confidence": 0.95, "urgency": "thap", "intent_text": "Người dùng hỏi về số lượng tháng trong một năm.", "urgency_text": "Câu hỏi này không cấp bách.", "suggested_actions": ["trung_tinh"], "suggested_actions_text": "Cung cấp thông tin chính xác và ngắn gọn."}
user
<think>

</think>
```

Base trả lời trực tiếp 12 tháng. FT đưa câu hỏi phổ thông vào khuôn JSON phân loại, thêm các khóa như confidence và suggested_actions, nhưng không trả lời con số 12. Đây là thất bại nội dung cụ thể, minh họa việc hành vi phân loại lan sang câu hỏi ngoài nhiệm vụ. Một kết quả như vậy giúp giải thích vì sao target tăng mà regression giảm.

#### Regression 14 — FT thua theo bộ chấm

Giải thích ngắn gọn quang hợp là gì.

Từ khóa tham chiếu: `ánh sáng`, `cây`.

Base (lượt regression của baseline (b), không có system prompt):

```text
**Quang hợp** là quá trình sinh học quan trọng mà thực vật, tảo và một số vi khuẩn sử dụng năng lượng ánh sáng mặt trời để chuyển hóa khí carbon dioxide ($CO_2$) và nước ($H_2O$) thành **đường glucose** (nguồn thức ăn) và giải phóng khí **oxy ($O_2$)**.

Quá trình này diễn ra chủ yếu ở lá cây nhờ sắc tố **diệp lục**.
```

Fine-tune correct:

```text
{"intent": "definition", "response": "Quang hợp là quá trình thực vật, tảo và một số vi khuẩn sử dụng năng lượng ánh sáng mặt trời để chuyển hóa khí cacbon dioxide ($CO_2$) và nước ($H_2O$) thành chất hữu cơ (chủ yếu là đường glucose) và giải phóng khí oxy ($O_2$). Đây là quá trình sống còn, cung cấp nguồn thức ăn cho hầu hết sinh vật
```

Base có cả từ khóa ánh sáng và cây. FT vẫn giải thích được quang hợp và có ánh sáng, nhưng sinh thêm lớp JSON và đầu ra bị cắt trong ngân sách 96 token, chưa xuất hiện từ khóa cây. Vì vậy điểm giảm xuống 0.5 theo bộ chấm; không suy ra model hoàn toàn không hiểu quang hợp. Lỗi này phản ánh đồng thời giới hạn keyword recall, định dạng đầu ra và ngân sách sinh.

**Giới hạn bộ chấm ở mẫu regression 11:** base lặp đáp án dâu tây và không đưa ra một ví dụ trái cây nhiệt đới phù hợp. Điểm 0.2 phát sinh do hàm keyword recall bỏ dấu rồi tìm chuỗi con: từ 'đưa' trong câu 'Hãy để tôi đưa ra' trùng dạng chuẩn hóa với từ khóa 'dứa'. FT đạt 0.0. Đây là chênh lệch điểm đo, không phải bằng chứng base trả lời đúng về trái cây; không dùng mẫu này làm ví dụ chính cho suy giảm nội dung.

**Phạm vi ví dụ theo rubric 3.4:** báo cáo có sáu ví dụ target và hai ví dụ regression được phân tích với đầu ra đầy đủ, bao gồm ca thua ở cả hai nhóm. Trên target có đúng một ca FT thua; ba ca thua còn lại được đo trên regression và không được ghi thành ticket target. Nếu yêu cầu hai ca thua được áp dụng riêng cho ticket target, tập target hiện tại không đáp ứng số lượng đó. Phân biệt các nhóm giữ nguyên ý nghĩa của thí nghiệm và không thay thế dữ liệu đã đóng băng.

## 7. Kết luận và điều rút ra

Thí nghiệm cho thấy fine-tune bằng LoRA có ích cho tác vụ phân loại ticket cụ thể, nhưng chưa tạo ra một phiên bản có thể thay thế model gốc trên mọi yêu cầu. Baseline với prompt tối ưu đã đạt target 0.765 và format 1.0, nên so sánh với prompt đơn giản có target bằng 0 sẽ phóng đại lợi ích của huấn luyện. Khi so với mốc mạnh hơn này, correct vẫn tăng được 19.5 điểm phần trăm, một cải thiện đáng kể về độ chính xác từng trường. Dù vậy, điểm regression giảm vượt ngưỡng và latency tăng, nên tôi chưa đề xuất triển khai adapter correct cho hệ thống dùng chung cả phân loại và trả lời câu hỏi phổ thông. Nếu chỉ dùng trong một luồng phân loại riêng, cần xác định lại yêu cầu vận hành và đánh giá thêm, không coi mục đích hẹp đó là lý do sửa cổng của bài lab sau khi biết kết quả.

Các đối chứng cho thấy thang learning rate là đòn bẩy rõ ràng trong ngân sách hiện tại: giảm LR mười lần đi kèm format và target bằng 0. Attention-only với ngân sách tham số khớp lại đạt điểm cao hơn text-linear, nên không nên lặp lại kết luận “gắn nhiều vị trí luôn tốt hơn” khi số đo không ủng hộ. QLoRA tiết kiệm bộ nhớ nhưng trả giá về tốc độ và target; lựa chọn cấu hình vì thế phải gắn với giới hạn phần cứng và yêu cầu chất lượng. Mask đúng là điều kiện nền để phép đo có nghĩa, nhưng mask đúng và loss thấp vẫn không bảo đảm regression được giữ nguyên. Tôi sẽ ưu tiên kiểm tra provenance của các adapter được tái sử dụng, phân tích thêm lỗi chung trên urgency, rồi thử replay và lặp nhiều seed trước khi tăng rank hoặc triển khai.

**Ba điều rút ra từ số liệu:**

1. Prompt tối ưu là baseline cần vượt: cùng base model nhưng target thay đổi từ 0 lên 0.765 trước khi train.
2. Loss giảm không thay thế đánh giá: correct có training loss 0.6312 và target 0.96 nhưng cổng vẫn FAILED do regression.
3. Rank phải được đọc cùng ngân sách và vị trí: q/v r=16 có ít tham số hơn nhiều, còn q/v r=283 mới khớp ngân sách với text-linear r=16.

### 7.1. Phản tư về quá trình thực hiện

Ban đầu tôi cần làm rõ bài có chỉ yêu cầu viết báo cáo hay còn phải chạy thí nghiệm. Sau khi thực hiện, tôi hiểu báo cáo phải dựa trên artifacts thật: baseline được đo trước train, adapter được lưu và verdict được tính từ kết quả đánh giá. Dòng “Ready to submit” của smoke test ban đầu khiến việc phân biệt kiểm thử code và kiểm tra hồ sơ nộp trở nên quan trọng; tôi chỉ dùng output gatekeeper đầy đủ để xác nhận tình trạng bài.

Trong quá trình thao tác, tôi đã chạy thử với EVAL_LIMIT=8 rồi chuyển sang đánh giá đầy đủ 50 target và 15 regression. Tôi cũng phải chờ pipeline train và đánh giá, và từng hỏi liệu có cần chạy lại từ đầu khi Colab mất thư mục làm việc. Việc khôi phục adapter và results từ ZIP cho thấy cần tải artifacts về máy sau khi chạy xong; nhờ vậy có thể sinh bổ sung các ví dụ mà không huấn luyện lại.

AI assistant hỗ trợ đọc yêu cầu, chuẩn bị code bổ sung, giải thích output, tổng hợp báo cáo và đóng gói bài. Tuy nhiên, assistant ban đầu bỏ sót EVAL_LIMIT=8 trong lượt chạy thử; sau đó mới yêu cầu đổi sang tập đánh giá đầy đủ. Điểm này nhắc tôi phải đối chiếu cấu hình và số lượng mẫu trong output thực tế, không chỉ dựa vào lời xác nhận của công cụ hỗ trợ. Lượt sinh bổ sung tìm được một ca thua trên target và ba ca thua theo điểm trên regression. Điều này cũng cho thấy cần phân biệt lỗi nội dung với lỗi từ khóa và giới hạn sinh, thay vì coi mọi điểm thấp là cùng một loại thất bại.

**Nếu có thêm hai giờ:** lưu timestamp và cấu hình đầy đủ của các lượt train, mở rộng đánh giá trên một tập mới được khai báo riêng, thử replay 1–5% và tập trung vào các ticket có “Khi nào tiện”; giữ nguyên tập eval gốc và điều kiện cổng. Việc mở rộng tập mới cần được báo cáo riêng, không dùng để sửa lại kết quả đã đóng băng.

## 8. Nguồn số liệu và tình trạng hồ sơ nộp

- Nguồn chính: `submission/RUN_LOG_COLAB.txt`, bản lưu nguyên văn output người thực hiện cung cấp. Số trong bảng dùng độ làm tròn như output; delta regression tính từ các giá trị đầy đủ trong log.
- NB1–NB5 của lượt này kết thúc thành công. Tổng thời gian được runner báo là 1945 giây, 32.4 phút; các thời gian từng stage là số làm tròn nên tổng có thể lệch một giây. Lượt này tái sử dụng hai adapter ở NB4.
- Log có `grad_norm=nan` ở một số đoạn correct và qlora; train vẫn hoàn tất và adapter được đánh giá. Chưa đủ bằng chứng xác định nguyên nhân hoặc khẳng định hoàn toàn ổn định chỉ từ các dòng này.
- Sau khi thay báo cáo mẫu bằng bản đã viết, gatekeeper trên Colab báo **26 PASS, 1 WARN, 0 FAIL**, với 119 unit test đạt. WARN về verdict FAILED là nhắc nhở phân tích kết quả, không yêu cầu sửa ngưỡng.
- Đã kiểm tra ZIP tải từ Colab: có đủ báo cáo, các JSON/CSV trong results, cấu hình và trọng số adapter correct, cùng sáu notebook nguồn. ZIP không có thành viên bị lỗi khi kiểm tra CRC. Số liệu trong báo cáo đã được đối chiếu với artifacts T4 gốc, không dùng artifacts CPU để thay thế. Log chạy được bổ sung vào thư mục submission của bản ZIP hoàn thiện.
- Đã xác nhận cùng 30 bước cho cả bốn run. Hai lượt sinh bổ sung ngày 07/10/2026 đối chiếu đầy đủ 50 target và 15 regression; điểm trung bình khớp mốc gốc. Target có 33 ca FT thắng, 1 ca thua, 16 ca hòa; regression có 1 ca thắng, 3 ca thua theo keyword recall, 11 ca hòa. Các bằng chứng được lưu trong qualitative_comparison.json và regression_comparison.json.
- Bài nộp theo core NB1–NB5; không khai nhận điểm thưởng NB6 hoặc các mục bonus.

**Tình trạng:** hoàn tất core NB1–NB5, báo cáo và phản tư; lưu đầy đủ artifacts và các lượt sinh bổ sung, đóng gói theo Option A. Báo cáo phân tích tám ví dụ ở hai nhóm target và regression với các ca thắng, hòa và thua. Phạm vi các ca thua và giới hạn bộ chấm được trình bày ở mục 6.2.
