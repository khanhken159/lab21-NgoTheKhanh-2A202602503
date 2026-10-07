# Lab 21 — Ngô Thế Khanh — 2A202602503

Bài thực hành LoRA/QLoRA cho phân loại ticket CSKH tiếng Việt thành JSON.
Source thí nghiệm dựa trên [VinUni-AI20k/Day21-Track3-Finetuning-Lab](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab), commit `d27c1c0`.
Nguồn được giữ theo commit đã chạy; hai script bổ sung chỉ sinh bằng chứng định tính, không train hoặc sửa mốc gốc. License gốc được giữ trong `LICENSE`.

## Hồ sơ và kết quả

- [Báo cáo](submission/REPORT.md): model, dataset, mask, baseline, bốn run, phán quyết và ví dụ định tính.
- [Phản tư](submission/REFLECTION.md).
- [Results](results/): toàn bộ JSON/CSV thí nghiệm và hai lượt sinh bổ sung.
- [Adapter correct](adapters/correct/): trọng số, cấu hình và tokenizer; trọng số được quản lý bằng Git LFS.
- [Notebook nguồn](notebooks/), [mã dùng chung](src/labkit/) và [rubric](rubric.md).

GPU Tesla T4, fp16; train/val 225/25, eval target 50 và regression 15.
Target: baseline tối ưu 0.765, LoRA 0.960. Regression: 0.7911 → 0.7444.
Cổng model FAILED do regression giảm quá ngưỡng 0.02; đây là kết quả được phân tích trong báo cáo.
Gatekeeper Colab sau hoàn thiện báo cáo: 26 PASS, 1 WARN, 0 FAIL; 119 unit test đạt trong môi trường Colab.

Lượt sinh bổ sung khớp điểm gốc: target có 33 thắng, 1 thua, 16 hòa; regression có 1 thắng, 3 thua theo keyword recall, 11 hòa.
Các ca regression được ghi đúng nhóm, không được tính thành ticket target. Xem mục 6 của báo cáo để biết phạm vi bằng chứng và giới hạn bộ chấm.

## Tải và kiểm tra

```bash
git lfs install
git clone <URL_REPO_BAI_NOP>
cd lab21_2A202602503
git lfs pull
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe scripts/verify.py
```

Linux/Colab:

```bash
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python scripts/verify.py
```

Kiểm tra artifacts không cần train lại. Trọng số adapter phải là file thật sau `git lfs pull`, không chỉ là LFS pointer.
Để tái lập lượt GPU, dùng T4 và `requirements.txt`, chạy NB1 → NB5 theo [README gốc](README_LAB.md).
Hai script `scripts/complete_qualitative.py` và `scripts/complete_regression_examples.py` sinh lại đầu ra theo từng mẫu, giữ nguyên baseline và verdict gốc.
File requirements thể hiện khoảng phiên bản của lab; repo không có pip freeze của phiên GPU cũ, nên không khai đây là môi trường đã khóa chính xác mọi phiên bản.
