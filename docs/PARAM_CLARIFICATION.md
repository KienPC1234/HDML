# Làm rõ con số tham số của mô hình nền tảng (foundation)

_Cập nhật: 09/10/2026._

Trong quá trình audit có hai con số tham số dễ gây hiểu nhầm:

| Con số | Nguồn | Ý nghĩa |
|---|---|---|
| **16.503.732** | checkpoint `checkpoints/hdml_foundation/hdml_foundation_best.pt` | Tổng tham số khi **đăng ký đủ 11 bộ điều hợp hình thái** trong lúc pre-train (mỗi adapter là một `nn.Module` riêng được ghi vào `state_dict`). |
| **11.989.918** | mô hình nền tảng trước khi đăng ký adapter | Phần đường trục dùng chung thực tế (`mamba_backbone` + `fusion` + `cfc_filter` + các head). |

Khi **chuyển giao few-shot** sang một robot mục tiêu, script
`scripts/evaluate_foundation_transfer.py`:
1. nạp phần dùng chung (`shared`, loại bỏ `adapters.*`),
2. chỉ đăng ký **một** adapter cho robot mục tiêu,
3. `freeze_backbone()` → đóng băng toàn bộ đường trục, chỉ huấn luyện adapter.

Vì vậy, với robot Unitree A1 (Maze, 53→12):

```
frozen backbone   = 11.989.918   (96,8 %)
trainable adapter =    399.564   ( 3,2 %)
total             = 12.389.482
```

Con số trong `logs/benchmark_foundation_results.json` là:
`frozen = 11.606.044`, `trainable = 783.438`, `% frozen = 93,7 %`.
Hai con số `frozen` khác nhau do bản ghi cũ tính theo một cấu hình khác; **đây là điểm
cần thống nhất nếu muốn trích dẫn tuyệt đối**. Trong báo cáo, nhóm chỉ nêu:
"đóng băng ~11,6 triệu tham số đường trục, tinh chỉnh khoảng 760–900 nghìn tham số bộ
điều hợp (~6–7 %)" — bám sát đúng bản ghi log, tránh suy diễn thêm.

## Việc cần làm nếu muốn con số tuyệt đối chuẩn
Chạy lại `scripts/evaluate_foundation_transfer.py` cho từng robot và dùng trực tiếp
`frozen_params` / `trainable_params` trong JSON mới, bỏ hẳn bản ghi cũ.
