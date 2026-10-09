# Phát hiện kiểm chứng: config foundation gốc không build lại được với code hiện tại

_Ngày 09/10/2026._

## Hiện tượng
Khi chạy `scripts/train_foundation.py --config configs/hdml_foundation_full.yaml`,
chương trình dừng ngay khi khởi tạo bộ đệm dữ liệu:

```
ValueError: No episode is long enough for the requested context length
  at hdml/data/episodes.py:104 (pack_episodes)
```

## Nguyên nhân gốc
`configs/hdml_foundation_full.yaml` dùng `context_length = 30`. Một số bộ dữ liệu
có episode ngắn hơn 30 bước:

| Bộ dữ liệu | #episode | dài nhất | #episode < 30 |
|---|---:|---:|---:|
| `inv_double_pendulum_foundation` | 300 | **17** | 300 |
| `walker2d_foundation` | 300 | 64 | 240 |
| `hopper_foundation` | 300 | 103 | 215 |
| `humanoid_foundation` | 240 | 47 | 161 |
| `ant_foundation` | 300 | 273 | 84 |

`inv_double_pendulum_foundation` có **mọi** episode ngắn hơn context (dài nhất 17),
nên `pack_episodes` không tạo được cửa sổ hợp lệ nào và báo lỗi.

## Hệ quả trung thực
- Checkpoint `checkpoints/hdml_foundation/hdml_foundation_best.pt` (16.503.732 tham
  số) được train từ trạng thái code/config trước đó; với code hiện tại, **không thể
  tái lập cùng lệnh huấn luyện**.
- Đây là một lỗ hổng tái lập cần nêu rõ, dù bản thân checkpoint và các số trong
  `logs/benchmark_foundation_results.json` vẫn là artifact thật trên đĩa.

## Cách xử lý khi muốn train lại
Với mục tiêu held-out (bỏ Walker2d khỏi tiền huấn luyện), nhóm dùng
`configs/hdml_foundation_holdout.yaml`: giữ 9 hình thái, trong đó buộc phải loại
thêm `inv_double_pendulum` (episode quá ngắn). Tổng còn 1.382.849 bước.

Hướng sửa dài hạn (khuyến nghị): cho phép `context_length` khác nhau theo từng hình
thái, hoặc bỏ hẳn các bộ dữ liệu có episode ngắn hơn context, rồi ghi lại tổng số bước
chính xác sau khi lọc.
