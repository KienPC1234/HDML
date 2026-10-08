# HỒ SƠ NỘP DỰ ÁN HDML (BẢN ẨN DANH)

> Toàn bộ hồ sơ đã **loại bỏ thông tin cá nhân** (không có tên/email/công ty/GitHub/DOI cá nhân).
> Không có thông tin đơn vị dự thi và thí sinh trong báo cáo/poster, đúng yêu cầu Phụ lục 6.

## Cấu trúc thư mục

```
submission/
├── README_NOP_BAI.md                 <-- file này
├── 01_Bai_bao/
│   ├── HDML_paper.pdf                (bài báo đã ẩn danh)
│   ├── LaTeX_source/                 (main.tex, references.bib, arxiv.sty, figures/)
│   └── previews/                     (ảnh xem trước + hình kiến trúc, biểu đồ)
├── 02_Bao_cao_ket_qua/
│   ├── HDML_BaoCao_KetQua_PhuLuc6.docx   (bản chỉnh sửa được, có ô ký)
│   └── HDML_BaoCao_KetQua_PhuLuc6.pdf    (bản in để ký nộp)
├── 03_Poster/
│   └── HDML_Poster_PhuLuc7.docx
├── 04_Du_lieu_minh_chung/
│   ├── results/                      (log bảng điểm gốc)
│   ├── plots/                        (biểu đồ)
│   └── videos/                       (video minh chứng .mp4 + .png + gif nhỏ)
└── 05_Ma_nguon/
    └── HDML_source.zip               (mã nguồn hdml/, scripts/, configs/, tests/)
```

## Đối chiếu với hướng dẫn cuộc thi (tài liệu trong thư mục KH thi KHKT)

| Phụ lục | Mục đích | File tương ứng |
| :-- | :-- | :-- |
| Phụ lục 6 | Báo cáo kết quả thực hiện dự án | `02_Bao_cao_ket_qua/...docx` |
| Phụ lục 7 | Poster | `03_Poster/...docx` |
| Phụ lục 8 | Hướng dẫn dùng AI tạo sinh | Cần ghi nhật ký AI riêng (xem mục dưới) |
| Phụ lục 9 | Tiêu chí chấm điểm | Đối chiếu phần "Bản đồ tiêu chí" |

## Bản đồ tiêu chí chấm điểm (Phụ lục 9)

| Tiêu chí | Bằng chứng trong hồ sơ |
| :-- | :-- |
| I. Mục tiêu, đóng góp | `02...docx` mục I; `01_Bai_bao` Abstract |
| I. Tính khả kiểm chứng | 5 hạt giống; giao thức rliable; `04.../results` |
| II. So sánh phương án | `02...docx` mục II.1 |
| II. Nguyên mẫu/mô hình | Hình kiến trúc + bảng tham số |
| III. Chế tạo & kiểm tra nhiều điều kiện | 6 robot đích; nhiễu vật lý; ONNX |
| III. Đủ dữ liệu minh chứng | `04_Du_lieu_minh_chung` + `05_Ma_nguon` |
| IV. Tính mới | `02...docx` mục IV |
| V. Trình bày | Poster `03...docx` + bài báo |

## Lưu ý quan trọng

1. **Báo cáo kết quả (Phụ lục 6)** đã có sẵn **cả .docx và .pdf** (Times New Roman 13, lề 3/2/2/2 cm,
   cách dòng đơn) và **có sẵn ô ký** ở cuối → in bản .pdf để ký nộp.
2. **Poster** hiện là `.docx`; máy không có Word/LibreOffice nên chưa xuất PDF — mở bằng Word rồi
   **Save as PDF** (hoặc nhờ tôi dựng poster điện tử khổ lớn xuất PDF).
3. **Nhật ký nghiên cứu (Phụ lục 3)** và **nhật ký sử dụng AI (Phụ lục 8)** phải viết tay/đính
   kèm riêng theo quy định; hồ sơ này chưa thay thế được các phần đó.
4. Kiểm tra lại giới hạn 15 trang khi bổ sung thông tin đơn vị/thí sinh vào bản chính thức.
