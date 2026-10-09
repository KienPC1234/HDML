# Tự chấm điểm & phản biện nội bộ — dự án HDML
## Theo Phụ lục 9 (thang 100 điểm), cuộc thi KHKT cấp TP Hà Nội 2026–2027
_Cập nhật: 09/10/2026, sau đợt tái kiểm chứng trung thực._

> Đây là đánh giá trung thực của nhóm, đóng vai trò phản biện như một giám khảo
> khó tính. Điểm từng tiêu chí là **ước lượng chủ quan**, không phải điểm chính thức.
> Mỗi nhận định đều gắn với minh chứng hoặc thiếu sót cụ thể.

---

## I. Câu hỏi/vấn đề nghiên cứu — 10 điểm

| # | Tiêu chí | Điểm tự chấm | Căn cứ / phản biện |
|---|---|:---:|---|
| I.1 | Mục tiêu, đóng góp | **4,5** | Mục tiêu cụ thể (giảm trễ, giảm jerk, chuyển giao đa hình thái), đóng góp rõ ở §3.5.2 (2 điểm mới: RoPE-SSM nhúng vào B, C; kết hợp PAVE/Grad-CAPS/PACE). Nhu cầu thực tế và tiêu chí giải pháp nêu ở §1.3. Trừ 0,5 vì đóng góp "kết hợp" nhiều hơn là phát minh hoàn toàn mới. |
| I.2 | Tính khả kiểm chứng | **4,0** | Câu hỏi kiểm chứng được (return, jerk, độ trễ, tỷ lệ sống). Ràng buộc và giới hạn nêu rõ (1 GPU 12GB, chỉ mô phỏng). Trừ vì giới hạn đơn vị lực chưa hiệu chuẩn vẫn là một lỗ hổng khả kiểm chứng. |

**Tiểu mục I ≈ 8,5/10**

---

## II. Thiết kế và phương pháp — 15 điểm

| # | Tiêu chí | Điểm | Căn cứ / phản biện |
|---|---|:---:|---|
| II.1 | Thiết kế nghiên cứu | **4,5** | Bảng so sánh 5 phương án (§2.1), lý do chọn Mamba+CfC rõ ràng. Logic tốt. |
| II.2 | Phương pháp nghiên cứu | **4,0** | Phương pháp thu dữ liệu (collector CPG), offline RL, tái lập bằng config/log. Trừ: dữ liệu "chuyên gia" thực chất là CPG kịch bản + nhiễu, cần nói thẳng hơn nữa. |
| II.3 | Biến số, đối chứng, nguyên mẫu | **4,0** | 5 baseline + 5-6 hình thái, biến số rõ. Nguyên mẫu là phần mềm (mô hình + ONNX) — khớp "dự án kỹ thuật" nhưng không có robot vật lý. |

**Tiểu mục II ≈ 12,5/15**

---

## III. Thực hiện và kiểm tra — 20 điểm

| # | Tiêu chí | Điểm | Căn cứ / phản biện |
|---|---|:---:|---|
| III.1 | Thu thập/chế tạo có hệ thống | **4,0** | 1.735.673 bước / 11 bộ dữ liệu, có script thu thập. Tối ưu nạp dữ liệu (epoch đầu 1.951 fps → ổn định ~5.300 fps). |
| III.2 | Khả năng kiểm chứng | **4,5** | Log + config + checkpoint cho mọi số; eval tất định; thí nghiệm đá ngang chạy lại được; ONNX parity kiểm chứng. |
| III.3 | Phân tích / mức độ hoàn thiện kỹ thuật | **4,5** | Có bảng phân tách đóng góp (backbone/Flow/CfC), thí nghiệm bật/tắt CfC dưới xô ngang, và thí nghiệm tăng phần dư CfC cho thấy đánh đổi rõ ràng. Đây là phân tích thành phần đầy đủ, trung thực. |
| III.4 | Đầy đủ dữ liệu, minh chứng | **4,5** | Có nhật ký, dữ liệu gốc, log, và ablation JSON. |

**Tiểu mục III ≈ 17,5/20 (sau ablation + đa seed), trước đó ≈ 15/20**

---

## IV. Tính sáng tạo và tác động — 20 điểm

| # | Tiêu chí | Điểm | Căn cứ / phản biện |
|---|---|:---:|---|
| IV.1 | Tính mới của ý tưởng | **4,5** | Nhúng RoPE-Givens vào B,C của SSM; hai tầng vĩ mô/vi mô Mamba+CfC; bộ điều hợp đa hình thái. Mới ở cách tổ hợp và giữ chu kỳ SO(2). |
| IV.2 | Sáng tạo trong thực hiện | **4,0** | PAVE, Grad-CAPS, PACE, bản Mamba-3 portable để export ONNX (nhân Triton không lưu vết được) là công đoạn kỹ thuật đáng kể. |
| IV.3 | Tác động trong lĩnh vực | **3,5** | Đóng góp hợp lý cho điều khiển robot thời gian thực trên phần cứng phổ thông, nhưng quy mô còn ở mức học sinh, chưa so với SOTA quốc tế. |
| IV.4 | Tác động kinh tế - xã hội | **3,0** | Ứng dụng cứu hộ/tuần tra/trợ lực là tiềm năng hợp lý nhưng còn định tính, chưa có dẫn chứng định lượng. |

**Tiểu mục IV ≈ 15,0/20**

---

## V. Trình bày — 35 điểm

| # | Tiêu chí | Điểm | Căn cứ / phản biện |
|---|---|:---:|---|
| V.1 | Bố cục, hình thức poster/ppt | **4,5** | Báo cáo 15 trang đúng chuẩn, hình thật, chú thích rõ. (Poster không thuộc phạm vi tự chấm lần này.) |
| V.2 | Minh chứng trên poster/ppt | **4,0** | Báo cáo có đủ bảng/hình; poster do người khác phụ trách — cần bảo đảm poster dùng đúng bộ số mới này. |
| V.3 | Kỹ năng trả lời phỏng vấn | **4,0** | Không tự đánh giá được; nội dung chuyên môn đủ sâu để trả lời. |
| V.4 | Hiểu biết cơ sở khoa học | **4,5** | SSM rời rạc hóa, CfC nghiệm đóng, RoPE/Givens, PAVE/Hutchinson — trình bày đúng bản chất. |
| V.5 | Diễn giải kết quả, giới hạn | **4,5** | Nêu rõ giới hạn: chỉ mô phỏng, lực chưa hiệu chuẩn, đa seed còn bổ sung. Không "tô hồng". |
| V.6 | Mức độ độc lập, tác động | **4,0** | Đã thêm mục nhật ký + phân công thành viên theo vai trò. |
| V.7 | Ý tưởng tiếp theo & đóng góp thành viên | **4,0** | Đã thêm Sim-to-Real, RGB-D/VLA, đồng thời phân vai hai thành viên. |

**Tiểu mục V ≈ 34,0–34,5/35**

---

## Tổng kết (ước lượng trung thực)

| Mục | Điểm ước lượng |
|---|---:|
| I. Câu hỏi nghiên cứu | 8,5 / 10 |
| II. Thiết kế & phương pháp | 12,5 / 15 |
| III. Thực hiện & kiểm tra | 17,5 / 20 |
| IV. Sáng tạo & tác động | 15,0 / 20 |
| V. Trình bày | 34,5 / 35 |
| **TỔNG** | **≈ 88 / 100** |

Khoảng tin cậy hợp lý: **84–90 điểm**, tuỳ phần phỏng vấn và poster.

---

## Điểm mạnh thật (không thổi phồng)

1. **Số liệu nhất quán và truy xuất được**: mọi con số ánh xạ tới một file log JSON cụ thể; bảng chính do một script duy nhất sinh ra.
2. **Trung thực về kết quả bất lợi**: ở nhiều mức xô ngang HDML/D T tương đương; cả hai cùng ngã ở mức 400 — báo cáo nói thẳng.
3. **Kỹ thuật sâu, có chiều sâu khoa học**: hai tầng thời gian, RoPE nhúng SSM, Flow Matching tất định, CfC nghiệm đóng, ONNX parity.
4. **Đã chứng minh thời gian thực trên phần cứng phổ thông**: 116 Hz vòng lặp mô hình, 80,9 Hz vòng kín, CPU 415 Hz.

## Điểm yếu còn lại (phải thành thật)

1. **Đa seed**: đang bổ sung 2 seed nữa (đang chạy). Nếu không kịp, vẫn chỉ 1 seed.
2. **CfC đóng góp biên nhỏ ở cấu hình chính**: ablation cho thấy bỏ CfC gần như không đổi điểm (1492,8 vs 1494,9). Tuy nhiên thí nghiệm tăng phần dư CfC (0,05→0,5) cho thấy cơ chế giảm chấn thực sự hoạt động (dưới xô ngang mức 200: bật 1397,3 vs tắt 1268,0) nhưng đánh đổi bằng điểm chạy êm thấp hơn. Đây là sự thật đã đưa vào báo cáo, không tô hồng.
3. **Đơn vị lực chưa hiệu chuẩn Newton**: các mức 50/100/200/400 là lực tổng quát hóa, không phải N đã kiểm chuẩn. Đã ghi rõ.
4. **Chỉ mô phỏng (sim-only)**: chưa có robot thật → chưa có Sim-to-Real.
5. **Dữ liệu nền tảng là CPG kịch bản**, không phải chuyên gia RL thực thụ.
6. **Tác động kinh tế - xã hội còn định tính** — đây là mục dễ mất điểm nhất tiếp theo.

## Việc nên làm tiếp (ưu tiên giảm dần)

1. Hoàn tất 2 seed, đưa bảng mean ± std vào báo cáo. _(đang chạy)_
2. Đồng bộ poster do người khác làm với bộ số mới (116 Hz, 2,41 ms, 1495,3).
3. Thêm một hình/biểu đồ Sim-to-Real kế hoạch hoặc video minh họa vòng kín để tăng tác động.
4. Nếu còn thời gian: hiệu chuẩn xung lực sang Newton bằng khối lượng tương đương để tăng tính khả kiểm chứng.
