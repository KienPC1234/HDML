#!/usr/bin/env python3
"""Build standalone poster HTML with all images embedded as Base64 data URIs.
Ensures zero broken images when downloaded and opened locally on any machine.
Applies professional scientific styling and natural Vietnamese academic tone.
"""
from __future__ import annotations

import base64
from pathlib import Path

def main() -> None:
    repo_root = Path("/data/HDML_Model")
    assets_dir = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_assets"
    output_html = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_khkt.html"
    output_b64_copy = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_khkt_base64.html"
    artifact_dir_env = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")
    artifact_dir = Path(artifact_dir_env) if artifact_dir_env else None

    images = {
        "a1_thumb.png": "image/png",
        "walker2d_raw.png": "image/png",
        "humanoid_raw.png": "image/png",
        "chart_torque_chatter_k1.png": "image/png",
        "hdml_architecture.png": "image/png",
        "chart_unitree_returns.png": "image/png",
        "chart_unitree_multiseed.png": "image/png",
        "chart_unitree_jerk.png": "image/png",
        "waveforms_unitree.png": "image/png",
        "unitree_a1_kick_recovery_real.png": "image/png",
        "qr_code.png": "image/png",
    }

    b64_data = {}
    for filename, mime in images.items():
        file_path = assets_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required asset: {file_path}")
        encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        b64_data[filename] = f"data:{mime};base64,{encoded}"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Poster Trưng Bày Cuộc Thi KHKT - Dự Án HDML</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --poster-width: 1200px;
    }}

    body {{
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      background-color: #0f172a;
      margin: 0;
      padding: 0;
      color: #0f172a;
      -webkit-font-smoothing: antialiased;
    }}

    .font-heading {{
      font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    }}

    /* Khung poster chính (Standee 80x160cm - Tỷ lệ 1:2) */
    .poster-canvas {{
      width: var(--poster-width);
      min-height: 2400px;
      background: linear-gradient(180deg, #e6f4fa 0%, #edf7fc 40%, #e1eff7 100%);
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
      margin: 20px auto;
      padding: 32px 36px 40px 36px;
      box-sizing: border-box;
      position: relative;
      transition: transform 0.2s ease-in-out;
    }}

    /* Các khung nội dung bo tròn chuẩn triển lãm */
    .panel-box {{
      background: #ffffff;
      border: 1.5px solid #cce3f0;
      border-radius: 24px;
      box-shadow: 0 10px 25px -5px rgba(12, 74, 110, 0.06), 0 4px 6px -4px rgba(12, 74, 110, 0.03);
      padding: 24px 28px;
      position: relative;
    }}

    /* Nhãn tiêu đề từng khung */
    .panel-pill {{
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #ffffff;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: 0.5px;
      padding: 8px 24px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
      margin-bottom: 18px;
    }}

    /* Ảnh nét và giữ tỷ lệ */
    img {{
      image-rendering: auto;
      object-fit: contain;
    }}

    /* Chuẩn in Standee thương mại 800mm x 1600mm hoặc A0 */
    @media print {{
      @page {{
        size: 800mm 1600mm;
        margin: 0;
      }}
      body {{
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .no-print {{
        display: none !important;
      }}
      .poster-canvas {{
        margin: 0 !important;
        width: 100vw !important;
        min-height: 100vh !important;
        box-shadow: none !important;
        padding: 30mm 35mm !important;
        transform: none !important;
      }}
    }}
  </style>
</head>
<body class="py-6 px-2 sm:px-4">

  <!-- Thanh công cụ điều khiển (Tự động ẩn khi in hoặc xuất PDF) -->
  <header class="no-print max-w-5xl mx-auto mb-6 bg-slate-900/90 backdrop-blur-md text-white px-6 py-4 rounded-2xl shadow-xl flex flex-wrap items-center justify-between gap-4 border border-slate-700/60 sticky top-4 z-50">
    <div class="flex items-center gap-3">
      <div class="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
      <div>
        <h1 class="text-sm font-bold tracking-wide">Poster Standee KHKT (Bản Nhúng Sẵn Ảnh Base64 Offline)</h1>
        <p class="text-xs text-slate-400">Tệp HTML độc lập 100% | Tải về máy mở là xem được ngay | Xuất PDF / In khổ 80x160cm</p>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <button onclick="toggleFit()" id="fitBtn" class="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-xs font-semibold rounded-xl border border-slate-600 transition flex items-center gap-1.5">
        <svg class="w-4 h-4 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path></svg>
        <span>Vừa màn hình</span>
      </button>
      <button onclick="window.print()" class="px-5 py-2 bg-sky-600 hover:bg-sky-500 text-xs font-bold rounded-xl shadow-lg shadow-sky-600/30 transition flex items-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"></path></svg>
        <span>In Poster / Xuất PDF</span>
      </button>
    </div>
  </header>

  <!-- KHUNG POSTER CHÍNH -->
  <main id="posterContainer" class="poster-canvas flex flex-col justify-between space-y-6">

    <!-- PHẦN ĐẦU POSTER (HEADER) -->
    <header class="space-y-4">
      <!-- Dải băng thông tin cuộc thi & lĩnh vực -->
      <div class="bg-gradient-to-r from-slate-900 via-sky-950 to-slate-900 text-white rounded-2xl px-6 py-3.5 shadow-md border border-sky-900/60 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
          <span class="text-xs font-extrabold tracking-wider text-amber-300 uppercase">MÃ SỐ DỰ ÁN: .........</span>
          <span class="text-slate-400 text-xs mx-2">|</span>
          <span class="text-xs font-semibold text-slate-200 uppercase">CUỘC THI NCKH KỸ THUẬT (KHKT) HÀ NỘI 2026 - 2027</span>
        </div>
        <div class="bg-sky-500/20 border border-sky-400/40 text-sky-200 text-xs font-bold px-3.5 py-1 rounded-full uppercase tracking-wide">
          LĨNH VỰC: RÔ BỐT VÀ MÁY THÔNG MINH
        </div>
      </div>

      <!-- Tên đề tài nghiên cứu -->
      <div class="text-center px-4 py-2">
        <h2 class="font-heading text-[27px] font-extrabold text-sky-950 leading-snug tracking-tight uppercase">
          NGHIÊN CỨU PHÁT TRIỂN KIẾN TRÚC MÔ HÌNH NỀN TẢNG PHÂN CẤP MAMBA-LIQUID ĐIỀU KHIỂN CHUYỂN ĐỘNG ROBOT ĐA HÌNH THÁI VỚI ĐỘ TRỄ THẤP VÀ KHÁNG NHIỄU CAO
        </h2>
      </div>
    </header>

    <!-- KHUNG 1: VẤN ĐỀ NGHIÊN CỨU -->
    <section class="panel-box">
      <div class="panel-pill">
        <span>1. VẤN ĐỀ NGHIÊN CỨU</span>
      </div>

      <div class="grid grid-cols-12 gap-6 items-center">
        <!-- Cột trái: 3 Robot đa hình thái (7/12) -->
        <div class="col-span-7 space-y-3">
          <div class="grid grid-cols-3 gap-3 bg-sky-50/50 p-3.5 rounded-2xl border border-sky-100">
            <!-- Robot 1: Unitree A1 -->
            <div class="bg-white p-2.5 rounded-xl border border-slate-200/80 shadow-2xs text-center flex flex-col items-center justify-between">
              <div class="h-28 flex items-center justify-center w-full">
                <img src="{b64_data['a1_thumb.png']}" alt="Quadruped 12 DoF" class="max-h-24 max-w-full">
              </div>
              <span class="text-[11px] font-bold text-slate-800 mt-1 block">Quadruped (12 DoF)</span>
              <span class="text-[10px] text-slate-500">Robot 4 chân</span>
            </div>
            <!-- Robot 2: Walker2d -->
            <div class="bg-white p-2.5 rounded-xl border border-slate-200/80 shadow-2xs text-center flex flex-col items-center justify-between">
              <div class="h-28 flex items-center justify-center w-full">
                <img src="{b64_data['walker2d_raw.png']}" alt="Biped Walker 6 DoF" class="max-h-24 max-w-full">
              </div>
              <span class="text-[11px] font-bold text-slate-800 mt-1 block">Biped Walker (6 DoF)</span>
              <span class="text-[10px] text-slate-500">Robot 2 chân</span>
            </div>
            <!-- Robot 3: Humanoid -->
            <div class="bg-white p-2.5 rounded-xl border border-slate-200/80 shadow-2xs text-center flex flex-col items-center justify-between">
              <div class="h-28 flex items-center justify-center w-full">
                <img src="{b64_data['humanoid_raw.png']}" alt="Humanoid 17 DoF" class="max-h-24 max-w-full">
              </div>
              <span class="text-[11px] font-bold text-slate-800 mt-1 block">Humanoid (17 DoF)</span>
              <span class="text-[10px] text-slate-500">Robot hình người</span>
            </div>
          </div>
          <p class="text-xs text-slate-600 leading-relaxed font-medium px-1">
            <strong class="text-slate-800">Thách thức đa hình thái:</strong> Khác biệt về số khớp và cảm biến (A1: 12 khớp, Walker2d: 6 khớp, Humanoid: 17 khớp) khiến mô hình điều khiển khó dùng chung mà không phải huấn luyện lại từ đầu.
          </p>
        </div>

        <!-- Cột phải: Đồ thị Torque Chatter + Vấn đề & Mục tiêu (5/12) -->
        <div class="col-span-5 space-y-3">
          <!-- Đồ thị so sánh rung giật -->
          <div class="bg-white p-2 rounded-xl border border-slate-200 shadow-2xs">
            <img src="{b64_data['chart_torque_chatter_k1.png']}" alt="So sánh rung giật lực xoắn" class="w-full h-auto rounded-lg">
          </div>

          <!-- Khối chữ Vấn đề & Mục tiêu -->
          <div class="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-2 text-xs text-slate-700 leading-relaxed">
            <div>
              <strong class="text-red-700 font-bold block mb-0.5">Vấn đề kỹ thuật:</strong>
              <p>• Transformer truyền thống có độ trễ tăng nhanh theo chuỗi O(L^2), dễ trễ chu kỳ điều khiển 50 Hz.</p>
              <p>• Đầu ra rời rạc sinh rung giật mô-men cao, gây nóng động cơ và mất thăng bằng khi tiếp xúc.</p>
            </div>
            <div class="border-t border-slate-200 pt-2">
              <strong class="text-sky-800 font-bold block mb-0.5">Mục tiêu giải pháp:</strong>
              <p>Thiết kế kiến trúc phân cấp hai tầng HDML: Tầng Mamba vĩ mô xử lý ngữ cảnh dài với độ phức tạp tuyến tính O(L); Tầng Liquid vi mô liên tục hóa tín hiệu điều khiển, giảm rung giật và bảo vệ khớp động cơ.</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- KHUNG 2: THIẾT KẾ VÀ PHƯƠNG PHÁP NGHIÊN CỨU -->
    <section class="panel-box">
      <div class="panel-pill">
        <span>2. THIẾT KẾ VÀ PHƯƠNG PHÁP NGHIÊN CỨU</span>
      </div>

      <div class="grid grid-cols-12 gap-6 items-center">
        <!-- Ảnh sơ đồ kiến trúc HDML 400 DPI (8/12) -->
        <div class="col-span-8 bg-sky-50/40 p-3 rounded-2xl border border-sky-100 flex items-center justify-center">
          <img src="{b64_data['hdml_architecture.png']}" alt="Sơ đồ kiến trúc mô hình HDML" class="w-full h-auto rounded-xl shadow-2xs">
        </div>

        <!-- Hộp Phương pháp nghiên cứu (4/12) -->
        <div class="col-span-4 bg-slate-50 p-5 rounded-2xl border border-slate-200 space-y-3.5 text-xs text-slate-700 leading-relaxed">
          <h3 class="font-heading text-sm font-bold text-sky-950 uppercase border-b border-slate-200 pb-2">
            PHƯƠNG PHÁP NGHIÊN CỨU
          </h3>
          <div class="space-y-3">
            <div>
              <strong class="text-slate-900 font-bold block">• Mô phỏng vật lý:</strong>
              <span class="text-slate-600">Môi trường MuJoCo (Gymnasium v5), chu kỳ điều khiển danh định 50 Hz (20 ms), bước tích phân 0,002 giây.</span>
            </div>
            <div>
              <strong class="text-slate-900 font-bold block">• Huấn luyện ngoại tuyến:</strong>
              <span class="text-slate-600">Huấn luyện trên 14 tập dữ liệu quỹ đạo (6,56 triệu bước) bằng 1 GPU 12GB; áp dụng điều chuẩn trơn PAVE và Grad-CAPS.</span>
            </div>
            <div>
              <strong class="text-slate-900 font-bold block">• Đánh giá thực nghiệm:</strong>
              <span class="text-slate-600">Kiểm chứng vòng kín trên robot Unitree A1 (12 DoF) qua 3 seed độc lập; đo kiểm độ trễ, rung giật và khả năng chịu xung lực xô đẩy.</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- KHUNG 3: DỮ LIỆU VÀ PHÂN TÍCH DỮ LIỆU -->
    <section class="panel-box">
      <div class="panel-pill">
        <span>3. DỮ LIỆU VÀ PHÂN TÍCH DỮ LIỆU</span>
      </div>

      <!-- Hàng trên: 4 Biểu đồ đo lường thực tế -->
      <div class="grid grid-cols-4 gap-4 mb-5">
        <div class="bg-white p-2 rounded-xl border border-slate-200 shadow-2xs text-center">
          <img src="{b64_data['chart_unitree_returns.png']}" alt="Điểm thưởng seed 42" class="w-full h-auto rounded-lg">
          <span class="text-[11px] font-bold text-slate-700 mt-1 block">Điểm thưởng (Seed 42)</span>
        </div>
        <div class="bg-white p-2 rounded-xl border border-slate-200 shadow-2xs text-center">
          <img src="{b64_data['chart_unitree_multiseed.png']}" alt="Điểm thưởng 3 seed" class="w-full h-auto rounded-lg">
          <span class="text-[11px] font-bold text-slate-700 mt-1 block">Điểm thưởng (3 Seed)</span>
        </div>
        <div class="bg-white p-2 rounded-xl border border-slate-200 shadow-2xs text-center">
          <img src="{b64_data['chart_unitree_jerk.png']}" alt="Độ mượt mô-men khớp J" class="w-full h-auto rounded-lg">
          <span class="text-[11px] font-bold text-slate-700 mt-1 block">Độ mượt mô-men khớp J</span>
        </div>
        <div class="bg-white p-2 rounded-xl border border-slate-200 shadow-2xs text-center">
          <img src="{b64_data['waveforms_unitree.png']}" alt="Đuôi phân bố rung giật" class="w-full h-auto rounded-lg">
          <span class="text-[11px] font-bold text-slate-700 mt-1 block">Đuôi phân bố rung giật</span>
        </div>
      </div>

      <!-- Hàng dưới: 4 Thẻ KPI nổi bật -->
      <div class="grid grid-cols-4 gap-4">
        <!-- Thẻ 1 -->
        <div class="bg-blue-50/70 border-1.5 border-blue-200 p-4 rounded-2xl flex flex-col justify-between">
          <div>
            <span class="text-[11px] font-bold text-blue-800 uppercase block tracking-wider">ĐỘ TRỄ SUY LUẬN</span>
            <div class="text-2xl font-black text-blue-700 my-1 font-heading">0,38 ms</div>
          </div>
          <p class="text-[11px] text-slate-600 leading-snug">CUDA Graph GPU (2649 Hz); CPU đạt 2,41 ms (415 Hz). Đáp ứng tốt chuẩn thời gian thực 50 Hz (20 ms).</p>
        </div>

        <!-- Thẻ 2 -->
        <div class="bg-emerald-50/70 border-1.5 border-emerald-200 p-4 rounded-2xl flex flex-col justify-between">
          <div>
            <span class="text-[11px] font-bold text-emerald-800 uppercase block tracking-wider">RUNG GIẬT MÔ-MEN</span>
            <div class="text-2xl font-black text-emerald-700 my-1 font-heading">Giảm 14,4%</div>
          </div>
          <p class="text-[11px] text-slate-600 leading-snug">Chỉ số jerk J = 0,0628 so với 0,0733 của DT. Biên độ dao động mượt, giảm hao mòn động cơ.</p>
        </div>

        <!-- Thẻ 3 -->
        <div class="bg-purple-50/70 border-1.5 border-purple-200 p-4 rounded-2xl flex flex-col justify-between">
          <div>
            <span class="text-[11px] font-bold text-purple-800 uppercase block tracking-wider">KHÁNG XUNG LỰC</span>
            <div class="text-2xl font-black text-purple-700 my-1 font-heading">Mức xô 200</div>
          </div>
          <p class="text-[11px] text-slate-600 leading-snug">Duy trì bước đi và đạt điểm thưởng 1445,8 khi chịu lực xô ngang mức 200 (DT chỉ đạt 1207,0).</p>
        </div>

        <!-- Thẻ 4 -->
        <div class="bg-amber-50/70 border-1.5 border-amber-200 p-4 rounded-2xl flex flex-col justify-between">
          <div>
            <span class="text-[11px] font-bold text-amber-800 uppercase block tracking-wider">DUNG LƯỢNG MÔ HÌNH</span>
            <div class="text-2xl font-black text-amber-700 my-1 font-heading">4,98 MB</div>
          </div>
          <p class="text-[11px] text-slate-600 leading-snug">1,28 triệu tham số, chiếm dụng VRAM 0,3 GB. Thuận tiện nạp lên vi xử lý nhúng của robot.</p>
        </div>
      </div>
    </section>

    <!-- KHUNG 4: GIẢI THÍCH, KẾT LUẬN VÀ ĐIỂM MỚI -->
    <section class="panel-box">
      <div class="panel-pill">
        <span>4. GIẢI THÍCH, KẾT LUẬN VÀ ĐIỂM MỚI</span>
      </div>

      <!-- Ảnh thực nghiệm Unitree A1 xô ngang -->
      <div class="bg-slate-50 p-2.5 rounded-2xl border border-slate-200 mb-4 text-center">
        <img src="{b64_data['unitree_a1_kick_recovery_real.png']}" alt="Phục hồi thăng bằng robot Unitree A1 sau xung lực xô ngang" class="w-full h-auto rounded-xl">
        <p class="text-xs font-semibold text-slate-700 mt-2">
          Quá trình Unitree A1 phục hồi thăng bằng sau khi chịu xung lực xô ngang mức 200 (a-d)
        </p>
      </div>

      <!-- Cột dưới: 4 Tiêu chí & Điểm mới, Hạn chế, Kết luận, QR -->
      <div class="grid grid-cols-12 gap-5 items-stretch">
        <!-- Cột trái: 4 Tiêu chí nghiên cứu (6/12) -->
        <div class="col-span-6 bg-slate-50/80 p-4 rounded-2xl border border-slate-200 space-y-2.5 text-xs text-slate-700 leading-relaxed flex flex-col justify-between">
          <div>
            <h3 class="font-heading text-xs font-bold text-sky-950 uppercase border-b border-slate-200 pb-1.5 mb-2">
              KẾT QUẢ THEO 4 TIÊU CHÍ NGHIÊN CỨU
            </h3>
            <div class="space-y-2">
              <p>• <strong class="text-slate-900">Tiêu chí 1 (Thời gian thực):</strong> Độ trễ 0,38 ms trên GPU và 2,41 ms trên CPU, tần số hoạt động 415 - 2649 Hz, vượt chuẩn tối thiểu 50 Hz.</p>
              <p>• <strong class="text-slate-900">Tiêu chí 2 (Độ trơn chấp hành):</strong> Rung giật mô-men J = 0,0628, giảm 14,4% so với DT; tỷ lệ vượt ngưỡng rung giật chỉ 7,4% (so với 11,6% của DT).</p>
              <p>• <strong class="text-slate-900">Tiêu chí 3 (Khả năng thích ứng):</strong> Bộ điều hợp thích nghi robot mới dưới 1 phút chỉ với 3,8% tham số điều chỉnh.</p>
              <p>• <strong class="text-slate-900">Tiêu chí 4 (Độ bền kháng nhiễu):</strong> Hoàn thành episode và giữ điểm thưởng cao ở các mức xô đẩy đến 200; chỉ ngã ở mức xô cực hạn 400.</p>
            </div>
          </div>
        </div>

        <!-- Cột phải: Điểm mới, Hạn chế, Kết luận & QR Code (6/12) -->
        <div class="col-span-6 space-y-3 flex flex-col justify-between">
          <!-- Điểm mới & Hạn chế -->
          <div class="grid grid-cols-2 gap-3 text-xs leading-relaxed">
            <div class="bg-sky-50/60 p-3 rounded-xl border border-sky-100">
              <strong class="text-sky-900 font-bold block mb-1">ĐIỂM MỚI:</strong>
              <p class="text-slate-700">• Phân cấp 2 tầng: Tầng Mamba vĩ mô bắt phụ thuộc dài hạn kết hợp tầng Liquid vi mô làm mượt lực xoắn thời gian thực.</p>
              <p class="text-slate-700 mt-1">• Thích ứng đa hình thái: Cố định mạng xương sống, chỉ cần gắn bộ điều hợp nhẹ để mở rộng sang robot khác.</p>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-200">
              <strong class="text-slate-800 font-bold block mb-1">HẠN CHẾ:</strong>
              <p class="text-slate-600">• Kết quả thực nghiệm hiện được đo đạc trong môi trường mô phỏng vật lý MuJoCo.</p>
              <p class="text-slate-600 mt-1">• Triển khai chuyển giao Sim-to-Real trên phần cứng robot thực tế ngoài đời là hướng tiếp theo.</p>
            </div>
          </div>

          <!-- Kết luận + QR Code -->
          <div class="grid grid-cols-12 gap-3 items-center">
            <!-- Hộp Kết luận xanh lá (9/12) -->
            <div class="col-span-9 bg-emerald-50 border-1.5 border-emerald-300 p-3.5 rounded-2xl">
              <strong class="text-emerald-900 font-bold text-xs uppercase block mb-1">KẾT LUẬN:</strong>
              <p class="text-xs text-emerald-950 leading-relaxed font-medium">
                Kiến trúc phân cấp HDML đồng thời giải quyết bài toán độ trễ (0,38 ms), độ trơn chấp hành (giảm 14,4% jerk) và khả năng chịu ngoại lực xô đẩy trên robot Unitree A1 trong mô phỏng vật lý.
              </p>
            </div>

            <!-- Khối QR Code (3/12) -->
            <div class="col-span-3 bg-white p-2 rounded-2xl border border-slate-200 text-center shadow-2xs flex flex-col items-center justify-center">
              <img src="{b64_data['qr_code.png']}" alt="Mã QR" class="w-16 h-16 rounded-md">
              <span class="text-[9px] font-bold text-slate-700 mt-1 uppercase block leading-tight">QUÉT MÃ QR</span>
              <span class="text-[8px] text-slate-500 block leading-none">Video mô phỏng</span>
            </div>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- Script điều khiển hiển thị trên trình duyệt -->
  <script>
    let isFitted = false;
    function toggleFit() {{
      const container = document.getElementById('posterContainer');
      const btn = document.getElementById('fitBtn');
      if (!isFitted) {{
        const scale = (window.innerWidth - 32) / 1200;
        if (scale < 1) {{
          container.style.transform = `scale(${{scale}})`;
          container.style.transformOrigin = 'top center';
          btn.innerHTML = `
            <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>
            <span>Kích thước gốc (100%)</span>
          `;
          isFitted = true;
          return;
        }}
      }}
      container.style.transform = 'none';
      btn.innerHTML = `
        <svg class="w-4 h-4 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path></svg>
        <span>Vừa màn hình</span>
      `;
      isFitted = false;
    }}
  </script>
</body>
</html>
"""

    output_html.write_text(html_content, encoding="utf-8")
    output_b64_copy.write_text(html_content, encoding="utf-8")
    if artifact_dir and artifact_dir.exists():
        (artifact_dir / "poster_khkt.html").write_text(html_content, encoding="utf-8")

    print(f"Successfully generated standalone Base64 poster at: {output_html}")
    print(f"File size: {output_html.stat().st_size:,} bytes (~{output_html.stat().st_size / (1024 * 1024):.2f} MB)")

if __name__ == "__main__":
    main()

