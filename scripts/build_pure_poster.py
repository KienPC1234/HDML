#!/usr/bin/env python3
"""Build 100% Pure HTML/CSS Standee Poster with Base64 embedded assets.
Completely rewritten without AI cliches, without bullet point addiction,
with natural, human-written scientific Vietnamese that is easily understood
by both judges and general visitors.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_assets"
    output_html = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_khkt.html"
    output_b64 = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_khkt_base64.html"
    artifact_dir_env = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")
    artifact_dir = Path(artifact_dir_env) if artifact_dir_env else None

    images = {
        "a1_thumb.png": "image/png",
        "walker2d_raw.png": "image/png",
        "humanoid_raw.png": "image/png",
        "waveforms_unitree.png": "image/png",
        "hdml_architecture.png": "image/png",
        "chart_k3_returns.png": "image/png",
        "chart_k3_smoothness.png": "image/png",
        "chart_k3_robustness.png": "image/png",
        "chart_k3_latency.png": "image/png",
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
  <style>
    /* RESET & BASE STYLES */
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: #0f172a;
      color: #0f172a;
      line-height: 1.42;
      -webkit-font-smoothing: antialiased;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    /* POSTER CANVAS (Chuẩn Standee tỷ lệ 1:2 / 80x160cm / 1200x2400px) */
    .poster-canvas {{
      width: 1200px;
      min-height: 2400px;
      height: auto;
      background-color: #d8eef8;
      margin: 20px auto;
      padding: 22px 28px 24px 28px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
      position: relative;
    }}

    /* HEADER BANNER */
    .header-box {{
      text-align: center;
      margin-bottom: 2px;
    }}

    .header-bar {{
      background-color: #0c384a;
      color: #ffffff;
      border-radius: 10px;
      padding: 7px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      border: 1px solid #164e63;
    }}

    .header-bar-left {{
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.4px;
      color: #fef08a;
    }}

    .header-bar-sub {{
      color: #e2e8f0;
      font-weight: 600;
      margin-left: 6px;
    }}

    .header-bar-right {{
      background: rgba(14, 116, 144, 0.45);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #bae6fd;
      font-size: 12px;
      font-weight: 800;
      padding: 3px 12px;
      border-radius: 9999px;
      letter-spacing: 0.4px;
    }}

    .header-title {{
      font-size: 21.5px;
      font-weight: 900;
      color: #0a3347;
      line-height: 1.32;
      text-transform: uppercase;
      letter-spacing: -0.2px;
      padding: 0 10px;
    }}

    /* KHUNG THẺ CHÍNH */
    .panel-card {{
      background-color: #ffffff;
      border: 1.5px solid #bce1f0;
      border-radius: 18px;
      padding: 14px 18px 16px 18px;
      box-shadow: 0 4px 14px rgba(12, 74, 110, 0.05);
      position: relative;
    }}

    /* TIÊU ĐỀ KHUNG */
    .panel-pill {{
      background-color: #0284c7;
      color: #ffffff;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.4px;
      padding: 5px 18px;
      border-radius: 9999px;
      display: inline-block;
      margin-bottom: 11px;
      box-shadow: 0 2px 5px rgba(2, 132, 199, 0.25);
    }}

    /* KHUNG 1: BỐ CỤC */
    .k1-body {{
      display: flex;
      gap: 16px;
      align-items: stretch;
    }}

    .k1-left {{
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .k1-robots-grid {{
      display: flex;
      gap: 10px;
      background-color: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 12px;
      padding: 8px;
    }}

    .k1-robot-item {{
      flex: 1;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 6px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: space-between;
    }}

    .k1-robot-img-wrap {{
      height: 95px;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
    }}

    .k1-robot-img-wrap img {{
      max-height: 88px;
      max-width: 100%;
      object-fit: contain;
    }}

    .k1-robot-name {{
      font-size: 11px;
      font-weight: 800;
      color: #1e293b;
      margin-top: 4px;
    }}

    .k1-robot-sub {{
      font-size: 10px;
      color: #64748b;
    }}

    .k1-robot-note {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 11.2px;
      color: #334155;
      line-height: 1.45;
      margin-top: 8px;
    }}

    .k1-right {{
      flex: 1.25;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .k1-chart-box {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 3px;
      text-align: center;
    }}

    .k1-chart-box img {{
      width: 100%;
      height: auto;
      max-height: 230px;
      object-fit: contain;
      border-radius: 8px;
      display: block;
    }}

    .k1-problem-grid {{
      display: flex;
      gap: 10px;
      margin-top: 8px;
    }}

    .k1-problem-card {{
      flex: 1;
      background: #fef2f2;
      border: 1.2px solid #fecaca;
      border-radius: 10px;
      padding: 8px 11px;
      font-size: 11px;
      color: #991b1b;
      line-height: 1.40;
    }}

    .k1-problem-card strong {{
      color: #b91c1c;
      display: block;
      margin-bottom: 2px;
      font-size: 11.2px;
    }}

    .k1-solution-card {{
      flex: 1.1;
      background: #f0fdf4;
      border: 1.2px solid #bbf7d0;
      border-radius: 10px;
      padding: 8px 11px;
      font-size: 11px;
      color: #166534;
      line-height: 1.40;
    }}

    .k1-solution-card strong {{
      color: #15803d;
      display: block;
      margin-bottom: 2px;
      font-size: 11.2px;
    }}

    /* KHUNG 2: BỐ CỤC */
    .k2-body {{
      display: flex;
      gap: 16px;
      align-items: center;
    }}

    .k2-arch-box {{
      flex: 3.3;
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 12px;
      padding: 6px;
      text-align: center;
    }}

    .k2-arch-box img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      display: block;
    }}

    .k2-info-box {{
      flex: 1.1;
      display: flex;
      flex-direction: column;
      gap: 9px;
    }}

    .k2-step-card {{
      background: #f8fafc;
      border: 1.2px solid #e2e8f0;
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 11px;
      color: #334155;
      line-height: 1.42;
    }}

    .k2-step-card strong {{
      color: #0f172a;
      display: block;
      margin-bottom: 2px;
      font-size: 11.4px;
    }}

    /* KHUNG 3: BỐ CỤC */
    .k3-charts-grid {{
      display: flex;
      gap: 10px;
      margin-bottom: 11px;
    }}

    .k3-chart-card {{
      flex: 1;
      background: #ffffff;
      border: 1.2px solid #cbd5e1;
      border-radius: 10px;
      padding: 3px;
      text-align: center;
    }}

    .k3-chart-card img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
    }}

    .k3-cards-grid {{
      display: flex;
      gap: 10px;
    }}

    .k3-kpi-card {{
      flex: 1;
      border-radius: 12px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      border: 1.5px solid transparent;
    }}

    .k3-card-green {{
      background: #f0fdf4;
      border-color: #bbf7d0;
    }}
    .k3-card-green .kpi-tag {{ color: #166534; }}
    .k3-card-green .kpi-num {{ color: #15803d; }}

    .k3-card-teal {{
      background: #f0fdfa;
      border-color: #99f6e4;
    }}
    .k3-card-teal .kpi-tag {{ color: #0f766e; }}
    .k3-card-teal .kpi-num {{ color: #0d9488; }}

    .k3-card-purple {{
      background: #faf5ff;
      border-color: #e9d5ff;
    }}
    .k3-card-purple .kpi-tag {{ color: #6b21a8; }}
    .k3-card-purple .kpi-num {{ color: #7e22ce; }}

    .k3-card-blue {{
      background: #eff6ff;
      border-color: #bfdbfe;
    }}
    .k3-card-blue .kpi-tag {{ color: #1e40af; }}
    .k3-card-blue .kpi-num {{ color: #1d4ed8; }}

    .kpi-tag {{
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}

    .kpi-num {{
      font-size: 19.5px;
      font-weight: 900;
      margin: 2px 0;
      line-height: 1.1;
    }}

    .kpi-desc {{
      font-size: 10.5px;
      color: #475569;
      line-height: 1.35;
    }}

    /* KHUNG 4: BỐ CỤC */
    .k4-row-top {{
      display: flex;
      gap: 14px;
      align-items: stretch;
      margin-bottom: 11px;
    }}

    .k4-img-col {{
      flex: 1.25;
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 6px 8px;
      text-align: center;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}

    .k4-img-col img {{
      width: 100%;
      max-height: 245px;
      object-fit: contain;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
    }}

    .k4-caption {{
      font-size: 9.8px;
      font-weight: 700;
      color: #475569;
      margin-top: 4px;
      line-height: 1.3;
    }}

    .k4-top-right {{
      flex: 1.75;
      display: flex;
      flex-direction: column;
      gap: 8px;
      justify-content: space-between;
    }}

    /* KHỐI GIẢI THÍCH 3 CÂU HỎI */
    .k4-qa-box {{
      background: #f0fdf4;
      border: 1.5px solid #86efac;
      border-radius: 12px;
      padding: 9px 12px;
    }}

    .k4-qa-header {{
      color: #166534;
      font-size: 11.4px;
      font-weight: 900;
      text-transform: uppercase;
      border-bottom: 1px solid #bbf7d0;
      padding-bottom: 3px;
      margin-bottom: 6px;
      letter-spacing: 0.3px;
    }}

    .k4-qa-item {{
      font-size: 10.6px;
      color: #1e293b;
      line-height: 1.40;
      margin-bottom: 5px;
    }}

    .k4-qa-item:last-child {{
      margin-bottom: 0;
    }}

    .k4-qa-item strong {{
      color: #15803d;
    }}

    .k4-meta-grid {{
      display: flex;
      gap: 9px;
    }}

    .k4-novelty-box {{
      flex: 1.1;
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 10.5px;
      color: #0c4a6e;
      line-height: 1.40;
    }}

    .k4-novelty-box strong {{
      color: #0369a1;
      font-size: 11px;
      display: block;
      margin-bottom: 2px;
    }}

    .k4-limit-box {{
      flex: 0.9;
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 10.5px;
      color: #92400e;
      line-height: 1.40;
    }}

    .k4-limit-box strong {{
      color: #b45309;
      font-size: 11px;
      display: block;
      margin-bottom: 2px;
    }}

    /* HÀNG DƯỚI KHUNG 4: 4 CHỈ SỐ + KẾT LUẬN + QR */
    .k4-row-bottom {{
      display: flex;
      gap: 12px;
      align-items: stretch;
    }}

    .k4-badges-col {{
      flex: 1.8;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 6px;
    }}

    .k4-badges-title {{
      font-size: 11.2px;
      font-weight: 900;
      color: #0c384a;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}

    .k4-badges-row {{
      display: flex;
      gap: 8px;
    }}

    .k4-badge-item {{
      flex: 1;
      background: #f8fafc;
      border: 1.2px solid #cbd5e1;
      border-radius: 8px;
      padding: 6px 8px;
      text-align: center;
    }}

    .k4-badge-val {{
      font-size: 13px;
      font-weight: 900;
      color: #0f172a;
      display: block;
    }}

    .k4-badge-lbl {{
      font-size: 9.5px;
      color: #64748b;
      font-weight: 600;
    }}

    .k4-conclusion-box {{
      flex: 1.5;
      background: #eff6ff;
      border: 1.5px solid #93c5fd;
      border-radius: 12px;
      padding: 9px 12px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}

    .k4-conclusion-box strong {{
      color: #1e40af;
      font-size: 11px;
      font-weight: 900;
      display: block;
      margin-bottom: 3px;
      text-transform: uppercase;
    }}

    .k4-conclusion-box p {{
      font-size: 10.2px;
      color: #1e3a8a;
      line-height: 1.40;
    }}

    .k4-qr-box {{
      flex: 0.8;
      background: #ffffff;
      border: 1.5px solid #fbbf24;
      border-radius: 12px;
      padding: 6px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }}

    .k4-qr-box img {{
      width: 52px;
      height: 52px;
      display: block;
    }}

    .k4-qr-text {{
      font-size: 8px;
      font-weight: 800;
      color: #1e293b;
      margin-top: 3px;
      text-transform: uppercase;
      line-height: 1.15;
    }}

    .k4-qr-sub {{
      font-size: 7px;
      color: #64748b;
      line-height: 1;
    }}

    /* IN ẤN STANDARDIZED PRINT TO PDF */
    @media print {{
      @page {{
        size: 1200px 2400px;
        margin: 0;
      }}
      body {{
        background: #d8eef8 !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .poster-canvas {{
        margin: 0 auto !important;
        width: 1200px !important;
        min-height: 2400px !important;
        height: auto !important;
        padding: 22px 28px 24px 28px !important;
        box-shadow: none !important;
        gap: 14px !important;
      }}
    }}
  </style>
</head>
<body>

  <!-- POSTER CANVAS -->
  <div class="poster-canvas">

    <!-- PHẦN ĐẦU POSTER (HEADER) -->
    <div class="header-box">
      <div class="header-bar">
        <div class="header-bar-left">
          MÃ SỐ DỰ ÁN: ......... <span class="header-bar-sub">| CUỘC THI NCKH KỸ THUẬT (KHKT) HÀ NỘI 2026 - 2027</span>
        </div>
        <div class="header-bar-right">
          LĨNH VỰC: RÔ BỐT VÀ MÁY THÔNG MINH
        </div>
      </div>
      <h1 class="header-title">
        NGHIÊN CỨU PHÁT TRIỂN KIẾN TRÚC MÔ HÌNH NỀN TẢNG PHÂN CẤP MAMBA-LIQUID ĐIỀU KHIỂN CHUYỂN ĐỘNG ROBOT ĐA HÌNH THÁI VỚI ĐỘ TRỄ THẤP VÀ KHÁNG NHIỄU CAO
      </h1>
    </div>

    <!-- KHUNG 1: VẤN ĐỀ NGHIÊN CỨU -->
    <div class="panel-card">
      <div class="panel-pill">1. VẤN ĐỀ NGHIÊN CỨU</div>
      <div class="k1-body">
        <!-- Cột trái: 3 Robot -->
        <div class="k1-left">
          <div class="k1-robots-grid">
            <div class="k1-robot-item">
              <div class="k1-robot-img-wrap">
                <img src="{b64_data['a1_thumb.png']}" alt="Quadruped 12 DoF">
              </div>
              <div class="k1-robot-name">Quadruped (12 DoF)</div>
              <div class="k1-robot-sub">Robot 4 chân</div>
            </div>
            <div class="k1-robot-item">
              <div class="k1-robot-img-wrap">
                <img src="{b64_data['walker2d_raw.png']}" alt="Biped Walker 6 DoF">
              </div>
              <div class="k1-robot-name">Biped Walker (6 DoF)</div>
              <div class="k1-robot-sub">Robot 2 chân</div>
            </div>
            <div class="k1-robot-item">
              <div class="k1-robot-img-wrap">
                <img src="{b64_data['humanoid_raw.png']}" alt="Humanoid 17 DoF">
              </div>
              <div class="k1-robot-name">Humanoid (17 DoF)</div>
              <div class="k1-robot-sub">Robot hình người</div>
            </div>
          </div>
          <div class="k1-robot-note">
            <strong>Bài toán đa hình thái:</strong> Mỗi dòng robot có cấu trúc khớp và cách đi lại khác hẳn nhau (6 đến 17 bậc tự do). Hiện nay, đổi sang robot mới là phải thiết kế và dạy lại AI từ đầu, rất tốn thời gian và thiết bị tính toán.
          </div>
        </div>

        <!-- Cột phải: Đồ thị Đo Thực Nghiệm + Vấn đề thực tế & Giải pháp -->
        <div class="k1-right">
          <div class="k1-chart-box">
            <img src="{b64_data['waveforms_unitree.png']}" alt="Hình 3: Phân bố rung giật và rung giật tích lũy (HDML vs Transformer qua 998 bước)">
          </div>
          <div class="k1-problem-grid">
            <div class="k1-problem-card">
              <strong>Vấn đề kỹ thuật (Transformer):</strong>
              Tính toán O(L^2) trễ nhịp 50 Hz (20 ms); lệnh rời rạc sinh rung giật mô-men (torque chatter), gây mòn bánh răng và quá nhiệt động cơ.
            </div>
            <div class="k1-solution-card">
              <strong>Mục tiêu giải pháp (HDML):</strong>
              Phân cấp 2 tầng: Tầng Mamba tính nhanh O(L) lo chiến lược; tầng nơ-ron lỏng vuốt mượt tín hiệu liên tục để dập rung giật và bảo vệ khớp máy.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- KHUNG 2: THIẾT KẾ VÀ PHƯƠNG PHÁP NGHIÊN CỨU -->
    <div class="panel-card">
      <div class="panel-pill">2. THIẾT KẾ VÀ PHƯƠNG PHÁP NGHIÊN CỨU</div>
      <div class="k2-body">
        <div class="k2-arch-box">
          <img src="{b64_data['hdml_architecture.png']}" alt="Sơ đồ kiến trúc phân cấp HDML">
        </div>
        <div class="k2-info-box">
          <div class="k2-step-card">
            <strong>Môi trường thử nghiệm</strong>
            Mô phỏng động lực học trên MuJoCo với tần số điều khiển 50 Hz (20 ms mỗi bước), bước tích phân vật lý 0,002s để bám sát phản hồi thực tế của robot 4 chân Unitree A1.
          </div>
          <div class="k2-step-card">
            <strong>Dữ liệu & Huấn luyện</strong>
            Huấn luyện mô hình nền tảng trên 14 tập dữ liệu chuyển động (tổng cộng 6,56 triệu bước chuyển tiếp) bằng 1 card đồ họa RTX 4070 SUPER; bổ sung hàm phạt chống giật chuyển động.
          </div>
          <div class="k2-step-card">
            <strong>Quy trình đo kiểm</strong>
            Đánh giá điểm thưởng Unitree A1 qua 3 seed huấn luyện độc lập. Thử lực xô ngang với 3 episode mỗi mức lực, cùng seed; đo riêng độ trơn và độ trễ suy luận.
          </div>
        </div>
      </div>
    </div>

    <!-- KHUNG 3: DỮ LIỆU VÀ PHÂN TÍCH DỮ LIỆU -->
    <div class="panel-card">
      <div class="panel-pill">3. DỮ LIỆU VÀ PHÂN TÍCH DỮ LIỆU</div>
      <!-- Hàng trên: 4 Biểu đồ trực quan -->
      <div class="k3-charts-grid">
        <div class="k3-chart-card">
          <img src="{b64_data['chart_k3_returns.png']}" alt="Điểm thưởng bước đi 3 seed">
        </div>
        <div class="k3-chart-card">
          <img src="{b64_data['chart_k3_smoothness.png']}" alt="Độ mượt mô-men và an toàn cơ cấu">
        </div>
        <div class="k3-chart-card">
          <img src="{b64_data['chart_k3_robustness.png']}" alt="Kháng ngoại lực xô ngang theo các mức xô">
        </div>
        <div class="k3-chart-card">
          <img src="{b64_data['chart_k3_latency.png']}" alt="Độ trễ suy luận thời gian thực">
        </div>
      </div>

      <!-- Hàng dưới: 4 Thẻ KPI ánh xạ trực tiếp -->
      <div class="k3-cards-grid">
        <div class="k3-kpi-card k3-card-green">
          <div>
            <div class="kpi-tag">DÁNG ĐI VỮNG CHÃI</div>
            <div class="kpi-num">1466 Điểm</div>
          </div>
          <div class="kpi-desc">Dẫn đầu trong 6 mô hình được so sánh.</div>
        </div>

        <div class="k3-kpi-card k3-card-teal">
          <div>
            <div class="kpi-tag">ĐỘ MƯỢT KHỚP CHÂN</div>
            <div class="kpi-num">Giảm 14,4% Jerk</div>
          </div>
          <div class="kpi-desc">Tỷ lệ bước có chỉ số rung giật vượt 0,2 giảm từ 11,6% xuống 7,4%, tương đương giảm 36,2%.</div>
        </div>

        <div class="k3-kpi-card k3-card-purple">
          <div>
            <div class="kpi-tag">CHỊU LỰC XÔ NGANG</div>
            <div class="kpi-num">200 N trong 60 ms</div>
          </div>
          <div class="kpi-desc">Giữ vững bước đi đạt 1446 điểm (+239 điểm so với Transformer 1207). Độ nghiêng người tối đa chỉ 28,2° vs 33,2°.</div>
        </div>

        <div class="k3-kpi-card k3-card-blue">
          <div>
            <div class="kpi-tag">SUY LUẬN NHANH</div>
            <div class="kpi-num">0,38 ms</div>
          </div>
          <div class="kpi-desc">Suy luận GPU đạt 0,38 ms với CUDA Graph; ONNX đạt 2,41 ms trên CPU máy trạm Intel Xeon E5-2690 v4.</div>
        </div>
      </div>
    </div>

    <!-- KHUNG 4: GIẢI THÍCH, KẾT LUẬN VÀ ĐIỂM MỚI -->
    <div class="panel-card">
      <div class="panel-pill">4. GIẢI THÍCH, KẾT LUẬN VÀ ĐIỂM MỚI</div>
      <!-- Hàng trên: Ảnh thực nghiệm phục hồi + Khối giải thích cơ chế -->
      <div class="k4-row-top">
        <div class="k4-img-col">
          <img src="{b64_data['unitree_a1_kick_recovery_real.png']}" alt="Phục hồi thăng bằng robot Unitree A1 sau xung lực xô ngang">
          <div class="k4-caption">Thử nghiệm lực xô ngang 200 N trong 60 ms: robot chịu tác động, điều chỉnh tư thế và phục hồi dáng đi.</div>
        </div>
        <div class="k4-top-right">
          <!-- HỘP GIẢI THÍCH CƠ CHẾ DỄ HIỂU -->
          <div class="k4-qa-box">
            <div class="k4-qa-header">Giải thích cơ chế động học</div>
            <div class="k4-qa-item">
              <strong>Tốc độ suy luận:</strong> CUDA Graph giảm chi phí gọi các phép tính GPU, đưa độ trễ từ 7,75 ms xuống 0,38 ms trong cùng phép đo.
            </div>
            <div class="k4-qa-item">
              <strong>Độ trơn hành động:</strong> HDML giảm 14,4% chỉ số rung giật so với Decision Transformer. Kết quả này đánh giá toàn bộ kiến trúc, chưa tách riêng đóng góp của CfC khi va chạm.
            </div>
            <div class="k4-qa-item">
              <strong>Khả năng chịu va chạm:</strong> ở mức 200 N trong 60 ms, HDML hoàn thành các episode thử nghiệm và có góc nghiêng đỉnh thấp hơn đối chứng; cả hai cùng ngã ở mức 400 N.
            </div>
          </div>

          <div class="k4-meta-grid">
            <div class="k4-novelty-box">
              <strong>ĐIỂM MỚI NỔI BẬT:</strong>
              Kết hợp Mamba, Flow Matching, CfC và giám sát PACE trong kiến trúc phân cấp; tinh chỉnh bộ điều hợp với khoảng 3,5–3,8% tham số để dự đoán hành động trên các hình thái robot.
            </div>
            <div class="k4-limit-box">
              <strong>HẠN CHẾ THỰC TẾ:</strong>
              Toàn bộ dữ liệu hiện được đo đạc trong môi trường mô phỏng vật lý MuJoCo. Chưa kiểm chứng điều khiển vòng kín trên hình thái mới và chưa đo trên CPU nhúng.
            </div>
          </div>
        </div>
      </div>

      <!-- Hàng dưới: 4 Chỉ số tóm tắt + Kết luận + QR -->
      <div class="k4-row-bottom">
        <div class="k4-badges-col">
          <div class="k4-badges-title">Tóm tắt kết quả theo 4 tiêu chuẩn đề ra</div>
          <div class="k4-badges-row">
            <div class="k4-badge-item">
              <span class="k4-badge-val">0,38 ms</span>
              <span class="k4-badge-lbl">Suy luận GPU / CUDA Graph</span>
            </div>
            <div class="k4-badge-item">
              <span class="k4-badge-val">-14,4%</span>
              <span class="k4-badge-lbl">Giảm giật khớp toàn thân</span>
            </div>
            <div class="k4-badge-item">
              <span class="k4-badge-val">&lt; 1 Phút</span>
              <span class="k4-badge-lbl">Tinh chỉnh dự đoán hành động</span>
            </div>
            <div class="k4-badge-item">
              <span class="k4-badge-val">200 N · 60 ms</span>
              <span class="k4-badge-lbl">Hoàn thành 3/3 episode</span>
            </div>
          </div>
        </div>

        <div class="k4-conclusion-box">
          <strong>KẾT LUẬN</strong>
          <p>Mô hình phân cấp HDML giải quyết đồng thời 3 bài toán lớn: phản xạ thời gian thực, bước đi êm ái bảo vệ khớp máy và đứng vững trước va chạm, mở ra hướng ứng dụng linh hoạt cho nhiều dòng robot chân.</p>
        </div>

        <div class="k4-qr-box">
          <img src="{b64_data['qr_code.png']}" alt="Mã QR">
          <div class="k4-qr-text">Quét mã QR</div>
          <div class="k4-qr-sub">Mã nguồn & Video</div>
        </div>
      </div>
    </div>

  </div>

</body>
</html>
"""

    output_html.write_text(html_content, encoding="utf-8")
    output_b64.write_text(html_content, encoding="utf-8")
    if artifact_dir and artifact_dir.exists():
        (artifact_dir / "poster_khkt.html").write_text(html_content, encoding="utf-8")

    print(f"Successfully generated clean human-written poster at: {output_html}")
    print(f"File size: {output_html.stat().st_size:,} bytes (~{output_html.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
