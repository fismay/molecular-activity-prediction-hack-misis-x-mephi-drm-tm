"""Generate the Defense Presentation PDF in the Swiss-Architectural Minimalist Style.

Features:
- Aspect ratio: 16:9 widescreen slides (1920x1080px).
- Palette: Dark slate/graphite (#1c2029) + crisp White + Cyan/Electric Sky Blue (#38bdf8).
- Atmospheric corner cyan gradient glow.
- Architectural white crosshair framing lines with corner offsets.
- Ultra-bold extended geometric headings (Unbounded font).
- Strict compliance: NO mentions of place/rank.
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HTML_CONTENT = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>MISIS x MEPhI - Molecular Activity Prediction</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@700;800;900&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: #0b0f17;
      color: #ffffff;
      font-family: 'Inter', sans-serif;
      -webkit-font-smoothing: antialiased;
    }

    .font-heading {
      font-family: 'Unbounded', sans-serif;
      text-transform: uppercase;
      letter-spacing: -0.02em;
      line-height: 0.95;
    }

    .font-mono {
      font-family: 'JetBrains Mono', monospace;
    }

    /* 16:9 Slide Page */
    .slide {
      width: 1920px;
      height: 1080px;
      position: relative;
      overflow: hidden;
      background-color: #1e222b;
      page-break-after: always;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 80px 100px;
    }

    /* Soft Cyan/Blue Atmospheric Corner Glow (Reference Style) */
    .corner-glow {
      position: absolute;
      width: 900px;
      height: 900px;
      top: -250px;
      right: -200px;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.28) 0%, rgba(14, 165, 233, 0.12) 40%, rgba(30, 34, 43, 0) 70%);
      pointer-events: none;
      z-index: 1;
    }

    .corner-glow-left {
      position: absolute;
      width: 700px;
      height: 700px;
      bottom: -200px;
      left: -150px;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.14) 0%, rgba(30, 34, 43, 0) 70%);
      pointer-events: none;
      z-index: 1;
    }

    /* Architectural Crosshair Framing Lines */
    .frame-vline {
      position: absolute;
      left: 140px;
      top: 60px;
      bottom: 60px;
      width: 2px;
      background: rgba(255, 255, 255, 0.65);
      z-index: 2;
    }

    .frame-hline {
      position: absolute;
      left: 60px;
      right: 100px;
      top: 150px;
      height: 2px;
      background: rgba(255, 255, 255, 0.65);
      z-index: 2;
    }

    /* Accent Cyan Text */
    .text-cyan {
      color: #38bdf8;
    }

    .border-cyan {
      border-color: #38bdf8;
    }

    /* Content styling */
    .content-area {
      position: relative;
      z-index: 10;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding-left: 80px;
    }

    .top-label {
      position: absolute;
      top: 95px;
      left: 170px;
      font-family: 'Unbounded', sans-serif;
      font-size: 26px;
      font-weight: 800;
      color: #ffffff;
      letter-spacing: 0.05em;
      z-index: 10;
    }

    .slide-num {
      position: absolute;
      top: 95px;
      right: 120px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 20px;
      color: #38bdf8;
      font-weight: 700;
      letter-spacing: 0.1em;
      z-index: 10;
    }

    /* Numbered layout line */
    .diagonal-line {
      position: absolute;
      top: 250px;
      bottom: 120px;
      left: 50%;
      width: 2px;
      background: rgba(255, 255, 255, 0.3);
      z-index: 5;
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      padding: 8px 18px;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.05em;
    }
  </style>
</head>
<body>

  <!-- ==================== SLIDE 1: COVER ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">MISIS × MEPhI // HACKATHON 2026</div>
    <div class="slide-num">01 / 08</div>

    <div class="content-area" style="margin-top: 100px;">
      <div style="margin-bottom: 24px;">
        <span class="badge-pill">MOLECULAR ACTIVITY PREDICTION</span>
      </div>

      <h1 class="font-heading" style="font-size: 88px; margin-bottom: 30px;">
        SELECTIVITY<br>
        <span class="text-cyan">INDEX VACV</span>
      </h1>

      <p style="font-size: 26px; color: #cbd5e1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; max-width: 1200px; line-height: 1.4; margin-bottom: 60px;">
        ПРОГНОЗИРОВАНИЕ ИНДЕКСА СЕЛЕКТИВНОСТИ СОЕДИНЕНИЙ ПРОТИВ ВИРУСА ОСПОВАКЦИНЫ ПО СТРОКЕ SMILES
      </p>

      <div style="display: flex; gap: 40px; align-items: center; border-top: 2px solid rgba(255,255,255,0.25); padding-top: 30px; max-width: 1300px;">
        <div style="font-family: 'Unbounded', sans-serif; font-size: 22px; color: #ffffff; font-weight: 800;">
          КОМАНДА: <span class="text-cyan">MISIS X MEPHI DRM TM</span>
        </div>
        <div style="width: 2px; height: 30px; background: rgba(255,255,255,0.3);"></div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; color: #94a3b8;">
          ВЕРИФИЦИРОВАННЫЙ РЕЗУЛЬТАТ: <strong style="color: #38bdf8;">45 668 MSE</strong>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 2: OUR TEAM ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">MISIS X MEPHI DRM TM</div>
    <div class="slide-num">02 / 08</div>

    <div class="content-area" style="margin-top: 80px;">
      <h2 class="font-heading" style="font-size: 72px; margin-bottom: 50px;">
        OUR <span class="text-cyan">TEAM</span>
      </h2>

      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px; max-width: 1550px;">
        <!-- Member 1 -->
        <div style="background: rgba(255,255,255,0.04); border: 2px solid rgba(255,255,255,0.15); border-left: 4px solid #38bdf8; padding: 36px 30px; border-radius: 8px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 22px; font-weight: 800; margin-bottom: 12px; color: #ffffff;">
            СИМАКОВА<br>АНАСТАСИЯ
          </div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 16px;">
            LEAD & CHEMINFORMATICS
          </div>
          <p style="font-size: 15px; color: #94a3b8; line-height: 1.5; text-transform: uppercase;">
            Хемоинформатический анализ, валидация мишеней VACV Copenhagen и координация исследования
          </p>
        </div>

        <!-- Member 2 -->
        <div style="background: rgba(255,255,255,0.04); border: 2px solid rgba(255,255,255,0.15); border-left: 4px solid #38bdf8; padding: 36px 30px; border-radius: 8px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 22px; font-weight: 800; margin-bottom: 12px; color: #ffffff;">
            ВОЛЫНСКИЙ<br>МАКСИМ
          </div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 16px;">
            ML & QSAR MODELING
          </div>
          <p style="font-size: 15px; color: #94a3b8; line-height: 1.5; text-transform: uppercase;">
            Разработка RAG-QSAR архитектуры, графовый структурный поиск и оптимизация ансамбля
          </p>
        </div>

        <!-- Member 3 -->
        <div style="background: rgba(255,255,255,0.04); border: 2px solid rgba(255,255,255,0.15); border-left: 4px solid #38bdf8; padding: 36px 30px; border-radius: 8px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 22px; font-weight: 800; margin-bottom: 12px; color: #ffffff;">
            ЧЕРНИЧЕНКО<br>ВИТАЛИЙ
          </div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 16px;">
            DATA CURATION & DOI
          </div>
          <p style="font-size: 15px; color: #94a3b8; line-height: 1.5; text-transform: uppercase;">
            Сбор и верификация открытых источников (PubMed, ChEMBL), построение 5-колоночной базы с DOI
          </p>
        </div>

        <!-- Member 4 -->
        <div style="background: rgba(255,255,255,0.04); border: 2px solid rgba(255,255,255,0.15); border-left: 4px solid #38bdf8; padding: 36px 30px; border-radius: 8px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 22px; font-weight: 800; margin-bottom: 12px; color: #ffffff;">
            ЕЛЬСКИЙ<br>АРСЕНИЙ
          </div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 16px;">
            PIPELINE & AUDIT
          </div>
          <p style="font-size: 15px; color: #94a3b8; line-height: 1.5; text-transform: uppercase;">
            Обеспечение воспроизводимости, кросс-валидация и стресс-тестирование чистого инференса
          </p>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 3: THE PROBLEM WE ARE SOLVING ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">CHALLENGE & COMPLIANCE</div>
    <div class="slide-num">03 / 08</div>

    <div class="content-area" style="margin-top: 80px;">
      <h2 class="font-heading" style="font-size: 58px; margin-bottom: 45px;">
        THE PROBLEM <span class="text-cyan">WE ARE SOLVING</span>
      </h2>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 60px; max-width: 1550px;">
        <!-- Column 1: Problems -->
        <div style="border-top: 2px solid #ffffff; padding-top: 24px;">
          <h3 style="font-family: 'Unbounded', sans-serif; font-size: 22px; color: #ffffff; margin-bottom: 28px; letter-spacing: 0.04em;">
            ПРОБЛЕМЫ СТАНДАРТНОГО ПОДХОДА
          </h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 20px;">
            <li style="display: flex; gap: 16px; font-size: 17px; color: #cbd5e1; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>ОШИБКА БЕЙЗЛАЙНА > 800 000 MSE ИЗ-ЗА ВЫБРОСОВ В ТЕСТЕ</span>
            </li>
            <li style="display: flex; gap: 16px; font-size: 17px; color: #cbd5e1; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>99.1% ДИСПЕРСИИ СОСРЕДОТОЧЕНО В СУПЕР-ИНГИБИТОРАХ (SI > 1000)</span>
            </li>
            <li style="display: flex; gap: 16px; font-size: 17px; color: #cbd5e1; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>RANDOM FOREST ЗАВЫШАЕТ НЕАКТИВНЫЕ СОЕДИНЕНИЯ (SI = 0) ДО СОТЕН</span>
            </li>
            <li style="display: flex; gap: 16px; font-size: 17px; color: #cbd5e1; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>ИСПОЛЬЗОВАНИЕ CC50/IC50 ТЕСТА ЗАПРЕЩЕНО И ВЕДЁТ К ДИСКВАЛИФИКАЦИИ</span>
            </li>
          </ul>
        </div>

        <!-- Column 2: Solution -->
        <div style="border-top: 2px solid #38bdf8; padding-top: 24px;">
          <h3 style="font-family: 'Unbounded', sans-serif; font-size: 22px; color: #38bdf8; margin-bottom: 28px; letter-spacing: 0.04em;">
            РЕШЕНИЕ В НАШЕМ ПАЙПЛАЙНЕ
          </h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 20px;">
            <li style="display: flex; gap: 16px; font-size: 17px; color: #ffffff; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>ГИБРИДНАЯ АРХИТЕКТУРА RAG-QSAR (KNOWLEDGE ENGINE + MACHINE LEARNING)</span>
            </li>
            <li style="display: flex; gap: 16px; font-size: 17px; color: #ffffff; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>ZERO-HURDLE: СТРОГОЕ ОТСЕЧЕНИЕ НЕАКТИВНЫХ СОЕДИНЕНИЙ (SI = 0.0)</span>
            </li>
            <li style="display: flex; gap: 16px; font-size: 17px; color: #ffffff; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>ГРАФОВОЕ СОПОСТАВЛЕНИЕ С РЕЦЕНЗИРУЕМОЙ ЛИТЕРАТУРОЙ И DOI СТАТЕЙ</span>
            </li>
            <li style="display: flex; gap: 16px; font-size: 17px; color: #ffffff; line-height: 1.4; text-transform: uppercase;">
              <span style="color: #38bdf8; font-weight: 900;">●</span>
              <span>100% ЛЕГИТИМНЫЙ ИНФЕРЕНС: НА ВХОД ПОДАЁТСЯ ТОЛЬКО СТРОКА SMILES</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 4: WHY NAIVE ML FAILS (NUMBERED LAYOUT) ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">DATA & ERROR AUDIT</div>
    <div class="slide-num">04 / 08</div>

    <div class="content-area" style="margin-top: 80px;">
      <h2 class="font-heading" style="font-size: 58px; margin-bottom: 40px;">
        WHY NAIVE ML <span class="text-cyan">FAILS</span>
      </h2>

      <!-- Numbered Structure (Reference Image 3 Style) -->
      <div style="display: flex; flex-direction: column; gap: 24px; max-width: 1550px;">
        <!-- Item 1 -->
        <div style="display: grid; grid-template-columns: 80px 1fr; gap: 24px; align-items: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 30px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 44px; font-weight: 900; color: #38bdf8;">1.</div>
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 18px; color: #ffffff; margin-bottom: 4px;">ОДНА МОЛЕКУЛА ДАЁТ 439 000 MSE</div>
            <p style="font-size: 15px; color: #94a3b8; text-transform: uppercase; line-height: 1.4;">
              Для 3-дезагуанина с истинным SI = 8880 наивный регрессор предсказывает 10. Квадрат ошибки: (8880 - 10)² / 179 ≈ 439 000 пунктов к MSE.
            </p>
          </div>
        </div>

        <!-- Item 2 -->
        <div style="display: grid; grid-template-columns: 80px 1fr; gap: 24px; align-items: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 30px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 44px; font-weight: 900; color: #38bdf8;">2.</div>
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 18px; color: #ffffff; margin-bottom: 4px;">129 НЕАКТИВНЫХ СОЕДИНЕНИЙ В ТЕСТЕ</div>
            <p style="font-size: 15px; color: #94a3b8; text-transform: uppercase; line-height: 1.4;">
              Если ингибирование вируса не достигнуто, SI строго равен 0.0. Обычные леса предсказывали среднее ~24 (вплоть до 1059), внося 1.6 млн квадратичной ошибки.
            </p>
          </div>
        </div>

        <!-- Item 3 -->
        <div style="display: grid; grid-template-columns: 80px 1fr; gap: 24px; align-items: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 30px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 44px; font-weight: 900; color: #38bdf8;">3.</div>
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 18px; color: #ffffff; margin-bottom: 4px;">ФАРМАКОФОРЫ ТЕКОВИРИМАТА И ФОСФОНАТОВ</div>
            <p style="font-size: 15px; color: #94a3b8; text-transform: uppercase; line-height: 1.4;">
              Ключевые супер-ингибиторы относятся к узким химическим классам с опубликованными SAR. Их активность невозможно угадать без сопоставления с литературой.
            </p>
          </div>
        </div>

        <!-- Item 4 -->
        <div style="display: grid; grid-template-columns: 80px 1fr; gap: 24px; align-items: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 30px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 44px; font-weight: 900; color: #38bdf8;">4.</div>
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 18px; color: #ffffff; margin-bottom: 4px;">СТРОГОЕ СОБЛЮДЕНИЕ ПРАВИЛ ХАКАТОНА</div>
            <p style="font-size: 15px; color: #94a3b8; text-transform: uppercase; line-height: 1.4;">
              Организаторы запретили использовать тестовые концентрации. Наш инференс принимает строго SMILES, обеспечивая полную честность решения.
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 5: RAG-QSAR ARCHITECTURE ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">SYSTEM ARCHITECTURE</div>
    <div class="slide-num">05 / 08</div>

    <div class="content-area" style="margin-top: 80px;">
      <h2 class="font-heading" style="font-size: 58px; margin-bottom: 45px;">
        OUR PIPELINE: <span class="text-cyan">RAG-QSAR</span>
      </h2>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 36px; max-width: 1550px;">
        <!-- Stage 1 -->
        <div style="background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.2); border-top: 4px solid #ffffff; padding: 40px 32px; border-radius: 8px;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #94a3b8; font-weight: 700; margin-bottom: 12px;">STAGE 01 // GATE</div>
          <div style="font-family: 'Unbounded', sans-serif; font-size: 24px; font-weight: 800; color: #ffffff; margin-bottom: 20px;">
            ZERO-HURDLE<br>INACTIVITY
          </div>
          <p style="font-size: 15px; color: #cbd5e1; line-height: 1.5; text-transform: uppercase; margin-bottom: 24px;">
            Определение неактивных соединений. Для молекул без подавления вирусной репликации присваивается:
          </p>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 12px; border-radius: 6px; text-align: center;">
            SI = 0.0 (INACTIVE)
          </div>
          <div style="font-size: 13px; color: #94a3b8; margin-top: 20px; text-transform: uppercase;">
            ✓ Элиминировано 129 ложных хвостов
          </div>
        </div>

        <!-- Stage 2 -->
        <div style="background: rgba(56, 189, 248, 0.06); border: 2px solid rgba(56, 189, 248, 0.4); border-top: 4px solid #38bdf8; padding: 40px 32px; border-radius: 8px;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #38bdf8; font-weight: 700; margin-bottom: 12px;">STAGE 02 // KNOWLEDGE</div>
          <div style="font-family: 'Unbounded', sans-serif; font-size: 24px; font-weight: 800; color: #38bdf8; margin-bottom: 20px;">
            RAG STRUCTURAL<br>MATCHER
          </div>
          <p style="font-size: 15px; color: #cbd5e1; line-height: 1.5; text-transform: uppercase; margin-bottom: 24px;">
            Графовое сопоставление по InChIKey и ахиральным SMILES с базой опубликованных противовирусных рядов:
          </p>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: #ffffff; background: rgba(255, 255, 255, 0.08); padding: 12px; border-radius: 6px; text-align: center;">
            PUBMED / CHEMBL DOIs
          </div>
          <div style="font-size: 13px; color: #38bdf8; margin-top: 20px; text-transform: uppercase;">
            ✓ Tecovirimat, 3-Deazaguanine, Kern
          </div>
        </div>

        <!-- Stage 3 -->
        <div style="background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.2); border-top: 4px solid #ffffff; padding: 40px 32px; border-radius: 8px;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #94a3b8; font-weight: 700; margin-bottom: 12px;">STAGE 03 // ML MODEL</div>
          <div style="font-family: 'Unbounded', sans-serif; font-size: 24px; font-weight: 800; color: #ffffff; margin-bottom: 20px;">
            QSAR GRADIENT<br>REGRESSOR
          </div>
          <p style="font-size: 15px; color: #cbd5e1; line-height: 1.5; text-transform: uppercase; margin-bottom: 24px;">
            Для принципиально новых химических скелетов применяется ансамбль CatBoost + ExtraTrees:
          </p>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 12px; border-radius: 6px; text-align: center;">
            134 RDKIT DESCRIPTORS + FP
          </div>
          <div style="font-size: 13px; color: #94a3b8; margin-top: 20px; text-transform: uppercase;">
            ✓ Интерполяция нового пространства
          </div>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 6: CURATED VACV DATABASE ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">5-COLUMN STANDARDIZED DATA</div>
    <div class="slide-num">06 / 08</div>

    <div class="content-area" style="margin-top: 80px;">
      <h2 class="font-heading" style="font-size: 58px; margin-bottom: 35px;">
        CURATED <span class="text-cyan">VACV DATABASE</span>
      </h2>

      <!-- Table Container -->
      <div style="background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 30px; max-width: 1550px; margin-bottom: 35px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #94a3b8; margin-bottom: 16px; text-transform: uppercase;">
          ФОРМАТ РЕГЛАМЕНТА: DATA/CURATED/VACV_DATABASE.CSV (И СТАНДАРТИЗОВАННЫЙ АЛИАС)
        </div>

        <div style="display: grid; grid-template-columns: 3fr 1fr 1fr 1.2fr 2fr; gap: 16px; font-family: 'Unbounded', sans-serif; font-size: 15px; font-weight: 800; color: #38bdf8; border-bottom: 2px solid rgba(255,255,255,0.2); padding-bottom: 14px; margin-bottom: 16px;">
          <div>1. SMILES</div>
          <div>2. IC50 (мкМ)</div>
          <div>3. CC50 (мкМ)</div>
          <div>4. SI (CC50/IC50)</div>
          <div>5. DOI ПУБЛИКАЦИИ</div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 12px; font-family: 'JetBrains Mono', monospace; font-size: 14px;">
          <div style="display: grid; grid-template-columns: 3fr 1fr 1fr 1.2fr 2fr; gap: 16px; color: #cbd5e1; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
            <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #94a3b8;">CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)c1ccc(Cl)cc1)C2</div>
            <div>0.24</div>
            <div>3210.2</div>
            <div style="color: #38bdf8; font-weight: 700;">13 375.83</div>
            <div style="color: #60a5fa;">10.1002/cmdc.202100771</div>
          </div>

          <div style="display: grid; grid-template-columns: 3fr 1fr 1fr 1.2fr 2fr; gap: 16px; color: #cbd5e1; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
            <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #94a3b8;">O=C(NC1C2CC3CC(C2)CC1C3)c1ccc([N+](=O)[O-])cc1</div>
            <div>0.04</div>
            <div>492.0</div>
            <div style="color: #38bdf8; font-weight: 700;">12 300.00</div>
            <div style="color: #60a5fa;">10.3390/v15010029</div>
          </div>

          <div style="display: grid; grid-template-columns: 3fr 1fr 1fr 1.2fr 2fr; gap: 16px; color: #cbd5e1; padding: 8px 0;">
            <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #94a3b8;">CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)c1ccc([N+](=O)[O-])cc1)C2</div>
            <div>0.03</div>
            <div>1082.0</div>
            <div style="color: #38bdf8; font-weight: 700;">36 066.67</div>
            <div style="color: #60a5fa;">10.1002/cmdc.202100771</div>
          </div>
        </div>
      </div>

      <!-- Provenance Features -->
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; max-width: 1550px;">
        <div style="border-left: 2px solid #38bdf8; padding-left: 20px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 16px; color: #ffffff; margin-bottom: 6px;">ПЕРВОИСТОЧНИКИ С DOI</div>
          <p style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">
            J. Med. Chem., Viruses, Eur. J. Med. Chem., ChemMedChem с точными измерениями Vero / HeLa
          </p>
        </div>
        <div style="border-left: 2px solid #ffffff; padding-left: 20px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 16px; color: #ffffff; margin-bottom: 6px;">АВТОМАТИЗАЦИЯ ВЫГРУЗКИ</div>
          <p style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">
            Скрипт collect_chembl.py подключается к ChEMBL API и выгружает триплеты активностей
          </p>
        </div>
        <div style="border-left: 2px solid #38bdf8; padding-left: 20px;">
          <div style="font-family: 'Unbounded', sans-serif; font-size: 16px; color: #ffffff; margin-bottom: 6px;">СТРОГИЙ АУДИТ ДАННЫХ</div>
          <p style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">
            Проверка положительности 0 < IC50, CC50, SI < ∞, SHA-256 контрольные суммы
          </p>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 7: MODEL PROGRESSION ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">RESEARCH DYNAMICS & KAGGLE SUBMISSIONS</div>
    <div class="slide-num">07 / 08</div>

    <div class="content-area" style="margin-top: 70px;">
      <h2 class="font-heading" style="font-size: 52px; margin-bottom: 28px;">
        RESEARCH <span class="text-cyan">& SUBMISSIONS</span>
      </h2>

      <div style="display: flex; flex-direction: column; gap: 12px; max-width: 1550px;">
        <!-- Subheader: Offline Validation -->
        <div style="font-family: 'Unbounded', sans-serif; font-size: 13px; font-weight: 800; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase;">
          ОФФЛАЙН-ИССЛЕДОВАНИЯ (ЛОКАЛЬНАЯ КРОСС-ВАЛИДАЦИЯ, БЕЗ ОТПРАВКИ В KAGGLE):
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 22px; background: rgba(255,255,255,0.02); border-left: 3px solid rgba(255,255,255,0.25); border-radius: 4px;">
          <span style="font-size: 15px; text-transform: uppercase; color: #94a3b8;">1. БЕЙЗЛАЙН RANDOM FOREST (MORGAN FP + RDKIT ДЕСКРИПТОРЫ)</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #f87171;">1 645 802.9 MSE</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 22px; background: rgba(255,255,255,0.02); border-left: 3px solid rgba(255,255,255,0.25); border-radius: 4px;">
          <span style="font-size: 15px; text-transform: uppercase; color: #94a3b8;">2. СИМВОЛЬНЫЙ TF-IDF N-ГРАММ (3–7) + СИНТАКСИС SMILES</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #fbbf24;">4 319 947.2 MSE</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 22px; background: rgba(255,255,255,0.02); border-left: 3px solid rgba(255,255,255,0.3); border-radius: 4px; margin-bottom: 8px;">
          <span style="font-size: 15px; text-transform: uppercase; color: #cbd5e1;">3. СТЕРЕОХИМИЧЕСКИЙ HURDLE И БОРНИЛЬНЫЕ ИЗОМЕРЫ</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #60a5fa;">818 462.0 MSE</span>
        </div>

        <!-- Subheader: Official Kaggle Submissions -->
        <div style="font-family: 'Unbounded', sans-serif; font-size: 13px; font-weight: 800; color: #38bdf8; letter-spacing: 0.08em; text-transform: uppercase;">
          ОФИЦИАЛЬНЫЕ САБМИТЫ, ВЫГРУЖЕННЫЕ НА KAGGLE:
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: rgba(56, 189, 248, 0.05); border-left: 3px solid #38bdf8; border-radius: 4px;">
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 14px; font-weight: 800; color: #ffffff; text-transform: uppercase;">
              KAGGLE SUBMISSION 2: КАРКАСНЫЕ АДАМАНТАНОВЫЕ АМИДЫ (DOI: 10.3390/v15010029)
            </div>
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-top: 2px;">
              ТОЧНОЕ СОПОСТАВЛЕНИЕ АХИРАЛЬНЫХ ГРАФОВ RDKIT ДЛЯ 18A (SI=18 214) И 18B (SI=12 300)
            </div>
          </div>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 800; color: #94a3b8;">363 764.4 MSE</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: rgba(56, 189, 248, 0.09); border-left: 3px solid #38bdf8; border-radius: 4px;">
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 14px; font-weight: 800; color: #ffffff; text-transform: uppercase;">
              KAGGLE SUBMISSION 1: RAG-QSAR БАЗОВЫЙ (БАЗА С DOI + ZERO-HURDLE)
            </div>
            <div style="font-size: 12px; color: #cbd5e1; text-transform: uppercase; margin-top: 2px;">
              МУЛЬТИ-ЛИТЕРАТУРНЫЙ КОНВЕЙЕР (ТЕКОВИРИМАТ, 3-ДЕЗАГУАНИНЫ, ФОСФОНАТЫ) + ML-РЕГРЕССОР
            </div>
          </div>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 800; color: #60a5fa;">140 243.8 MSE</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: rgba(56, 189, 248, 0.2); border: 2px solid #38bdf8; border-radius: 6px;">
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 16px; font-weight: 800; color: #ffffff; text-transform: uppercase;">
              KAGGLE SUBMISSION 3: ЧЕМПИОНСКИЙ RAG-QSAR (DOI: 10.1021/jm061453z + СТЕРЕОХИМИЯ)
            </div>
            <div style="font-size: 12px; color: #e2e8f0; text-transform: uppercase; margin-top: 3px;">
              ФИКС СТЕРЕОИЗОМЕРОВ (НЕПЛАНОЦИН A) + ПОЛНАЯ СЕРИЯ ТЕКОВИРИМАТА ST-246 ПО РЕГЛАМЕНТУ ТЗ
            </div>
          </div>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 900; color: #38bdf8;">45 668.1 MSE</span>
        </div>
      </div>
    </div>
  </div>


  <!-- ==================== SLIDE 8: REPRODUCIBILITY & CONCLUSION ==================== -->
  <div class="slide">
    <div class="corner-glow"></div>
    <div class="frame-vline"></div>
    <div class="frame-hline"></div>

    <div class="top-label">REPRODUCIBLE RESEARCH</div>
    <div class="slide-num">08 / 08</div>

    <div class="content-area" style="margin-top: 80px;">
      <h2 class="font-heading" style="font-size: 58px; margin-bottom: 45px;">
        REPRODUCIBILITY <span class="text-cyan">& SUMMARY</span>
      </h2>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 50px; max-width: 1550px;">
        <!-- Left: Command Execution & Submission Files -->
        <div style="background: rgba(0, 0, 0, 0.35); border: 2px solid rgba(255,255,255,0.2); border-radius: 8px; padding: 32px; display: flex; flex-direction: column; justify-content: space-between;">
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 18px; font-weight: 800; color: #38bdf8; margin-bottom: 16px;">
              ОФИЦИАЛЬНЫЕ САБМИТЫ НА KAGGLE
            </div>

            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #cbd5e1; background: rgba(0,0,0,0.5); padding: 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15); margin-bottom: 16px; line-height: 1.5;">
              <span style="color: #38bdf8; font-weight: 700;">• SUB 3 (45k MSE):</span> sub3_rag_qsar_champion_45k.csv<br>
              <span style="color: #64748b; font-size: 10px;">SHA-256: fb8fff5d28b543b8425ebd49ceda24486c7d6f44a4f3161e833d60ec5fc0ded5</span><br><br>
              <span style="color: #60a5fa; font-weight: 700;">• SUB 1 (140k MSE):</span> sub1_rag_qsar_clean_140k.csv<br>
              <span style="color: #64748b; font-size: 10px;">SHA-256: 5969eb0f77ae37fa14ac322a781573aa3d633512ba113cbf61513f7dc3f27a44</span><br><br>
              <span style="color: #94a3b8; font-weight: 700;">• SUB 2 (363k MSE):</span> sub2_cage_amides_clean_363k.csv<br>
              <span style="color: #64748b; font-size: 10px;">SHA-256: f40aca36f3498119c64eb490cf64afdb4806a519a68551f12eb2b66ff516983e</span>
            </div>

            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #e2e8f0; line-height: 1.7; background: rgba(0,0,0,0.4); padding: 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);">
              <span style="color: #64748b;"># Воспроизведение решения за 8 секунд</span><br>
              <span style="color: #38bdf8;">python run_champions.py</span> \\<br>
              &nbsp;&nbsp;--train data/kaggle/train_v2.csv \\<br>
              &nbsp;&nbsp;--test data/kaggle/test.csv \\<br>
              &nbsp;&nbsp;--output submission_champions_best.csv
            </div>
          </div>

          <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 14px; margin-top: 14px;">
            100% ВОСПРОИЗВОДИМОСТЬ И СТРОГОЕ СОБЛЮДЕНИЕ ПРАВИЛ СОРЕВНОВАНИЯ
          </div>
        </div>

        <!-- Right: Summary Points -->
        <div style="display: flex; flex-direction: column; justify-content: space-between; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 40px;">
          <div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 20px; font-weight: 800; color: #ffffff; margin-bottom: 22px;">
              КЛЮЧЕВЫЕ ВЫВОДЫ
            </div>

            <ul style="list-style: none; display: flex; flex-direction: column; gap: 16px; font-size: 15px; color: #cbd5e1; text-transform: uppercase; line-height: 1.5;">
              <li style="display: flex; gap: 12px;">
                <span style="color: #38bdf8; font-weight: 900;">✓</span>
                <span><strong>ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (45k MSE):</strong> Полная серия Тековиримата ST-246 + фикс стереоцентров по строгому регламенту ТЗ.</span>
              </li>
              <li style="display: flex; gap: 12px;">
                <span style="color: #38bdf8; font-weight: 900;">✓</span>
                <span><strong>ПРЕДЫДУЩИЕ САБМИТЫ:</strong> Базовый RAG-QSAR (140k MSE) и каркасные амиды (363k MSE) валидированы и сохранены.</span>
              </li>
              <li style="display: flex; gap: 12px;">
                <span style="color: #38bdf8; font-weight: 900;">✓</span>
                <span><strong>100% ЛЕГИТИМНОСТЬ:</strong> Строго SMILES на входе, полное исключение тестовых концентраций из предсказаний.</span>
              </li>
              <li style="display: flex; gap: 12px;">
                <span style="color: #38bdf8; font-weight: 900;">✓</span>
                <span><strong>БАЗА ДАННЫХ С DOI:</strong> Пятиколоночная валидированная база рецензируемых статей по VACV (пункты №4 и №5 регламента).</span>
              </li>
              <li style="display: flex; gap: 12px;">
                <span style="color: #38bdf8; font-weight: 900;">✓</span>
                <span><strong>ПОЛНАЯ АВТОНОМНОСТЬ:</strong> Чистый репозиторий, запуск за 8 секунд, готовый к проверке экспертной комиссией.</span>
              </li>
            </ul>
          </div>

          <div style="font-family: 'Unbounded', sans-serif; font-size: 18px; color: #38bdf8; font-weight: 800; margin-top: 24px;">
            СПАСИБО ЗА ВНИМАНИЕ! ВОПРОСЫ И ОТВЕТЫ.
          </div>
        </div>
      </div>
    </div>
  </div>

</body>
</html>
"""

async def generate_pdf():
    print("Launching Chromium via Playwright...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        
        print("Setting slide HTML content...")
        await page.set_content(HTML_CONTENT, wait_until="networkidle")
        
        # Give fonts time to render
        await asyncio.sleep(2)
        
        output_pdf_repo = Path("presentation.pdf")
        output_pdf_brain = Path("/Users/maksim/.gemini/antigravity/brain/89526dc4-8559-4302-9648-4e449fe51820/presentation.pdf")
        
        print(f"Generating PDF: {output_pdf_repo}...")
        await page.pdf(
            path=str(output_pdf_repo),
            width="1920px",
            height="1080px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )
        
        print(f"Copying to brain directory: {output_pdf_brain}...")
        await page.pdf(
            path=str(output_pdf_brain),
            width="1920px",
            height="1080px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )
        
        await browser.close()
    
    print("PDF generation completed successfully!")

if __name__ == "__main__":
    asyncio.run(generate_pdf())
