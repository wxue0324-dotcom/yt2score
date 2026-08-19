# yt2score

把 YouTube 連結變成樂譜。在本機執行，音訊與運算都不離開你的電腦。

```
YouTube URL
   → yt-dlp 下載音訊
   → librosa 分析速度 / 拍號 / 調性
   → Demucs 分軌（人聲 / 鼓 / 貝斯 / 其他）
   → basic-pitch 逐軌採譜 → 量化到節拍格線
   → music21 組譜 → MuseScore 排版與合成
   → 五線譜 PDF・簡譜・MusicXML・MIDI・試聽 MP3
```

## 啟動

```bash
./start.sh
# 開 http://127.0.0.1:8420
```

貼上連結，等它跑完。3 分鐘的歌在 M1 上大約 1–2 分鐘。

## 輸出

| 檔案 | 用途 |
|---|---|
| `score.pdf` | 五線譜總譜，可直接列印 |
| `score.musicxml` | **最重要**：用 MuseScore / Sibelius / Logic 開啟修改 |
| `score.mid` | MIDI，可丟進 DAW |
| `jianpu.html` | 簡譜，瀏覽器開啟後用「列印 → 存成 PDF」 |
| `score.mp3` | **試聽**：照著譜合成的聲音，用來檢查採譜對不對 |
| `part*.mp3` | 分聲部試聽，可單獨聽旋律或鋼琴 |
| `stems/htdemucs/*.wav` | 原始分軌音檔，和上面的合成音對照用 |

## 下載失敗時

先跑環境檢查，它會分辨「你的環境壞了」和「YouTube 擋這支片」：

```bash
./venv/bin/python doctor.py
```

### 程式如何應付 YouTube 的封鎖

YouTube 要求客戶端解開一個 JS 簽章挑戰（「n challenge」），解不開就會拿到
403 或看不到音訊格式。程式已預設處理兩件事：

1. **自動下載 yt-dlp 的挑戰求解器**（`--remote-components ejs:github`）。
   首次執行會從 yt-dlp 官方 GitHub 抓一份 JS 腳本，需要能連外網。
   另外需要 `deno` 當 JS runtime（`brew install deno`）。
2. **格式階梯**：YouTube 常常對高音質格式回 403，卻放行最低音質那支。
   程式會依 `bestaudio → 140 → 251 → 18 → 139 → worstaudio` 逐級退讓，
   同時輪替多種 client，直到某個組合真的抓得下來。

實測結果：原本硬性 403 的影片，靠退到 format 139（49kbps）就能取得。
**退到低音質時介面會明確警告**，因為那確實會拉低分軌與採譜的準確度。

### 什麼時候才需要 cookies

只有在整個階梯都失敗時 —— 通常是年齡限制、地區限制或會員限定影片。
也可以用來拿高音質版本（避免被迫退到 139）：

```bash
export YT2SCORE_COOKIES_FROM_BROWSER=chrome   # 或 chrome:"Profile 1"
./start.sh
```

首次讀取 Chrome cookies 時 macOS 會跳出鑰匙圈授權，**必須由你本人按「一律允許」**，
所以第一次請在終端機手動跑一次：

```bash
./venv/bin/yt-dlp --cookies-from-browser chrome \
  --simulate --print '%(title)s' 'https://www.youtube.com/watch?v=0mAuj1rtx6k'
```

不想碰鑰匙圈的話，改用瀏覽器擴充套件匯出的 cookies.txt：

```bash
export YT2SCORE_COOKIES_FILE=/path/to/cookies.txt
```

## 試聽：怎麼判斷採得準不準

介面上有兩組播放器，這是驗收採譜品質最快的方法：

- **試聽採譜結果**（譜面上方）— 由 MuseScore 照著譜合成，聽到的就是譜上寫的音
- **分聲部試聽**（頁尾左側）— 只聽旋律、只聽鋼琴、只聽鼓
- **原始音軌**（頁尾右側）— 從原曲分離出來的真實錄音

把同一個聲部的合成音和原始音軌輪流播（一次只會播一個），差異在哪一聽就知道，
再回頭到 MuseScore 裡修那幾個小節。

音色由 MuseScore 內建音源合成，全程離線，不需要瀏覽器外掛。

**主唱聲部在試聽時用鋼琴音色播放。** MuseScore 合成的人聲是一團沒有起音的
「啊」聲，聽不出音準對不對；鋼琴每個音都有清楚的音頭，錯音一聽就發現。
五線譜、PDF 與 MusicXML 仍標示為 Vocal，不會損失「這是人聲旋律」的資訊。

## 準確度：請把它當草稿

這是機器聽寫，不是定稿。實測下來：

- **單一旋律線**（人聲、獨奏）最準，大致堪用
- **鋼琴/伴奏**中等，和聲走向通常對，內聲部細節常錯
- **鼓組**只做到 kick / snare / hihat 三分類，是節奏示意
- **調性判斷**約八成準；信心低於 0.6 時介面會提示
- **速度**抓得到「一個」拍子，但不保證是你會寫在譜上的那個

工作流程建議：拿 `score.musicxml` 到 MuseScore 裡修，比從零開始抄快得多。

### 速度為什麼特別容易錯

一首連續十六分音符的慢曲，和一首四倍快的曲子，**音符落點完全一樣**。光看 onset
位置，「拍子是四分音符、曲子裡有十六分音符」和「拍子本來就那麼快」在數學上無法
區分——這是聽感問題，不是訊號問題。實測過 onset 自相關、chroma 自相似、
tempogram ratio 三種訊號，都無法可靠分辨。

所以程式用一個**速度先驗**（`analyze.py` 的 `_TEMPO_CENTRE`）在證據相當時偏向
聽起來合理的拍速。這對多數曲子有效，但慢板獨奏和三連音曲目仍常被讀快。

譜上速度不對時，在 MuseScore 裡改速度記號比重跑一次快。怎麼判斷見下節。

### 確認速度對不對

把節拍器點擊聲混進原曲，聽哪個踩得準：

```bash
cd eval
../venv/bin/python click.py ../work/<id>     # 產生各候選速度的試聽檔
```

它用兩套追蹤器的**真實節拍位置**（不是等距格線）產生 mp3，所以連相位和彈性速度
都聽得出來。點擊聲從頭到尾都踩在拍子上的那個就是答案。

第二套追蹤器 `beat_this` 是選用的，沒裝就只出現 librosa 的結果：

```bash
./venv/bin/pip install beat_this      # 首次執行下載 77MB 模型，之後離線
```

系統會自動略過能量過低的音軌——純演奏曲不會硬生出一條假的主唱旋律。

## 量測準確度

改採譜參數很容易「感覺變好了」，所以這裡有一套可量測的基準：用已知的樂譜由
MuseScore 合成音訊當標準答案，跑完流程後用 `mir_eval` 算音符層級的 F1
（起始點在容差內且音高正確才算命中）。

```bash
cd eval
../venv/bin/python benchmark.py --label myrun    # 音符 F1 基準
../venv/bin/python compare.py                    # 改動前後對照
../venv/bin/python sweep_on_stems.py             # 掃描採譜參數
../venv/bin/python tempo_bench.py --label myrun --compare baseline   # 速度與拍號
```

速度另外量，因為它的對錯不是連續的。60 被讀成 120 和被讀成 119，絕對誤差都是
「差 59」，但前者整份譜的時值都寫成一半，後者只是彈快了一點。所以
`tempo_bench.py` 按**比值**分類（2 倍、一半、3 倍），只有比值接近 1 才算對。
它同時檢查一個不變式：譜頭寫的速度，和量化用的節拍格線，必須描述同一首曲子。

分成兩階段量測，這個區分很重要：

- **階段 A**：對「單一聲部各自合成」的乾淨音檔採譜。不經過分軌，完全可重現，
  調參數時該看這個。
- **階段 B**：完整流程跑混音，含 Demucs。基礎 htdemucs 模型還算穩定（σ≈0.007），
  但 `htdemucs_ft` 實測完全不可重現，別用它比較。

**合成音訊比真實錄音乾淨，絕對數值偏樂觀。** 它可靠的是*相對*變化——某個改動
到底有沒有幫助。而且只有三個測試案例，容易過擬合；調參數時要看平台區而不是尖峰。

同樣重要的是：**音符 F1 不等於譜面品質**。速度抓成一半的譜，音符起始點分數可能
差不多，但每個音的時值都寫成兩倍，當作譜是錯的。兩者要分開看。

## 調整參數

| 想改什麼 | 檔案 |
|---|---|
| 採譜靈敏度（各軌的 onset 門檻、音域） | `backend/pipeline/transcribe.py` 的 `PROFILES` |
| 節奏精細度（預設 8 分音符格線） | `backend/pipeline/quantize.py` 的 `SUBDIVISION` |
| 音軌保留門檻（預設佔全曲能量 8%） | `backend/pipeline/separate.py` 的 `PRESENCE_THRESHOLD` |
| 左右手分界 | `backend/pipeline/quantize.py` 的 `split_hands` |
| 簡譜排版 | `backend/pipeline/jianpu.py` |
| 歌曲長度上限（預設 12 分鐘） | `backend/pipeline/download.py` 的 `max_duration` |
| 速度先驗（證據相當時偏向的拍速） | `backend/pipeline/analyze.py` 的 `_TEMPO_CENTRE` |

## 環境需求

macOS（已在 M1 測試）、Python 3.11、ffmpeg、MuseScore 4、deno（yt-dlp 解 YouTube 簽章用，缺了會大量 403）。

```bash
brew install python@3.11 ffmpeg deno
brew install --cask musescore
python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

## 使用須知

下載 YouTube 影音牴觸 YouTube 服務條款，多數音樂也受著作權保護。自己抓來練習、學習、研究是一回事；散布產出的樂譜或音檔是另一回事。這個工具只在本機跑，不提供公開服務，請自行斟酌使用範圍。

## CLI

不想開網頁時：

```bash
./venv/bin/python -m pipeline.run "<youtube-url>" work/mysong
# 從 backend/ 目錄執行
```
