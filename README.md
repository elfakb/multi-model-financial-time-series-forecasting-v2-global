# multi-model-finansal-zaman-serisi-tahmini-v2-global

ABD hisse senetleri ve kripto para birimleri üzerinde LSTM, GRU, XGBoost, ARIMA, SARIMA, Prophet ve GARCH modellerini karşılaştıran, Streamlit arayüzü ve MLflow deney takibi içeren çok modelli finansal zaman serisi tahmin sistemi.

> **v1 (Türk Varlıkları):** [multi-model-financial-time-series-forecasting](https://github.com/elfakb/multi-model-financial-time-series-forecasting)

---

## Demo

https://youtu.be/agyiSmGjb6A

---

## Proje Özeti

Bu proje ABD hisse senetleri ve kripto hisseleri kullanarak aşağıdaki konular üzerine çözüm araştırılmıştır:

- Kısa veri penceresi (6 ay) mi, uzun veri penceresi (5 yıl) mi daha iyi tahmin üretir?
- Derin öğrenme modelleri (LSTM, GRU) istatistiksel modellere (ARIMA, SARIMA) kıyasla ne kadar avantaj sağlar?
- Ensemble yaklaşımı tek bir modelden daha iyi performans gösterir mi?
- Aynı modeller USD bazlı varlıklarda TRY bazlı varlıklara kıyasla nasıl performans gösterir?

---

## v1 vs v2

| | v1 — Türk Varlıkları | v2 — Global Varlıklar |
|--|----------------------|----------------------|
| Varlıklar | Türk hisse senetleri + Altın | ABD hisse senetleri + Kripto |
| Para birimi | TRY | USD |
| Borsa | Borsa İstanbul | NASDAQ / NYSE / Kripto |
| Veri kaynağı | Yahoo Finance | Yahoo Finance |

---

## Veri Toplama

Tüm finansal veriler `yfinance` kütüphanesi aracılığıyla Yahoo Finance'den otomatik olarak çekildi. Her varlık için Open, High, Low, Close, Volume (OHLCV) formatında günlük frekansta veri indirildi.

Üç farklı veri penceresi kullanıldı: son 6 ay, son 1 yıl ve son 5 yıl. Her pencere ayrı bir klasörde saklandı:

```
data/raw/6m/
data/raw/1y/
data/raw/5y/
```

Her varlık ayrı bir CSV dosyası olarak kaydedildi.

---

## Varlık Listesi

| Sembol | Açıklama | Kaynak |
|--------|----------|--------|
| AAPL | Apple Inc. | Yahoo Finance |
| TSLA | Tesla Inc. | Yahoo Finance |
| NVDA | NVIDIA Corporation | Yahoo Finance |
| GOOGL | Alphabet Inc. | Yahoo Finance |
| MSFT | Microsoft Corporation | Yahoo Finance |
| AMZN | Amazon.com Inc. | Yahoo Finance |
| META | Meta Platforms Inc. | Yahoo Finance |
| BTC-USD | Bitcoin | Yahoo Finance |
| ETH-USD | Ethereum | Yahoo Finance |

---

## Eksik Veri

Kısa boşluklar (1-2 günlük eksiklikler) için lineer interpolasyon uygulandı. Uzun boşluklar (hafta sonu, resmi tatil, borsa kapalı günler) için forward-fill (ffill) yöntemi kullanıldı: bir önceki işlem gününün değeri olduğu gibi taşındı. Borsa kapalıyken fiyat değişmez, dolayısıyla interpolasyon yapmak yapay bir hareket üretir.

---

## Feature Engineering

Ham OHLCV verisine ek olarak 13 teknik indikatör hesaplandı ve her varlığın işlenmiş veri setine eklendi. Bu indikatörler `ta` kütüphanesi kullanılarak üretildi.

**Momentum göstergeleri:**
RSI , MACD ve MACD Signal.

**Volatilite göstergeleri:**
Bollinger Bands , ATR 

**Trend göstergeleri:**
SMA 20, SMA 50, EMA 20, EMA 50.

**Hacim göstergesi:**
OBV (On Balance Volume).

---

## Modeller

| Model | Tür | Kullanım Amacı |
|-------|-----|----------------|
| ARIMA | İstatistiksel | Durağan zaman serisi tahmini |
| SARIMA | İstatistiksel | Mevsimsel zaman serisi tahmini |
| Prophet | İstatistiksel/ML | Trend + mevsimsellik ayrıştırma |
| LSTM | Derin Öğrenme | Uzun vadeli bağımlılık öğrenme |
| GRU | Derin Öğrenme | LSTM'e alternatif, daha hızlı eğitim |
| XGBoost | Makine Öğrenmesi | Lag feature tabanlı tahmin |
| GARCH | Ekonometrik | Volatilite tahmini + güven bandı |
| Ensemble | Hibrit | Modellerin ağırlıklı ortalaması |

---

## Veri Pencereleri ve Tahmin Ufukları

| Veri Penceresi | Eğitim Verisi | Tahmin Ufukları |
|----------------|---------------|-----------------|
| 6 Ay | Son 180 gün | 5, 20, 60 gün |
| 1 Yıl | Son 365 gün | 5, 20, 60 gün |
| 5 Yıl | Son 1825 gün | 30, 60, 90 gün |

---

## Model Eğitimi

Her model, her varlık ve her veri penceresi kombinasyonu için bağımsız olarak eğitildi. Toplam **9 varlık × 3 veri penceresi × 6 model = 162 ayrı eğitim** gerçekleştirildi.

**Train/Test ayrımı:** Verinin ilk %80'i eğitim, son %20'si test için kullanıldı.

**Scaler (LSTM ve GRU):** MinMaxScaler yalnızca train seti üzerinde fit edildi, test setine ise sadece transform uygulandı. Veri sızıntısı (data leakage) bu şekilde önlendi.

**Lag feature (XGBoost):** Geçmiş 60 günün kapanış fiyatı ayrı sütunlar olarak (lag_1 ... lag_60) modele verildi. Feature oluşturma train ve test setleri için ayrı ayrı yapıldı.

**MLflow:** Her eğitim oturumu otomatik olarak MLflow'a loglandı. Model adı, varlık sembolü, veri penceresi, hiperparametreler ve RMSE/MAE/MAPE metrikleri kaydedildi. Modeller `models/saved/{period}/{asset}_{model}.pkl` veya `.pt` formatında fiziksel olarak da saklandı.

---

## Değerlendirme Metrikleri

- **RMSE** — Büyük hatalara daha duyarlı
- **MAE** — Ortalama mutlak hata
- **MAPE** — Ölçekten bağımsız, farklı fiyat seviyelerindeki varlıkları karşılaştırmak için en uygun metrik

> **GARCH notu:** GARCH fiyat değil, getiri oynaklığını (volatilite) tahmin ettiğinden MAPE değerleri anlamsız çıkmaktadır. GARCH yalnızca Streamlit arayüzünde volatilite güven bandı üretmek için kullanılmıştır.

---

## Model Performans Sonuçları

### En İyi Model — Varlık Bazlı (MAPE'ye göre)

| Varlık | En İyi Model | Veri Penceresi | MAPE |
|--------|-------------|----------------|------|
| AAPL | GRU | 6 Ay | %2.46 |
| TSLA | LSTM | 6 Ay | %3.20 |
| NVDA | LSTM | 6 Ay | %1.94 |
| GOOGL | GRU | 6 Ay | %2.83 |
| MSFT | XGBoost | 5 Yıl | %2.69 |
| AMZN | GRU | 6 Ay | %2.61 |
| META | XGBoost | 5 Yıl | %3.31 |
| BTC | GRU | 5 Yıl | %3.18 |
| ETH | XGBoost | 5 Yıl | %3.02 |

### LSTM Sonuçları (MAPE)

| Varlık | 6 Ay | 1 Yıl | 5 Yıl |
|--------|------|-------|-------|
| AAPL | %8.07 | %11.56 | %11.49 |
| TSLA | %3.20 | %3.54 | %29.42 |
| NVDA | %1.94 | %11.05 | %17.38 |
| GOOGL | %3.04 | %14.12 | %12.28 |
| MSFT | %6.25 | %9.76 | %16.96 |
| AMZN | %2.68 | %7.25 | %13.79 |
| META | %9.62 | %9.42 | %22.86 |
| BTC | %7.66 | %14.37 | %4.67 |
| ETH | %13.06 | %22.03 | %13.29 |

### GRU Sonuçları (MAPE)

| Varlık | 6 Ay | 1 Yıl | 5 Yıl |
|--------|------|-------|-------|
| AAPL | %2.46 | %8.14 | %2.67 |
| TSLA | %3.26 | %3.26 | %17.63 |
| NVDA | %1.96 | %10.20 | %10.19 |
| GOOGL | %2.83 | %7.58 | %8.23 |
| MSFT | %4.98 | %4.10 | %5.54 |
| AMZN | %2.61 | %5.41 | %3.79 |
| META | %7.65 | %6.60 | %3.85 |
| BTC | %4.38 | %3.87 | %3.18 |
| ETH | %4.41 | %13.97 | %5.20 |

### XGBoost Sonuçları (MAPE, sadece 5 yıl)

| Varlık | RMSE | MAE | MAPE |
|--------|------|-----|------|
| AAPL | 34.90 | 30.00 | %10.71 |
| TSLA | 39.42 | 33.68 | %7.80 |
| NVDA | 58.06 | 56.46 | %29.15 |
| GOOGL | 130.64 | 124.36 | %38.00 |
| MSFT | 16.97 | 12.41 | %2.69 |
| AMZN | 16.10 | 11.68 | %4.75 |
| META | 27.09 | 21.28 | %3.31 |
| BTC | 5114.32 | 3783.86 | %4.21 |
| ETH | 117.68 | 84.91 | %3.02 |

---

## Temel Bulgular

**1. GRU kısa vadeli tahminde baskın:** GRU, 6 aylık pencerede büyük çoğunlukla diğer tüm modelleri geride bıraktı. Özellikle AAPL, GOOGL ve AMZN gibi istikrarlı büyük ölçekli hisselerde en düşük MAPE değerlerini üretti.

**2. XGBoost 5 yıllık veride belirli varlıklarda öne çıktı:** v1'deki Türk varlıklarının aksine, USD bazlı varlıklar 5 yıllık pencerede daha fazla yapısal istikrar gösterdi. XGBoost MSFT (%2.69), META (%3.31) ve ETH (%3.02) için en iyi MAPE değerini elde etti.

**3. Kripto varlıklar beklentilerin üzerinde performans gösterdi:** BTC ve ETH, yüksek volatilitelerine rağmen %3-5 arasında rekabetçi MAPE değerleri üretti. GRU ile 5 yıllık BTC tahmini %3.18 MAPE ile en iyi sonucu verdi.

**4. LSTM uzun veri pencerelerinde zorlandı:** LSTM, tüm varlıklarda ve pencerelerde GRU'nun gerisinde kaldı. Fark 5 yıllık veride en belirgin şekilde ortaya çıktı — örneğin META için LSTM %22.86 MAPE'ye ulaşırken GRU %3.85'te kaldı.

**5. v1 vs v2 karşılaştırması:** USD bazlı global varlıklar genel olarak TRY bazlı Türk varlıklarına kıyasla daha düşük MAPE değerleri üretti. ABD hisse senetlerinin 5 yıllık fiyat yapısındaki istikrar XGBoost'un rekabetçi kalmasına olanak tanırken, Türk varlıklarında enflasyon kaynaklı fiyat bozulmaları nedeniyle kısa veri pencereleri zorunlu hale geldi.

---

## Kullanılan Kütüphaneler

```
Python 3.13
PyTorch 2.3
XGBoost
statsmodels
pmdarima
Prophet
arch
scikit-learn
yfinance
ta
MLflow
Streamlit
Plotly
```

---

## Kurulum

```bash
git clone https://github.com/elfakb/multi-model-finansal-zaman-serisi-tahmini-v2-global
cd multi-model-finansal-zaman-serisi-tahmini-v2-global
pip install -r requirements.txt
```

---

## Kullanım

```bash
# 1. Veri çek
python -m src.data.fetch_data

# 2. Temizle
python -m src.data.preprocess

# 3. Teknik indikatörler ekle
python -m src.data.technical_indicators

# 4. Modelleri eğit
python -m src.models.arima_model
python -m src.models.sarima_model
python -m src.models.prophet_model
python -m src.models.lstm_model
python -m src.models.gru_model
python -m src.models.xgboost_model
python -m src.models.garch_model

# 5. MLflow arayüzü
mlflow ui

# 6. Streamlit arayüzü
streamlit run app.py
```

---

## Proje Yapısı

```
multi-model-finansal-zaman-serisi-tahmini-v2-global/
├── data/
│   ├── raw/              # Ham OHLCV verisi (6m / 1y / 5y)
│   └── processed/        # Temizlenmiş + teknik indikatörlü veri
├── models/
│   └── saved/            # Eğitilmiş model dosyaları
├── src/
│   ├── data/
│   ├── models/
│   ├── ensemble/
│   └── evaluation/
├── streamlit_app/
│   └── app.py
├── reports/
│   ├── figures/
│   └── model_comparisons/
├── mlruns/
└── requirements.txt

```

# multi-model-financial-time-series-forecasting-v2-global

A multi-model financial time series forecasting system for global equities and cryptocurrencies, comparing LSTM, GRU, XGBoost, ARIMA, SARIMA, Prophet, and GARCH models with a Streamlit dashboard and MLflow experiment tracking.

> **v1 (Turkish Assets):** [multi-model-financial-time-series-forecasting](https://github.com/elfakb/multi-model-financial-time-series-forecasting)

---

## Demo

https://youtu.be/agyiSmGjb6A

---

## Project Overview

This project applies the same multi-model forecasting framework from v1 to global assets — US equities and cryptocurrencies — to answer:

- Does a shorter historical window (6 months) or a longer one (5 years) produce more accurate forecasts?
- How much advantage do deep learning models (LSTM, GRU) provide compared to traditional statistical models (ARIMA, SARIMA)?
- Can an ensemble approach outperform individual forecasting models?
- How do the same models perform on USD-denominated assets compared to TRY-denominated assets?

---

## v1 vs v2

| | v1 — Turkish Assets | v2 — Global Assets |
|--|---------------------|---------------------|
| Assets | Turkish equities + Gold | US equities + Crypto |
| Currency | TRY | USD |
| Exchange | Borsa Istanbul | NASDAQ / NYSE / Crypto |
| Data source | Yahoo Finance | Yahoo Finance |

---

## Data Collection

All financial data were automatically downloaded from Yahoo Finance using the `yfinance` library. Daily OHLCV (Open, High, Low, Close, Volume) data were collected for each asset.

Three different historical windows were used:

- Last 6 months
- Last 1 year
- Last 5 years

Each period was stored separately under:

```
data/raw/6m/
data/raw/1y/
data/raw/5y/
```

Each asset is saved as an individual CSV file.

---

## Asset List

| Symbol | Description | Source |
|--------|-------------|--------|
| AAPL | Apple Inc. | Yahoo Finance |
| TSLA | Tesla Inc. | Yahoo Finance |
| NVDA | NVIDIA Corporation | Yahoo Finance |
| GOOGL | Alphabet Inc. | Yahoo Finance |
| MSFT | Microsoft Corporation | Yahoo Finance |
| AMZN | Amazon.com Inc. | Yahoo Finance |
| META | Meta Platforms Inc. | Yahoo Finance |
| BTC-USD | Bitcoin | Yahoo Finance |
| ETH-USD | Ethereum | Yahoo Finance |

---

## Missing Value Handling

**Short gaps (1–2 trading days)** were filled using **linear interpolation**, where missing values were estimated from the neighboring observations.

**Long gaps** caused by weekends, public holidays, or market closures were filled using **forward-fill (ffill)**. Since no trading occurs during market closures, carrying the previous closing price forward is common practice and avoids introducing artificial price movements.

---

## Feature Engineering

In addition to the original OHLCV variables, **13 technical indicators** were calculated using the `ta` library.

### Momentum Indicators
- RSI (Relative Strength Index, 14-day window)
- MACD (Moving Average Convergence Divergence)
- MACD Signal

### Volatility Indicators
- Bollinger Bands (Upper, Middle, Lower)
- ATR (Average True Range)

### Trend Indicators
- SMA 20, SMA 50
- EMA 20, EMA 50

### Volume Indicator
- OBV (On Balance Volume)

---

## Models

| Model | Category | Purpose |
|-------|----------|---------|
| ARIMA | Statistical | Stationary time series forecasting |
| SARIMA | Statistical | Seasonal time series forecasting |
| Prophet | Statistical / ML | Trend and seasonality decomposition |
| LSTM | Deep Learning | Learning long-term temporal dependencies |
| GRU | Deep Learning | Faster alternative to LSTM |
| XGBoost | Machine Learning | Lag-feature-based forecasting |
| GARCH | Econometric | Volatility forecasting with confidence intervals |
| Ensemble | Hybrid | Weighted average of multiple models |

---

## Data Windows and Forecast Horizons

| Data Window | Training Data | Forecast Horizons |
|-------------|---------------|-------------------|
| 6 Months | Last 180 days | 5, 20, 60 days |
| 1 Year | Last 365 days | 5, 20, 60 days |
| 5 Years | Last 1825 days | 30, 60, 90 days |

---

## Model Training

Each model was trained independently for every combination of asset and historical window. In total: **9 assets × 3 historical windows × 6 forecasting models = 162 independent training experiments.**

**Train/Test Split:** The first 80% of each dataset was used for training and the remaining 20% for testing.

**Data Scaling (LSTM & GRU):** MinMaxScaler was fitted only on the training data, while the test data were transformed using the fitted scaler to prevent data leakage.

**Lag Features (XGBoost):** The previous 60 closing prices were transformed into lag features (lag_1 ... lag_60), generated separately for training and testing datasets.

**MLflow Integration:** Every training run was automatically logged into MLflow, including model name, asset, historical window, hyperparameters, RMSE, MAE, and MAPE. Trained models were saved locally under `models/saved/{period}/{asset}_{model}.pkl` or `.pt`.

---

## Evaluation Metrics

- **RMSE** — More sensitive to large prediction errors
- **MAE** — Average absolute prediction error
- **MAPE** — Scale-independent, suitable for comparing different assets

> **Note on GARCH:** GARCH forecasts conditional variance of returns, not price levels. Therefore MAPE values for GARCH are extremely large and not meaningful for price comparison. GARCH is used solely for generating volatility confidence bands in the Streamlit dashboard.

---

## Model Performance

### Best Model per Asset (Based on MAPE)

| Asset | Best Model | Data Window | MAPE |
|-------|-----------|-------------|------|
| AAPL | GRU | 6 Months | 2.46% |
| TSLA | LSTM | 6 Months | 3.20% |
| NVDA | LSTM | 6 Months | 1.94% |
| GOOGL | GRU | 6 Months | 2.83% |
| MSFT | XGBoost | 5 Years | 2.69% |
| AMZN | GRU | 6 Months | 2.61% |
| META | XGBoost | 5 Years | 3.31% |
| BTC | GRU | 5 Years | 3.18% |
| ETH | XGBoost | 5 Years | 3.02% |

### LSTM Results (MAPE)

| Asset | 6M | 1Y | 5Y |
|-------|-----|-----|-----|
| AAPL | 8.07% | 11.56% | 11.49% |
| TSLA | 3.20% | 3.54% | 29.42% |
| NVDA | 1.94% | 11.05% | 17.38% |
| GOOGL | 3.04% | 14.12% | 12.28% |
| MSFT | 6.25% | 9.76% | 16.96% |
| AMZN | 2.68% | 7.25% | 13.79% |
| META | 9.62% | 9.42% | 22.86% |
| BTC | 7.66% | 14.37% | 4.67% |
| ETH | 13.06% | 22.03% | 13.29% |

### GRU Results (MAPE)

| Asset | 6M | 1Y | 5Y |
|-------|-----|-----|-----|
| AAPL | 2.46% | 8.14% | 2.67% |
| TSLA | 3.26% | 3.26% | 17.63% |
| NVDA | 1.96% | 10.20% | 10.19% |
| GOOGL | 2.83% | 7.58% | 8.23% |
| MSFT | 4.98% | 4.10% | 5.54% |
| AMZN | 2.61% | 5.41% | 3.79% |
| META | 7.65% | 6.60% | 3.85% |
| BTC | 4.38% | 3.87% | 3.18% |
| ETH | 4.41% | 13.97% | 5.20% |

### XGBoost Results (MAPE, 5 Years only)

| Asset | RMSE | MAE | MAPE |
|-------|------|-----|------|
| AAPL | 34.90 | 30.00 | 10.71% |
| TSLA | 39.42 | 33.68 | 7.80% |
| NVDA | 58.06 | 56.46 | 29.15% |
| GOOGL | 130.64 | 124.36 | 38.00% |
| MSFT | 16.97 | 12.41 | 2.69% |
| AMZN | 16.10 | 11.68 | 4.75% |
| META | 27.09 | 21.28 | 3.31% |
| BTC | 5114.32 | 3783.86 | 4.21% |
| ETH | 117.68 | 84.91 | 3.02% |

---

## Key Findings

**1. GRU dominates short-term forecasting:** GRU outperformed all other models on the majority of assets in the 6-month window, particularly for stable large-cap stocks like AAPL, GOOGL, and AMZN.

**2. XGBoost excels on 5-year data for select assets:** Unlike the Turkish assets in v1 where 5-year data hurt performance, XGBoost achieved the best MAPE on MSFT (2.69%), META (3.31%), and ETH (3.02%) using 5-year windows. USD-denominated assets show more structural stability over longer periods.

**3. Crypto outperforms expectations:** BTC and ETH showed competitive MAPE values (3–5%) across multiple models, suggesting that despite high volatility, the price patterns are learnable. GRU on BTC with 5-year data achieved 3.18% MAPE.

**4. LSTM struggles on longer windows:** LSTM consistently underperformed GRU across all assets and windows. The gap was largest on 5-year data — for example META LSTM reached 22.86% vs GRU's 3.85%.

**5. v1 vs v2 comparison:** USD-denominated global assets generally yielded lower MAPE values than TRY-denominated Turkish assets. The structural stability of US equity prices over 5 years allowed XGBoost to perform competitively, whereas Turkish assets required shorter windows due to inflation-driven price distortions.

---

## Libraries Used

```text
Python 3.13
PyTorch 2.3
XGBoost
statsmodels
pmdarima
Prophet
arch
scikit-learn
yfinance
ta
MLflow
Streamlit
Plotly
```

---

## Installation

```bash
git clone https://github.com/elfakb/multi-model-financial-time-series-forecasting-v2-global
cd multi-model-financial-time-series-forecasting-v2-global
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Download raw data
python -m src.data.fetch_data

# 2. Preprocess datasets
python -m src.data.preprocess

# 3. Generate technical indicators
python -m src.data.technical_indicators

# 4. Train forecasting models
python -m src.models.arima_model
python -m src.models.sarima_model
python -m src.models.prophet_model
python -m src.models.lstm_model
python -m src.models.gru_model
python -m src.models.xgboost_model
python -m src.models.garch_model

# 5. Launch MLflow UI
mlflow ui

# 6. Launch Streamlit dashboard
streamlit run app.py
```

---

## Project Structure

```
multi-model-financial-time-series-forecasting-v2-global/
├── data/
│   ├── raw/              # Raw OHLCV datasets (6m / 1y / 5y)
│   └── processed/        # Cleaned datasets with technical indicators
├── models/
│   └── saved/            # Trained model files
├── src/
│   ├── data/
│   ├── models/
│   ├── ensemble/
│   └── evaluation/
├── streamlit_app/
│   └── app.py
├── reports/
│   ├── figures/
│   └── model_comparisons/
├── mlruns/
└── requirements.txt
```
```