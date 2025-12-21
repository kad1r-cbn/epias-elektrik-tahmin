# =============================================================================
# TRAINING.PY - FİNAL MODEL EĞİTİM MOTORU (V2 - GÜNCEL)
# =============================================================================
# Bu dosya, 'epias_analiz_guncel.py' dosyasındaki gelişmiş mantığı kullanır.
# Çıktı olarak Streamlit'in kullanacağı .pkl dosyasını üretir.
# =============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import warnings
import os
import holidays  # Tatil günleri için (pip install holidays)

# Ayarlar
warnings.filterwarnings("ignore")
print("🚀 GÜNCEL MODEL EĞİTİM SÜRECİ BAŞLATILIYOR (V8 Motor)...\n")

# 1. VERİ YÜKLEME
# -----------------------------------------------------------------------------
file_path = os.path.join('data_s', 'data_set_ex.xlsx')

try:
    df = pd.read_excel(file_path)
    print(f"✅ Veri Seti Yüklendi: {file_path}")
except FileNotFoundError:
    print(f"⚠️ Excel bulunamadı, CSV aranıyor...")
    df = pd.read_csv('data/data_set_ex.xlsx - Gercek Zamanli Uretim.csv')

# Temizlik ve Formatlama
df.columns = [col.strip() for col in df.columns]
target_col = 'PTF (TL/MWH)'
load_col = 'Yük Tahmin Planı (MWh)'

def clean_currency(x):
    if isinstance(x, str):
        if ',' in x and '.' in x: x = x.replace('.', '').replace(',', '.')
        elif ',' in x: x = x.replace(',', '.')
    try: return float(x)
    except: return np.nan

if df[target_col].dtype == 'O': df[target_col] = df[target_col].apply(clean_currency)
if df[load_col].dtype == 'O': df[load_col] = df[load_col].apply(clean_currency)

df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')

# Negatifleri Temizle
prod_cols = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Barajlı', 'Linyit', 'Akarsu']
for col in prod_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = 0

# Outlier Temizliği
df.loc[df[target_col] > 90000, target_col] = np.nan
df[target_col] = df[target_col].interpolate(method='linear', limit_direction='both')

# =============================================================================
# 2. FEATURE ENGINEERING (YENİ NESİL - GÜNCEL DOSYADAN ALINDI)
# =============================================================================
print("🛠️ Gelişmiş Özellikler (Holiday, Smart Rolling) Üretiliyor...")
df_final = df.copy()

# A. Tatil Değişkeni (YENİ!)
tr_holidays = holidays.TR(years=[2023, 2024, 2025, 2026])
df_final['Is_Holiday'] = df_final['Tarih'].apply(lambda x: 1 if x in tr_holidays else 0)

# B. Tarih/Saat
if 'Saat' in df_final.columns:
    if df_final['Saat'].dtype == 'O':
        df_final['Saat_Int'] = df_final['Saat'].astype(str).str.split(':').str[0].astype(int)
    else:
        df_final['Saat_Int'] = df_final['Saat']
else:
    df_final['Saat_Int'] = df_final['Tarih'].dt.hour

df_final['Month'] = df_final['Tarih'].dt.month
df_final['Day_of_Week'] = df_final['Tarih'].dt.dayofweek
df_final['Is_Weekend'] = df_final['Day_of_Week'].isin([5, 6]).astype(int)

# Sin/Cos Dönüşümleri
df_final['Hour_Sin'] = np.sin(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Hour_Cos'] = np.cos(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Day_Sin'] = np.sin(2 * np.pi * df_final['Day_of_Week'] / 7)
df_final['Day_Cos'] = np.cos(2 * np.pi * df_final['Day_of_Week'] / 7)

# C. Shift Operasyonu (Geleceği Silme)
future_cols = ['Doğalgaz', 'Rüzgar', 'Güneş', 'Barajlı', 'Linyit', 'İthal Kömür', 'Akarsu', 'Fuel Oil', 'Jeotermal', 'Biyokütle']
cols_to_shift = [c for c in future_cols if c in df_final.columns]
for col in cols_to_shift:
    df_final[f'{col}_Lag24'] = df_final[col].shift(24)
    df_final.drop(columns=[col], inplace=True)

# D. Lag Features (Fiyat Hafızası)
df_final['PTF_Lag_24'] = df_final[target_col].shift(24)
df_final['PTF_Lag_168'] = df_final[target_col].shift(168)

# E. Smart Rolling Mean (DÜZELTİLDİ: Sızıntısız!)
# Artık hedef değişkenden değil, Lag_24 (dün)'den hesaplıyoruz.
df_final['PTF_Roll_Mean_24'] = df_final['PTF_Lag_24'].rolling(24).mean()
df_final['PTF_Roll_Mean_168'] = df_final['PTF_Lag_24'].rolling(168).mean()
df_final['PTF_Roll_Std_24'] = df_final['PTF_Lag_24'].rolling(24).std()

# F. Sniper Özellikler
# Relative Price
df_final['Relative_Price_Pos'] = (df_final['PTF_Lag_24'] - df_final['PTF_Roll_Mean_168']) / (df_final['PTF_Roll_Mean_168'] + 1)

# Net Load
ren_cols = ['Rüzgar_Lag24', 'Güneş_Lag24', 'Akarsu_Lag24', 'Jeotermal_Lag24', 'Biyokütle_Lag24']
existing_ren = [c for c in ren_cols if c in df_final.columns]
df_final['Total_Renewable_Lag24'] = df_final[existing_ren].sum(axis=1)
df_final['Net_Load'] = df_final[load_col] - df_final['Total_Renewable_Lag24']

# Thermal Stress
therm_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Fuel Oil_Lag24']
existing_therm = [c for c in therm_cols if c in df_final.columns]
df_final['Total_Thermal_Lag24'] = df_final[existing_therm].sum(axis=1)
df_final['Thermal_Stress'] = df_final['Total_Thermal_Lag24'] / (df_final[load_col] + 1)

# Momentum
df_final['Price_Momentum'] = df_final['PTF_Lag_24'] - df_final['PTF_Lag_168']

# Temizlik
df_final.dropna(inplace=True)

# =============================================================================
# 3. MODEL EĞİTİMİ (YENİ PARAMETRELERLE)
# =============================================================================
print("🔥 Model Eğitiliyor (Regularization Aktif)...")

drop_cols = ['Tarih', 'Zaman', 'Saat', 'Saat_Int', target_col, 'Month', 'Day_of_Week']
X = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns])
y = df_final[target_col]

# Tarih Bazlı Bölme (Ekim'e kadar eğit)
train_end_date = '2025-10-31'
train_mask = (df_final['Tarih'] <= train_end_date)
X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

# GÜNCEL PARAMETRELER (Regularization Eklenmiş Hali)
# RandomizedSearch'ten çıkan veya manuel belirlediğimiz "Robust" ayarlar
final_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 5,           # Aşırı derin değil
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,         # L1 Regularization (Yeni)
    'reg_lambda': 5,          # L2 Regularization (Yeni)
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

model = xgb.XGBRegressor(**final_params)
model.fit(X_train, y_train)

print(f"✅ Model Başarıyla Eğitildi. Özellik Sayısı: {len(X_train.columns)}")

# =============================================================================
# 4. KAYIT (PKL OLUŞTURMA)
# =============================================================================
if not os.path.exists('models'):
    os.makedirs('models')

model_path = os.path.join('models', 'epias_model_final.pkl')

joblib.dump({
    'model': model,
    'features': X_train.columns.tolist(),
    'best_params': final_params,
    'training_date': str(pd.Timestamp.now())
}, model_path)

print(f"📦 Model Paketi Kaydedildi: {model_path}")
print("🏁 ŞİMDİ HAZIRIZ! 'streamlit run app.py' komutunu çalıştırabilirsin.")