# =============================================================================
# TRAINING.PY - OTOMATİK OPTİMİZASYON VE MODEL EĞİTİMİ
# =============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import os

# Ayarlar
warnings.filterwarnings("ignore")
print("🚀 EĞİTİM VE OPTİMİZASYON SÜRECİ BAŞLATILIYOR...\n")

# 1. VERİ YÜKLEME
# -----------------------------------------------------------------------------
# Veri yolunu 'data/' klasörü altında arayacak
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
if load_col in df.columns and df[load_col].dtype == 'O': df[load_col] = df[load_col].apply(clean_currency)

df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
prod_cols = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Barajlı', 'Linyit', 'Akarsu']
for col in prod_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = 0

df.loc[df[target_col] > 90000, target_col] = np.nan
df[target_col] = df[target_col].interpolate(method='linear', limit_direction='both')

# 2. FEATURE ENGINEERING (SNIPER)
# -----------------------------------------------------------------------------
print("🛠️ Özellikler Üretiliyor (Feature Engineering)...")
df_final = df.copy()

# Shift
future_cols = ['Doğalgaz', 'Rüzgar', 'Güneş', 'Barajlı', 'Linyit', 'İthal Kömür', 'Akarsu', 'Fuel Oil', 'Jeotermal', 'Biyokütle']
cols_to_shift = [c for c in future_cols if c in df_final.columns]
for col in cols_to_shift:
    df_final[f'{col}_Lag24'] = df_final[col].shift(24)
    df_final.drop(columns=[col], inplace=True)

# Date/Time
df_final['Saat_Int'] = df_final['Saat'].astype(str).str.split(':').str[0].astype(int) if df_final['Saat'].dtype == 'O' else df_final['Saat']
df_final['Hour_Sin'] = np.sin(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Hour_Cos'] = np.cos(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Day_Sin'] = np.sin(2 * np.pi * df_final['Tarih'].dt.dayofweek / 7)
df_final['Day_Cos'] = np.cos(2 * np.pi * df_final['Tarih'].dt.dayofweek / 7)
df_final['Is_Weekend'] = df_final['Tarih'].dt.dayofweek.isin([5, 6]).astype(int)

# Lag Features
df_final['PTF_Lag_24'] = df_final[target_col].shift(24)
df_final['PTF_Lag_168'] = df_final[target_col].shift(168)

# Sniper Features
df_final['PTF_Roll_Mean_168'] = df_final[target_col].rolling(168).mean()
df_final['Relative_Price_Pos'] = (df_final['PTF_Lag_24'] - df_final['PTF_Roll_Mean_168']) / (df_final['PTF_Roll_Mean_168'] + 1)

ren_cols = ['Rüzgar_Lag24', 'Güneş_Lag24', 'Akarsu_Lag24', 'Jeotermal_Lag24', 'Biyokütle_Lag24']
existing_ren = [c for c in ren_cols if c in df_final.columns]
df_final['Total_Renewable_Lag24'] = df_final[existing_ren].sum(axis=1)
df_final['Net_Load'] = df_final[load_col] - df_final['Total_Renewable_Lag24'] if load_col in df_final.columns else -df_final['Total_Renewable_Lag24']

therm_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Fuel Oil_Lag24']
existing_therm = [c for c in therm_cols if c in df_final.columns]
df_final['Total_Thermal_Lag24'] = df_final[existing_therm].sum(axis=1)
df_final['Thermal_Stress'] = df_final['Total_Thermal_Lag24'] / (df_final[load_col] + 1) if load_col in df_final.columns else 0

df_final['Price_Momentum'] = df_final['PTF_Lag_24'] - df_final['PTF_Lag_168']
df_final['Volatility'] = df_final[target_col].rolling(24).std().shift(24)

df_final.dropna(inplace=True)

# 3. VERİ HAZIRLIĞI VE OPTİMİZASYON (RANDOM SEARCH)
# -----------------------------------------------------------------------------
print("⚙️ Veri Bölünüyor ve Optimizasyon Başlıyor...")

drop_cols = ['Tarih', 'Zaman', 'Saat', 'Saat_Int', target_col, 'Month', 'Day_of_Week']
X = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns])
y = df_final[target_col]

# Tarih Bazlı Bölme (31 Ekim'e kadar eğitim)
train_end_date = '2025-10-31'
train_mask = (df_final['Tarih'] <= train_end_date)
X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

# SENİN İSTEDİĞİN OPTİMİZASYON KISMI BURADA 👇
param_dist = {
    'n_estimators': [1000, 2000, 3000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'objective': ['reg:squarederror']
}

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
tscv = TimeSeriesSplit(n_splits=3) # Zaman serisi bölünmesi

print("⏳ RandomizedSearchCV Çalışıyor (Biraz zaman alabilir)...")
search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10, # Hız için 10 yaptık, vaktin varsa 20 yapabilirsin
    scoring='neg_root_mean_squared_error',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print(f"\n🏆 EN İYİ PARAMETRELER BULUNDU: {search.best_params_}")
print(f"   En İyi Skor (Neg RMSE): {search.best_score_:.2f}")

# 4. FİNAL MODEL VE KAYIT
# -----------------------------------------------------------------------------
print("\n🔥 En İyi Parametrelerle Final Modeli Kaydediliyor...")

# En iyi modeli al
best_model = search.best_estimator_

# İsteğe bağlı: Modeli tüm veriyle (Training seti) tekrar fit edelim (zaten etti ama garanti olsun)
# best_model.fit(X_train, y_train) -> RandomSearch zaten en iyisini fit edip bırakır.

# Kayıt Klasörü Kontrolü
if not os.path.exists('models'):
    os.makedirs('models')

model_path = os.path.join('models', 'epias_model_final.pkl')

joblib.dump({
    'model': best_model,
    'features': X_train.columns.tolist(),
    'best_params': search.best_params_ # Merak edersen diye parametreleri de içine gömdüm
}, model_path)

print(f"📦 Model Başarıyla Kaydedildi: {model_path}")
print("🏁 SÜREÇ TAMAMLANDI KRAL! Şimdi 'app.py'ye geçebiliriz.")