"""
generate_dataset.py
===================

Pembangkit dataset sintetis permintaan harian untuk UMKM pengolah stroberi
(studi kasus Desa Wisata Alamendah, Ciwidey, Jawa Barat).

Permintaan harian dibangun dari model dekomposisi multiplikatif:

    qty(t) = mu_produk x S_weekly(t) x H(t) x W(t) x (1 + eps_t)

dengan:
  - mu_produk    : rata-rata permintaan harian dasar per produk (base demand).
  - S_weekly(t)  : weekly seasonality; faktor Sabtu 1.8x dan Minggu 1.5x
                   (rentang 1,5-1,8x) dibanding hari kerja.
  - H(t)         : holiday spike untuk hari libur nasional. Setiap posisi window
                   (H-1, H, H+1, H+2) diundi INDEPENDEN per hari libur dari
                   rentangnya masing-masing; hari-H mengikuti multiplier 2,5-4x.
  - W(t)         : weather effect; penalti 20%-40% pada hari dengan curah hujan
                   > 20 mm/hari (ambang hujan lebat BMKG/JMBSC).
  - eps_t        : Gaussian noise, standar deviasi 10% dari rata-rata.

Hari libur nasional diambil dari pustaka `holidays` (kalender Indonesia).
Curah hujan harian diambil dari Open-Meteo Historical Weather API (ERA5) pada
titik Desa Alamendah, di-cache lokal. Karena curah hujan bersumber dari arsip
historis (bukan prakiraan), sumber ini bersifat wajib -- tidak ada pembangkit
cadangan. Seed acak tetap agar komponen selain cuaca dapat direproduksi.

Menjalankan:
    python generate_dataset.py
Output:
    data_sintetis_permintaan.csv
"""

import numpy as np
import pandas as pd
import holidays

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
SEED = 42
TANGGAL_MULAI = "2023-01-01"
TANGGAL_AKHIR = "2025-12-31"
OUTPUT_CSV = "data_sintetis_permintaan.csv"

# mu_produk - base demand (unit/hari) dari karakteristik UMKM wisata
PRODUK = {
    "P001": {"nama": "Selai Stroberi (250g)",    "base": 40},
    "P002": {"nama": "Strawberry Cake (loyang)",  "base": 15},
    "P003": {"nama": "Jus Stroberi Segar (cup)",  "base": 80},
}

# S_weekly - weekly seasonality (rentang 1,5-1,8x)
FAKTOR_SABTU  = 1.8
FAKTOR_MINGGU = 1.5

# INDEPENDEN untuk setiap hari libur (dikalibrasi dari data; H = 2,5-4x laporan).
HOLIDAY_RANGE = {
    "H-1": (1.6, 2.4),
    "H":   (2.5, 4.0),
    "H+1": (1.5, 2.3),
    "H+2": (1.2, 1.7),
}

# W(t) --weather effect: penalti 20%-40% hanya untuk hujan > 20 mm/hari
AMBANG_HUJAN_LEBAT = 20.0
PENALTI_MIN = 0.20
PENALTI_MAKS = 0.40

# eps - Gaussian noise, sigma = 10% dari rata-rata
NOISE_SIGMA = 0.10

# Sumber curah hujan: Open-Meteo Historical Weather API (ERA5), titik Desa
# Alamendah, Kab. Bandung.
LAT_ALAMENDAH = -7.1667
LON_ALAMENDAH = 107.4167
RAINFALL_CACHE = "rainfall_alamendah.csv"  


# Komponen
# 
def build_holiday_effects(tanggal: pd.Series, rng: np.random.Generator):
    """Kembalikan label holiday_window dan faktor H(t) untuk tiap tanggal.

    Untuk setiap hari libur, tiap posisi window (H, H-1, H+1, H+2) memperoleh
    undian uniform independen dari rentangnya. Prioritas penetapan: H, lalu
    H-1, H+1, H+2 (posisi terdekat ke hari libur menang bila window bertumpuk).
    """
    tahun = sorted(tanggal.dt.year.unique())
    libur = holidays.Indonesia(years=tahun)
    hari_libur = sorted(pd.Timestamp(h) for h in libur.keys())

    window = pd.Series("---", index=tanggal.index)
    faktor = pd.Series(1.0, index=tanggal.index)
    tgl = tanggal.dt.normalize()

    for offset, label in [(0, "H"), (-1, "H-1"), (1, "H+1"), (2, "H+2")]:
        lo, hi = HOLIDAY_RANGE[label]
        for h in hari_libur:
            target = h + pd.Timedelta(days=offset)
            mask = (tgl == target) & (window == "---")
            if mask.any():
                window.loc[mask] = label
                faktor.loc[mask] = rng.uniform(lo, hi)   # undian independen per (libur, posisi)
    return window, faktor


def fetch_rainfall_openmeteo(start_date, end_date, lat, lon) -> pd.Series:
    """Ambil curah hujan harian (mm) dari Open-Meteo Historical Weather API (ERA5)
    menggunakan pustaka openmeteo-requests. Mengembalikan Series precipitation_sum
    harian, terindeks tanggal.
    """
    import openmeteo_requests
    client = openmeteo_requests.Client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Asia/Jakarta",
    }
    response = client.weather_api(url, params=params)[0]
    precip = response.Daily().Variables(0).ValuesAsNumpy()
    # Arsip mengembalikan satu nilai per hari, urut kronologis untuk rentang diminta
    dates = pd.date_range(start_date, end_date, freq="D")
    return pd.Series(precip[: len(dates)], index=dates).astype(float)


def get_rainfall(tanggal: pd.DatetimeIndex) -> np.ndarray:
    """Curah hujan harian (mm): cache lokal -> Open-Meteo (titik Alamendah).

    Sumber Open-Meteo bersifat wajib (arsip historis). Bila pengambilan gagal,
    error dibiarkan naik agar tidak diam-diam memakai data yang salah.
    """
    import os
    start = tanggal.min().strftime("%Y-%m-%d")
    end = tanggal.max().strftime("%Y-%m-%d")

    if os.path.exists(RAINFALL_CACHE):
        cache = pd.read_csv(RAINFALL_CACHE, parse_dates=["date"]).set_index("date")["rainfall_mm"]
        if cache.index.min() <= tanggal.min() and cache.index.max() >= tanggal.max():
            return cache.reindex(tanggal).fillna(0.0).round(1).to_numpy()

    s = fetch_rainfall_openmeteo(start, end, LAT_ALAMENDAH, LON_ALAMENDAH)
    s.rename_axis("date").rename("rainfall_mm").reset_index().to_csv(RAINFALL_CACHE, index=False)
    print(f"Curah hujan: Open-Meteo (Alamendah {LAT_ALAMENDAH}, {LON_ALAMENDAH}); "
          f"cache -> {RAINFALL_CACHE}")
    return s.reindex(tanggal).fillna(0.0).round(1).to_numpy()


def faktor_hari(dow: int) -> float:
    if dow == 5:
        return FAKTOR_SABTU
    elif dow == 6:
        return FAKTOR_MINGGU
    return 1.0


def faktor_hujan(mm: float, rng: np.random.Generator) -> float:
    """W(t): penalti 20%-40% hanya bila hujan > 20 mm/hari, selain itu 1.0."""
    if mm > AMBANG_HUJAN_LEBAT:
        return 1.0 - rng.uniform(PENALTI_MIN, PENALTI_MAKS)
    return 1.0


# PIPELINE UTAMA
def main():
    rng = np.random.default_rng(SEED)

    tanggal = pd.date_range(TANGGAL_MULAI, TANGGAL_AKHIR, freq="D")
    kalender = pd.DataFrame({"date": tanggal})
    kalender["dow"] = kalender["date"].dt.dayofweek
    kalender["is_weekend"] = kalender["dow"].isin([5, 6]).astype(int)
    hw, hfac = build_holiday_effects(kalender["date"], rng)
    kalender["holiday_window"] = hw
    kalender["holiday_factor"] = hfac
    kalender["is_holiday"] = (kalender["holiday_window"] == "H").astype(int)
    kalender["rainfall_mm"] = get_rainfall(pd.DatetimeIndex(kalender["date"]))

    baris = []
    for _, hari in kalender.iterrows():
        f_day = faktor_hari(hari["dow"])
        f_hw = hari["holiday_factor"]
        f_rain = faktor_hujan(hari["rainfall_mm"], rng)
        for pid, info in PRODUK.items():
            mu = info["base"] * f_day * f_hw * f_rain
            qty = mu * (1.0 + rng.normal(0.0, NOISE_SIGMA))   # Gaussian noise 10%
            baris.append({
                "date": hari["date"].date(),
                "product_id": pid,
                "product_name": info["nama"],
                "qty_sold": max(1, int(round(qty))),
                "is_weekend": hari["is_weekend"],
                "is_holiday": hari["is_holiday"],
                "holiday_window": hari["holiday_window"],
                "rainfall_mm": hari["rainfall_mm"],
            })

    df = pd.DataFrame(baris)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Tersimpan: {OUTPUT_CSV}  ({len(df)} baris, "
          f"{df['date'].min()} s.d. {df['date'].max()})")


if __name__ == "__main__":
    main()
