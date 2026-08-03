"""
generate_dataset.py
===================

Pembangkit dataset sintetis permintaan harian untuk UMKM
(studi kasus Desa Wisata Alamendah, Ciwidey, Jawa Barat).

Permintaan harian dimodelkan secara multiplikatif dari sebuah permintaan dasar
(base demand) per produk yang dimodulasi oleh empat faktor eksogen:

    qty = round( base_produk
                 x faktor_akhir_pekan
                 x faktor_holiday_window
                 x faktor_curah_hujan
                 x noise )

- Hari libur nasional diambil dari pustaka `holidays` (kalender Indonesia),
  sehingga rentang pengaruh libur (holiday window) mengikuti kalender resmi.
- Curah hujan dibangkitkan secara sintetis mengikuti pola musim (monsun):
  musim basah November-Maret, musim kering Juni-September. Nilai ini dapat
  diganti dengan data historis Open-Meteo bila diperlukan (lihat generate_rainfall).

Seluruh proses menggunakan seed acak tetap agar hasil dapat direproduksi.

Menjalankan:
    python generate_dataset.py
Keluaran:
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

# Produk dan permintaan dasar (unit/hari pada kondisi: hari kerja, tanpa
# window libur, curah hujan ringan). Dikalibrasi dari data lapangan UMKM.
PRODUK = {
    "P001": {"nama": "Selai Stroberi (250g)",    "base": 41},
    "P002": {"nama": "Strawberry Cake (loyang)",  "base": 15},
    "P003": {"nama": "Jus Stroberi Segar (cup)",  "base": 81},
}

# Faktor pengali eksogen
FAKTOR_AKHIR_PEKAN = 1.6          # Sabtu/Minggu -> kunjungan wisata naik

FAKTOR_HOLIDAY_WINDOW = {         # rentang pengaruh hari libur nasional
    "H-1": 2.2,                   # H-1 : sehari sebelum libur (belanja persiapan)
    "H":   3.5,                   # H   : hari libur itu sendiri (puncak)
    "H+1": 2.0,                   # H+1 : sehari sesudah libur
    "H+2": 1.6,                   # H+2 : dua hari sesudah libur (ekor kunjungan)
    "---": 1.0,                   # di luar window libur
}

# Faktor curah hujan (langkah/step): hujan lebat menekan kunjungan wisata
def faktor_hujan(mm: float) -> float:
    if mm < 10:
        return 1.00               # ringan / tidak hujan
    elif mm < 20:
        return 0.85               # sedang
    else:
        return 0.70               # lebat

NOISE_SIGMA = 0.11                # ragam acak multiplikatif (~11%)

# Rata-rata curah hujan harian per bulan (mm) -- pola musim Ciwidey/Jawa Barat
CURAH_HUJAN_BULANAN = {
    1: 8.0, 2: 8.1, 3: 7.4, 4: 6.2,  5: 6.3,  6: 2.9,
    7: 2.2, 8: 2.3, 9: 3.3, 10: 3.7, 11: 8.2, 12: 9.0,
}


# ---------------------------------------------------------------------------
# KOMPONEN PEMBANGKIT
# ---------------------------------------------------------------------------
def assign_holiday_window(tanggal: pd.Series) -> pd.Series:
    """Beri label holiday window (H-1, H, H+1, H+2, atau ---) untuk tiap tanggal.

    H diprioritaskan, lalu H-1, H+1, H+2. Bila dua libur berdekatan, label
    yang lebih dekat ke hari libur menang sesuai urutan prioritas tersebut.
    """
    tahun = sorted(tanggal.dt.year.unique())
    libur = holidays.Indonesia(years=tahun)
    hari_libur = {pd.Timestamp(h) for h in libur.keys()}

    window = pd.Series("---", index=tanggal.index)
    tgl = tanggal.dt.normalize()

    # Prioritas penetapan: H, lalu H-1, H+1, H+2 (hanya isi yang masih "---").
    # offset = selisih hari terhadap hari libur; tanggal penerima label = H + offset.
    offset_label = [(0, "H"), (-1, "H-1"), (1, "H+1"), (2, "H+2")]
    for offset, label in offset_label:
        target_dates = {h + pd.Timedelta(days=offset) for h in hari_libur}
        mask = tgl.isin(target_dates) & (window == "---")
        window.loc[mask] = label
    return window


def generate_rainfall(tanggal: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Bangkitkan curah hujan harian (mm) mengikuti rata-rata musiman.

    Distribusi gamma dipakai agar right-skewed (banyak hari kering, sedikit
    hari hujan lebat). Untuk memakai data nyata, ganti fungsi ini dengan
    pembacaan arsip harian Open-Meteo pada koordinat lokasi UMKM.
    """
    mean_harian = tanggal.month.map(CURAH_HUJAN_BULANAN).to_numpy(dtype=float)
    shape = 0.7                                   # <1 -> banyak nilai kecil/0
    scale = mean_harian / shape
    hujan = rng.gamma(shape=shape, scale=scale)
    return np.round(hujan, 1)


def compute_demand(base, is_weekend, hw_label, rain_mm, rng):
    """Hitung qty_sold satu baris dari model multiplikatif."""
    f_we = FAKTOR_AKHIR_PEKAN if is_weekend else 1.0
    f_hw = FAKTOR_HOLIDAY_WINDOW[hw_label]
    f_rain = faktor_hujan(rain_mm)
    noise = rng.lognormal(mean=0.0, sigma=NOISE_SIGMA)
    qty = base * f_we * f_hw * f_rain * noise
    return max(1, int(round(qty)))


# ---------------------------------------------------------------------------
# PIPELINE UTAMA
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    tanggal = pd.date_range(TANGGAL_MULAI, TANGGAL_AKHIR, freq="D")
    kalender = pd.DataFrame({"date": tanggal})
    kalender["is_weekend"] = kalender["date"].dt.dayofweek.isin([5, 6]).astype(int)
    kalender["holiday_window"] = assign_holiday_window(kalender["date"])
    kalender["is_holiday"] = (kalender["holiday_window"] == "H").astype(int)
    kalender["rainfall_mm"] = generate_rainfall(pd.DatetimeIndex(kalender["date"]), rng)

    baris = []
    for _, hari in kalender.iterrows():
        for pid, info in PRODUK.items():
            qty = compute_demand(
                info["base"], hari["is_weekend"], hari["holiday_window"],
                hari["rainfall_mm"], rng,
            )
            baris.append({
                "date": hari["date"].date(),
                "product_id": pid,
                "product_name": info["nama"],
                "qty_sold": qty,
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
