# Data Sintetis Permintaan — UMKM Stroberi Alamendah

Skrip `generate_dataset.py` membangkitkan dataset permintaan harian sintetis
untuk tiga produk olahan stroberi (studi kasus Desa Wisata Alamendah, Ciwidey)
mengikuti model dekomposisi multiplikatif yang dijelaskan pada laporan.

## Model

```
qty(t) = μ_produk × S_weekly(t) × H(t) × W(t) × (1 + ε_t)
```

| Komponen      | Aturan                                                                    |
|---------------|---------------------------------------------------------------------------|
| μ_produk      | base demand: P001=40, P002=15, P003=80 unit/hari                          |
| S_weekly(t)   | weekly seasonality: Sabtu ×1.8, Minggu ×1.5 (rentang 1,5–1,8×), lain ×1.0 |
| H(t)          | holiday spike: puncak diundi 2,5–4× per hari libur, taper pada window H-1…H+2 |
| W(t)          | weather effect: penalti 20–40% bila curah hujan (Open-Meteo) > 20 mm/hari    |
| ε_t           | Gaussian noise, σ = 10% dari rata-rata                                     |

- **Hari libur** diambil dari pustaka `holidays` (kalender Indonesia). Setiap
  libur nasional membentuk window H-1, H, H+1, H+2 dengan asumsi seragam; puncak
  spike diundi ulang per libur.
- **Curah hujan** diambil dari **Open-Meteo Historical Weather API (ERA5)** pada
  titik Desa Alamendah, Kec. Rancabali (lat -7.1667, lon 107.4167). Data di-cache ke
  `rainfall_alamendah.csv` pada run pertama; hapus file itu untuk mengambil ulang.
- Ambang hujan lebat 20 mm/hari mengacu BMKG/JMBSC (Tonouchi & Kurihara, 2020).
- Seed acak tetap (`SEED = 42`) → hasil reproducible.

## Menjalankan

```bash
pip install -r requirements.txt
python generate_dataset.py     # run pertama butuh internet (ambil hujan Open-Meteo)
```

Output: `data_sintetis_permintaan.csv` (3 tahun × 3 produk = 3288 baris)
dan `rainfall_alamendah.csv` (cache curah hujan Alamendah).

## Kolom output

`date, product_id, product_name, qty_sold, is_weekend, is_holiday, holiday_window, rainfall_mm`
