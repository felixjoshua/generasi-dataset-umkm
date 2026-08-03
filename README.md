# Data Sintetis Permintaan — UMKM Stroberi Alamendah

Skrip `generate_dataset.py` membangkitkan dataset permintaan harian sintetis
untuk tiga produk olahan stroberi (studi kasus Desa Wisata Alamendah, Ciwidey).

## Model

Permintaan harian dihitung secara multiplikatif:

```
qty = round( base_produk × f_akhir_pekan × f_holiday_window × f_curah_hujan × noise )
```

| Komponen         | Nilai / aturan                                             |
|------------------|------------------------------------------------------------|
| base_produk      | P001=41, P002=15, P003=81 (unit/hari, kondisi dasar)       |
| f_akhir_pekan    | 1.6 pada Sabtu/Minggu, selain itu 1.0                      |
| f_holiday_window | H-1=2.2, H=3.5, H+1=2.0, H+2=1.6, di luar window=1.0       |
| f_curah_hujan    | <10 mm → 1.0 ; 10–20 mm → 0.85 ; >20 mm → 0.70            |
| noise            | lognormal multiplikatif, σ≈0.11 (~11%)                    |

- **Hari libur** diambil dari pustaka `holidays` (kalender Indonesia). Setiap
  hari libur nasional membentuk *window* H-1, H, H+1, H+2.
- **Curah hujan** dibangkitkan sintetis mengikuti rata-rata musiman (basah
  Nov–Mar, kering Jun–Sep); dapat diganti dengan arsip harian Open-Meteo.
- Seed acak tetap (`SEED = 42`) → hasil reproducible.

## Menjalankan

```bash
pip install -r requirements.txt
python generate_dataset.py
```

Output: `data_sintetis_permintaan.csv` (3 tahun × 3 produk = 3288 baris).

## Kolom output

`date, product_id, product_name, qty_sold, is_weekend, is_holiday, holiday_window, rainfall_mm`
