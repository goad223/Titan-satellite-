# ChangeMaster Ultimate — Phase 1 (Foundation)

**English** | [العربية](#العربية)

ChangeMaster Ultimate is a 100% offline Windows-first desktop application for
detecting changes between satellite images. It adapts automatically to any
hardware, from low-end laptops to GPU workstations. This repository contains
**Phase 1 of 6**: the production-grade foundation.

## What's included in Phase 1

| Module | Description |
| --- | --- |
| `changemaster/core/hardware.py` | Hardware detection and capability tiering (CPU/RAM/disk/GPU) |
| `changemaster/core/config.py` | Persistent JSON settings in `%APPDATA%/ChangeMaster` (or `~/.config` on Linux CI) |
| `changemaster/core/logging_setup.py` | Rotating UTF-8 logs (5 MB × 3) + console, Arabic-safe |
| `changemaster/core/exceptions.py` | Error hierarchy with bilingual (English/Arabic) messages |
| `changemaster/io_engine/` | Unified readers: GeoTIFF/BigTIFF/JPEG2000/ENVI (rasterio), PNG/JPEG/BMP (Pillow), HDF5 (h5py), NetCDF (netCDF4), Sentinel SAFE, Landsat (folder or tar), tiled access for 100,000×100,000 px images, GeoTIFF/PNG writers |
| `changemaster/sensors/` | 12 sensor profiles with automatic detection: Sentinel-1/2, Landsat 5/7/8/9, MODIS, WorldView, Pleiades, SPOT, PlanetScope, generic |
| `scripts/titan_info.py` | CLI: hardware report + supported-format table |
| `scripts/titan_inspect.py` | CLI: inspect any supported image and print normalized metadata |

Heavy dependencies (`rasterio`, `h5py`, `netCDF4`) are imported lazily: when a
library is missing, the application keeps running with the available features
and shows a clear bilingual message — it never crashes at import time.

## Installation

```bash
# Core (always works: PNG/JPEG/BMP + hardware/config/logging)
pip install numpy Pillow psutil

# Optional format support
pip install rasterio   # GeoTIFF / JPEG2000 / ENVI / SAFE / Landsat
pip install h5py       # HDF5
pip install netCDF4    # NetCDF

# Or everything at once
pip install -e ".[all,dev]"
```

## Usage

```bash
python scripts/titan_info.py            # hardware + formats report
python scripts/titan_info.py --json     # machine readable
python scripts/titan_inspect.py path/to/image.tif
python scripts/titan_inspect.py path/to/S2A_...SAFE --json
```

```python
from changemaster import open_image
from changemaster.io_engine.tiled_access import TiledImageAccessor

with open_image("scene.tif") as reader:
    print(reader.metadata.to_dict())
    for tile, data in TiledImageAccessor(reader, tile_size=1024).iter_tiles():
        ...  # process each tile with bounded memory
```

## Tests

All test data is generated programmatically — no large files in the repo.

```bash
pip install -e ".[all,dev]"
pytest          # enforces >= 80% coverage (currently ~91%)
```

## Roadmap

* **Phase 1 — Foundation (this repo)**: hardware, config, logging, I/O engine, sensor profiles
* Phase 2 — Preprocessing (`changemaster/preprocessing/`)
* Phases 3–4 — Change detection engines (`changemaster/engines/`)
* Phase 5 — GUI (`changemaster/gui/`)
* Phase 6 — Packaging and distribution

---

# العربية

**ChangeMaster Ultimate** برنامج سطح مكتب لويندوز يعمل دون اتصال بالإنترنت بنسبة
100% لكشف التغيرات بين صور الأقمار الصناعية، ويتكيف تلقائياً مع أي عتاد من
الحواسيب المحمولة الضعيفة إلى محطات العمل المزودة ببطاقات رسومية. يحتوي هذا
المستودع على **المرحلة الأولى من ست مراحل**: الأساس بجودة إنتاجية كاملة.

## محتويات المرحلة الأولى

| الوحدة | الوصف |
| --- | --- |
| `core/hardware.py` | كشف العتاد وتصنيفه (المعالج/الذاكرة/القرص/البطاقة الرسومية) |
| `core/config.py` | إعدادات JSON دائمة في `%APPDATA%/ChangeMaster` |
| `core/logging_setup.py` | سجلات دوّارة (5MB×3) + كونسول، تدعم UTF-8 والعربية |
| `core/exceptions.py` | هرمية أخطاء برسائل ثنائية اللغة (عربي + إنجليزي) |
| `io_engine/` | قارئات موحدة: GeoTIFF/JPEG2000/ENVI وPNG/JPEG/BMP وHDF5 وNetCDF وSentinel SAFE وLandsat (مجلد أو tar)، وقراءة مجزأة للصور العملاقة 100,000×100,000 بكسل، وكتابة GeoTIFF/PNG |
| `sensors/` | 12 بروفايل مستشعر مع كشف تلقائي للقمر الصناعي |
| `scripts/titan_info.py` | تقرير العتاد + جدول الصيغ المتاحة |
| `scripts/titan_inspect.py` | فحص أي صورة مدعومة وعرض بياناتها الوصفية |

الاعتماديات الثقيلة (`rasterio`, `h5py`, `netCDF4`) تُستورد بشكل كسول: إذا لم
تكن مكتبة ما مثبتة، يستمر البرنامج بالعمل بالميزات المتاحة ويعرض رسالة واضحة
ثنائية اللغة — ولا ينهار أبداً عند الاستيراد.

## التثبيت

```bash
# الأساس (يعمل دائماً)
pip install numpy Pillow psutil

# دعم الصيغ الاختياري
pip install rasterio h5py netCDF4
```

## الاستخدام

```bash
python scripts/titan_info.py                 # تقرير العتاد والصيغ
python scripts/titan_inspect.py مسار/الصورة  # فحص صورة
```

## الاختبارات

كل بيانات الاختبار تُولَّد برمجياً — لا توجد ملفات ضخمة بالمستودع.

```bash
pytest    # التغطية الحالية ~91% (الحد الأدنى 80%)
```
