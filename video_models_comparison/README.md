# 🎬 Open-Source Video Generation Models - Karşılaştırma

> Sora kapandıktan sonra en güncel açık kaynak text-to-video ve image-to-video modelleri.
> HuggingFace trending (Mart 2026) baz alınarak hazırlanmıştır.

## 📋 Model Listesi

| # | Notebook | Model | Tip | Parametre | Çözünürlük | HF Repo |
|---|----------|-------|-----|-----------|------------|---------|
| 1 | `01_wan2.2_t2v.ipynb` | Wan2.2-T2V-A14B | Text-to-Video | 14B (MoE) | 720P/1080P | `Wan-AI/Wan2.2-T2V-A14B` |
| 2 | `02_wan2.2_i2v.ipynb` | Wan2.2-I2V-A14B | Image-to-Video | 14B (MoE) | 720P | `Wan-AI/Wan2.2-I2V-A14B` |
| 3 | `03_hunyuan_video.ipynb` | HunyuanVideo | Text-to-Video | 13B | 720P | `tencent/HunyuanVideo` |
| 4 | `04_hunyuan_video_1.5.ipynb` | HunyuanVideo 1.5 | Text/Image-to-Video | 8.3B | 720P+SR→1080P | `tencent/HunyuanVideo-1.5` |
| 5 | `05_ltx_2.3.ipynb` | LTX-2.3 | Image-to-Video | 21B | 4K@50fps | `Lightricks/LTX-2.3` |
| 6 | `06_ltx_2.ipynb` | LTX-2 | Text/Image-to-Video | - | 4K@50fps | `Lightricks/LTX-2` |
| 7 | `07_skyreels_v3.ipynb` | SkyReels-V3 | Image-to-Video | 19B | 720P | `Skywork/SkyReels-V3-A2V-19B` |
| 8 | `08_sana_video.ipynb` | SANA-Video | Text-to-Video | 2B | 720P | `Efficient-Large-Model/SANA-Video_2B_720p` |
| 9 | `09_mochi_1.ipynb` | Mochi 1 | Text-to-Video | 10B | 848x480 | `genmo/mochi-1-preview` |
| 10 | `10_cogvideox.ipynb` | CogVideoX-5B | Text-to-Video | 5B | 720x480 | `THUDM/CogVideoX-5b` |
| 11 | `11_longcat_video.ipynb` | LongCat-Video | Text-to-Video | - | 720P | `meituan-longcat/LongCat-Video` |
| 12 | `12_helios_distilled.ipynb` | Helios-Distilled | Text-to-Video | - | - | `BestWishYsh/Helios-Distilled` |
| 13 | `13_svd_xt.ipynb` | Stable Video Diffusion XT | Image-to-Video | - | 576x1024 | `stabilityai/stable-video-diffusion-img2vid-xt` |
| 14 | `14_wan2.2_vace.ipynb` | Wan2.2-VACE-Fun | T2V + Controllable | 14B | 720P | `alibaba-pai/Wan2.2-VACE-Fun-A14B` |

## 🔧 Gereksinimler

```bash
pip install torch torchvision diffusers transformers accelerate sentencepiece imageio[ffmpeg] opencv-python
```

## 🚀 Kullanım

Her notebook bağımsız çalışır. Sırayla veya istediğin modelden başlayarak çalıştırabilirsin.

## 💡 Notlar

- **VRAM**: Büyük modeller (14B+) A100/H100 gibi GPU'lar gerektirir. Küçük modeller (1.3B-5B) RTX 4090'da çalışabilir.
- **Quantization**: GGUF versiyonları `unsloth/LTX-2.3-GGUF`, `QuantStack/Wan2.2-T2V-A14B-GGUF` gibi repolarda mevcut.
- **Hız**: `lightx2v/Wan2.2-Lightning` gibi distill edilmiş versiyonlar çok daha hızlı inference sunar.
