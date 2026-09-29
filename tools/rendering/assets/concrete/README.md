# Scanned concrete PBR maps

The `concrete` material uses the 1K CC0 **Gravel Embedded Concrete** surface
captured by Charlotte Baglioni for [Poly Haven](https://polyhaven.com/a/gravel_embedded_concrete).
The original surface is 2 m wide and provides photographed aggregate/cement
color, measured roughness variation, and a height map. The three JPG files
in this directory are **material inputs**, not example render outputs.

| Local file | Upstream 1K map | SHA-256 |
|---|---|---|
| `diffuse.jpg` | [Color](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/gravel_embedded_concrete/gravel_embedded_concrete_diff_1k.jpg) | `3ea5b493379c4d30c02c04b93361d3e34f91ed04058ffb4ea915702f1cd17049` |
| `roughness.jpg` | [Roughness](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/gravel_embedded_concrete/gravel_embedded_concrete_rough_1k.jpg) | `6427f98c3e4a94f500f6c0ca0e453daff61592004f248bd06fd9e14d2c6c1ae1` |
| `height.jpg` | [Displacement](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/gravel_embedded_concrete/gravel_embedded_concrete_disp_1k.jpg) | `f594a6a434c8d15700d5ab83225fda8fb04f3df25a479e82959339048b6afbf8` |

The maps are [CC0](https://polyhaven.com/license). Attribution is not
required, but crediting the authors is encouraged. The 1K size keeps the
repository addition small; for close-up renders, pass `--solid-texture-dir`
pointing to higher-resolution maps downloaded from the same asset.
