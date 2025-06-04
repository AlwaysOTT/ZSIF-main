<div align="center">
  
# ZSIF-main: A Zero-shot Solar Irradiance Forecasting Model based on Satellite Images and Numerical Series
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/) 

</div>

## Description
This is the official repository to the paper ["ZSIF-main: A Zero-shot Solar Irradiance Forecasting Model based on Satellite Images and Numerical Series"] by **Chunru Dong**, **Jiahong Tang**, **Feng Zhang**, **Qiang Hua**, **Boon-han Lim**.

## Dataset
The paper uses publicly available data provided by Boussif et al[Improving day-ahead Solar Irradiance Time Series Forecasting by Leveraging Spatio-Temporal Context]. We would like to thank them for their research.

You can view the dataset [here](https://app.activeloop.ai/crossvivit/SunLake) and can you access it as follows:
```python
import deeplake

ds = deeplake.load('hub://crossvivit/SunLake')
```
If you wish to download it you can do the following:
```python
import deeplake

ds = deeplake.load('hub://crossvivit/SunLake')
local_dataset = ds.copy('/path/to/local/storage', num_workers=4)
```

## Installation
```bash
pip install -r requirements.txt
```

## Experiments
```bash
HYDRA_FULL_ERROR=1 python main.py experiment=exp_ZSIF
```

## License

ZSIF is licensed under the MIT License.

```
MIT License

Copyright (c) (2023) Ghait Boukachab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
