---
layout: page
title: Regresion Logistica
---



Creo datos de una sola variable y de dos clases,cada clase sera un distrubucion a la que le puedo cambiar el sigma (std) y el mu (center)


```python
import numpy as np
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=50, centers=np.array([1,3]).reshape(-1, 1), n_features=1,random_state=1,cluster_std=0.8)
```





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://github.com/muydipalma/home/raw/v3/fig0.html" height="525" width="100%"></iframe>



<iframe src="https://github.com/muydipalma/home/blob/v3/bfig0.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="500"
    scrolling="no"
    seamless="seamless"
    frameborder="0">
</iframe>

<iframe src="/assets/img/bfig0.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="500"
    scrolling="no"
    seamless="seamless"
    frameborder="0">
</iframe>
