# 源码阅读笔记

## 一、core
## 二、dataset
### 2.1 代码目录结构
``` dataset
src/gluonts/dataset
├── artificial
│   ├── _base.py
│   ├── generate_synthetic.py
│   ├── __init__.py
│   └── recipe.py
├── common.py
├── field_names.py
├── __init__.py
├── jsonl.py
├── loader.py
├── multivariate_grouper.py
├── parallelized_loader.py
├── repository
│   ├── _artificial.py
│   ├── datasets.py
│   ├── _gp_copula_2019.py
│   ├── __init__.py
│   ├── _lstnet.py
│   ├── _m4.py
│   ├── _m5.py
│   └── _util.py
├── split
│   ├── __init__.py
│   └── splitter.py
├── stat.py
└── util.py
```
