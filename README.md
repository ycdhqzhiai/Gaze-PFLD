# Gaze-PFLD
PFLD实现Gaze Estimation

## Datasets
* 使用[TEyeD数据集](https://unitc-my.sharepoint.com/personal/iitfu01_cloud_uni-tuebingen_de/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fiitfu01%5Fcloud%5Funi%2Dtuebingen%5Fde%2FDocuments%2F20MioEyeDS&originalPath=aHR0cHM6Ly91bml0Yy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9paXRmdTAxX2Nsb3VkX3VuaS10dWViaW5nZW5fZGUvRXZyTlBkdGlnRlZIdENNZUZLU3lMbFVCZXBPY2JYMG5Fa2Ftd2VlWmEwczlTUT9ydGltZT1rdC1fVTUwQTJVZw)
  这里将其转换为Json格式，只保留landmarks和gaze-vector，其他标注信息没有用到,转换脚本可以在[这里](https://blog.csdn.net/ycdhqzhiai/article/details/115750689?spm=1001.2014.3001.5501)
## Train
如果是按照上面脚本生成的数据集，可以直接`python train.py`，如果是其他格式数据集，修改下dataloder

## Run demo
```
python camera.py
```
![第一个epoch结果](https://github.com/ycdhqzhiai/Gaze-PFLD/blob/main/gaze.jpg)

## Export
```python
python export_onnx.py
```
## Reference resources
* 1.https://github.com/polarisZhao/PFLD-pytorch</br>
* 2.https://github.com/david-wb/gaze-estimation
