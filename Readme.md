## Human beautifying and Image matting
We build an end-to-end human beautifying and matting system, which can handle any human photos supplied by user. Firstly we beautify the photo depending on the users’ preferences(smooth, lip brighten, whiten, thin, etc.), and then we separate the man from the background(matting). Using this system users can generate ID Photos easily.
### example
![image](https://raw.githubusercontent.com/YuchaoZheng/readme_add_pic/master/images/example.jpg)

### network
![image](https://raw.githubusercontent.com/YuchaoZheng/readme_add_pic/master/images/network.jpg)

![image](https://raw.githubusercontent.com/YuchaoZheng/readme_add_pic/master/images/result.jpg)

### Deployment

Back end: Flask

Front end: HTML+JavaScript

![image](https://raw.githubusercontent.com/YuchaoZheng/readme_add_pic/master/images/web.jpg)

### Architecture

```angular2

.
├── Readme.md
├── app.py
├── beautify.py
├── infer.py
├── make_dataset.py
├── segmentation
├── simple_request.py
├── src
│   ├── __init__.py
│   ├── dataset.py
│   ├── resnetUnet.py
│   └── unet_model.py
├── static
│   ├── index.js
│   └── style.css
├── templates
│   ├── index.html
│   └── try.html
├── test.py
├── test_beauty.py
├── tmp_test_beauty.py
├── train.py
```

### Reference
https://blog.csdn.net/grafx/article/details/70232797?locationNum=11&fps=1
https://zhuanlan.zhihu.com/p/29718304
https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets
https://www.kaggle.com/vbookshelf/selfie-segmenter-keras-and-u-net
https://github.com/milesial/Pytorch-UNet
https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88
