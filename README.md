# MTDO (IEEE JSTARS 2025)
### 📖[**Paper**](https://doi.org/10.1109/JSTARS.2025.3590041) 

PyTorch codes for "[Multi-Temporal Difference and Dynamic Optimization Framework for Multi-Scale Motion Satellite Video Super-Resolution](https://doi.org/10.1109/JSTARS.2025.3590041)", **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing  (JSTARS)**, 2025.


### Abstract
> Motion alignment is a critical task in satellite video super-resolution. Most existing methods rely on optical flow or deformable convolution for motion alignment. However, these methods perform poorly in handling complex satellite video scenes and diverse moving objects. To address this challenge, we propose an efficient super-resolution method based on a Multi-Temporal Difference and Dynamic Optimization (MTDO) framework. Specifically, we introduce a Multi-level Temporal Difference Analysis Mechanism (MTDM) to rapidly capture comprehensive motion information. Based on the extracted temporal difference data, we further develop a Dynamic Routing Optimization Module (T-DROM) to extract multi-scale motion information. In addition, we introduce a Multi-Attention Enhancement and Correction Module (MAECM) to refine the long-term temporal difference features from T-DROM and reduce accumulated errors. These enhancements help recover spatial details and improve spatio-temporal consistency. We conducted detailed ablation studies to validate our contributions and compared the proposed method with state-of-the-art VSR approaches. Experimental results show that our method improves video reconstruction quality while achieving an effective balance between performance and efficiency.
> 
### Network  
 ![image](/fig/framework.png)
## 🧩Install
```
git clone https://github.com/zhaoqian469/MTDO.git
```
## Environment
 * CUDA 11.6
 * PyTorch  1.13.1+cu116
 * build DCNv2
 
 ## Dataset Preparation
 Please download our dataset in 
 * Baidu Netdisk [MTDO-datasets](https://pan.baidu.com/s/1HBhLiuDGVKYcaQt-Labnrg) Code:iksz
You can also train your dataset following the directory sturture below!


## Training
```
python main.py
```

## Test
```
python eval.py
python eval_VISO.py
```

### Qualitative results
 ![image](/fig/result1.png)
### Quantitative results
 ![image](/fig/result2.png)
#### More details can be found in our paper! The model parameters and Flops results we presented were tested on the unorganized model, and there is a slight deviation from the current model's results, but this does not affect the final outcome.

## Acknowledgement
We especially thank the contributors of the "[LGTD](https://github.com/XY-boy/LGTD?tab=readme-ov-file#lgtd-ieee-tcsvt-2023)" codebase for providing helpful code.

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: zhaoqian22@ucas.ac.cn

## Citation
If you find our work helpful in your research, please consider citing it. Thank you! 😊😊
Our paper and code are improved on "[Multi-Temporal Difference and Dynamic Optimization Framework for Multi-Scale Motion Satellite Video Super-Resolution](https://doi.org/10.1109/JSTARS.2025.3590041)". Thanks to the author for the support and help!

```
@article{zhao2025multi,
  title={Multi-Temporal Difference and Dynamic Optimization Framework for Multi-Scale Motion Satellite Video Super-Resolution},
  author={Zhao, Qian and Guo, Youming and Min, Lei and Rao, Changhui},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```


