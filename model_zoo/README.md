conda create -n shdr1 python=3.8
conda activate shdr1  
 conda env remove -n shdr1
 nvidia-smi
conda activate shdr1
 CUDA_VISIBLE_DEVICES=2 python3 test_demo.py --test_imgs /data2/wangzihao/singlehdrdata/test --output_path outputtest