conda create -n shdr1 python=3.8

conda activate shdr1  

pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python3 test_demo.py --test_imgs test --output_path outputtest

download the weights from"https://drive.google.com/drive/folders/1Derz0HH2s7c5PVjWVzAcpIFG7HaOTp2Z" put into model_zoo
