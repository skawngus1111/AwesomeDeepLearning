# Introduction
2D Image Classfication (2DIC)을 위한 AwesomeDeepLearning은 논문의 구현 및 성능 재현성을 위해 만들어진 프로젝트 페이지입니다.

# To-do-list
- [X] 사용자가 args로 넘겨주는 특정 에폭별 체크포인트 저장 (default = 10 epoch)
- [X] learning rate scheduler를 다양하게 적용할 수 있게 변경 + 어떤 LRS를 사용했는 지 저장 폴더에 표기
- [X] 성능 평가 시 FLOPs와 training/inference speed를 추가
- [X] train/evaluation 터미널 명령어 작성
- [ ] train log 파일 저장
- 다양한 Classification 데이터셋으로 실험
    - [X] CIFAR10/CIFAR100
    - [X] STL10
    - [ ] ImageNet
- [ ] 특정 Checkpoint부터 다시 retrain할 수 있는 코드 추가
- [X] Loss 그래프 그리는 코드 추가
- [X] 재현성 검사
- [X] inference만 하는 코드 추가
- [ ] train configuration.yaml 파일 만들기
- [X] 병렬학습(Distributed Data Parallel) 코드 추가
- Data Augmentation 적용
    - [X] MixUp

# Model Zoo

Various model architectures and results are available.

## CIFAR10

`CUDA_VISIBLE_DEVICES=0 python main.py --num_workers 8 --data_path TwoDIC_dataset/ --save_path TwoDIC_model_weight/ --data_type CIFAR10 --batch_size 256 --criterion CCE --final_epoch 200 --optimizer_name SGD --lr 0.1 --LRS_name CALRS --model_name VGG11 --linear_node 512 --train --step 100`

|Model|TOP-1 Error (%)|TOP-5 Error (%)|Parameters (M)|MAC (M)|training time per epoch (s)|inference time (s)|
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG11|11.86|0.79|9.76M|154.03M|7s|0.09s (0.0115s)|
|VGG13|10.41|0.39|9.94M|229.92M|10s|0.13s (0.0162s)|
|VGG16|10.30|0.44|15.25M|314.96M|13s|0.17s (0.0197s)|
|VGG19|9.98 |0.46|9.76M|154.03M|16s|0.21s|

## CIFAR100

|Model|TOP-1 Error (%)|TOP-5 Error (%)|Parameters (M)|MAC (M)|training time per epoch (s)|inference time (s)|
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG11|11.86|0.79|9.76M|154.03M|7s|0.09s (0.0115s)|
|VGG13|10.41|0.39|9.94M|229.92M|10s|0.13s (0.0162s)|
|VGG16|10.30|0.44|15.25M|314.96M|13s|0.17s (0.0197s)|
|VGG19|9.98 |0.46|9.76M|154.03M|16s|0.21s|

## STL10

|Model|TOP-1 Error (%)|TOP-5 Error (%)|Parameters (M)|MAC (M)|training time per epoch (s)|inference time (s)|
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG11|11.86|0.79|9.76M|154.03M|7s|0.09s (0.0115s)|
|VGG13|10.41|0.39|9.94M|229.92M|10s|0.13s (0.0162s)|
|VGG16|10.30|0.44|15.25M|314.96M|13s|0.17s (0.0197s)|
|VGG19|9.98 |0.46|9.76M|154.03M|16s|0.21s|

## ImageNet

|Model|TOP-1 Error (%)|TOP-5 Error (%)|Parameters (M)|MAC (M)|training time per epoch (s)|inference time (s)|
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG11|11.86|0.79|9.76M|154.03M|7s|0.09s (0.0115s)|
|VGG13|10.41|0.39|9.94M|229.92M|10s|0.13s (0.0162s)|
|VGG16|10.30|0.44|15.25M|314.96M|13s|0.17s (0.0197s)|
|VGG19|9.98 |0.46|9.76M|154.03M|16s|0.21s|
