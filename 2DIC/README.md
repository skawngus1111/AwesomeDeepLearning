# Introduction
2D Image Classfication (2DIC)을 위한 AwesomeDeepLearning은 논문의 구현 및 성능 재현성을 위해 만들어진 프로젝트 페이지입니다.

# To-do-list
- [ ] 사용자가 args로 넘겨주는 특정 에폭별 체크포인트 저장 (default = 10 epoch)
- [ ] learning rate scheduler를 다양하게 적용할 수 있게 변경 + 어떤 LRS를 사용했는 지 저장 폴더에 표기
- [ ] 성능 평가 시 FLOPs와 training/inference speed를 추가

# Model Zoo

Various model architectures and results are available.

## CIFAR10

|Model|TOP-1 Error (%)|TOP-5 Error (%)|
|------|---|---|
|VGG11|12.67|0.56|
|VGG13|10.41|0.39|
|VGG16|10.30|0.44|
|VGG19|9.98 |0.46|

## CIFAR100

|Model|TOP-1 Error (%)|TOP-5 Error (%)|
|------|---|---|
|VGG11|39.58|16.13|
|VGG13|36.18|14.05|
|VGG16|33.98|13.24|
|VGG19|34.02|13.08|

<img width="80%" src="https://user-images.githubusercontent.com/77310264/211721822-e6ab3e1e-5212-4a79-9f80-eb27d406ab25.png"/>
