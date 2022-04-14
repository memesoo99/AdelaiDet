# 1. BoxInst

### Inference

```
python demo/demo.py \
  --input /workspace/regression_data/images1 \
  --output viz/grape_from_real \
  --mask-path /workspace/regression_data/masks \
  --opts MODEL.WEIGHTS /workspace/AdelaiDet/training_dir/BoxInst_MS_R_50_1x_sick4/model_0084999.pth
```

input : inference 돌리고 싶은 모든 파일 (이미 파일들이 들어 있는 폴더 경로ok, 개별 경로들이 string으로 담겨있는 리스트 ok, 하나의 경로 ok) (***다음모델과 연결부위***)
output : 포도송이 mask가 그려진 이미지들(inference결과 visualization)이 저장될 폴더
mask-path : mask.pkl을 저장하고 싶은 폴더. (***다음모델과 연결부위***)

# 2. Feature Extraction

아래 rfr.py를 살행하면 모든 mask들을 iteration돌면서 feature_extraction.py의 Contours 호출 -> feature extraction마치면 result csv에 결과 한줄씩 입력 -> 입력 완료 되면 rfr.py로 다시 돌아와 regressor.predict실행하고, result csv에 'predict'(포도알 최종 예측값) 추가
```
python rfr.py \
  --inference \
  --regressor_path /workspace/grape_rfr/regressor_model.pkl \
  --csv-path /workspace/grape_rfr/sample_result.csv \
  --image-path /workspace/regression_data/images1 \
  --mask-path : /workspace/regression_data/masks
```

inference : inference 모드 (train할땐 inference 대신 train)
regressor_path : 학습킨 RFR모델 경로
csv-path : 결과 출력할 csv path (Sample올려놨는데, index는 잘못들어간거여서 없다 생각하면 돼!)
image-path : Boxinst 모델의 --input과 동일
mask-path : boxinst에서 demo.py돌려서 얻은 mask.pkl들 저장되어 있는 폴더 경로, Boxinst model의 --mask-path와 동일



### Result CSV
출력 항목들
image,number of instances,sunburn_ratio,diameter,circularity,density,aspect ratio,grade(포도 등급),average_hue,predict(최종 포도알수 예측 값)

- 'Thinning' : csv에 송이 다듬기 필요 여부 항목도 추가할거야!!(아마 4.18일에) True(필요) False(불필요)