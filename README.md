## File Structure
```
Tel_LLaVA_v2
├─ LLaVA-NeXT
│  └── ...
│
├─ database  
│ └─ extracted_sample_data  
│    ├─ extracted_video.json  
│    └─ extracted_video.npy  
│
├─ raw_data
│    └─ sample_data
│        └─ ... .mp4
│
├─ retrieval.py
├─ run.py
└─ vqa.py
```

## 비디오 영상 간 검색
```
python retrieval.py
```

## 비디오 영상 내 분석
```
python vqa.py
```

## 비디오 영상 간 검색 및 분석
```
python run.py
```
