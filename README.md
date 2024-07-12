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
├─ example
│    └─ ... .ipynb
│
├─ retrieval.py
├─ run.py
└─ vqa.py
```

## 비디오 영상 간 검색  
영상 간 검색을 위해 raw_data/sample_data에 검색하고자 하는 영상을 업로드 해주세요.  
```
python retrieval.py
```

## 비디오 영상 내 분석  
영상 내 분석을 위해 database/extracted_sample_data에 .josn, .npy가 있는지 확인해주세요.  
```
python vqa.py
```

## 비디오 영상 간 검색 및 분석  
```
python run.py
```
