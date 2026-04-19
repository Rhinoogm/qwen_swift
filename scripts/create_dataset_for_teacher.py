import pandas as pd
import json
import os

def generate_dataset_jsonl(parquet_path, json_path, output_jsonl):
    # 1. 클래스 이름 매핑 적용 (1000번 추가)
    class_mapping = {0: "human", 15: "cat", 16: "dog", 1000: "face"} 

    print("1. Loading and processing Auto Crop JSON...")
    with open(json_path, 'r') as f:
        crop_data = json.load(f)
    
    # 이미지 정보 DataFrame화
    df_images = pd.DataFrame(crop_data['images'])
    df_images.rename(columns={'id': 'image_id'}, inplace=True)
    
    # 어노테이션 정보 DataFrame화 및 필터링 (category_id 0번 기준)
    df_anno = pd.DataFrame(crop_data['annotations'])
    df_anno = df_anno[df_anno['category_id'] == 0]
    
    # image_id 기준으로 score가 가장 높은 Top 1만 추출
    df_anno = df_anno.sort_values('score', ascending=False).drop_duplicates('image_id')
    
    # 이미지 정보와 어노테이션 병합
    df_crop = pd.merge(df_anno, df_images, on='image_id', how='inner')
    
    # Relative 좌표로 변환
    df_crop['x1'] = round(df_crop['bbox'].apply(lambda b: b[0]) / df_crop['width'], 4)
    df_crop['y1'] = round(df_crop['bbox'].apply(lambda b: b[1]) / df_crop['height'], 4)
    df_crop['x2'] = round((df_crop['bbox'].apply(lambda b: b[0] + b[2])) / df_crop['width'], 4)
    df_crop['y2'] = round((df_crop['bbox'].apply(lambda b: b[1] + b[3])) / df_crop['height'], 4)
    
    # 파일명을 키로 하여 빠르게 찾을 수 있도록 Dictionary 변환
    df_crop['base_filename'] = df_crop['file_name'].apply(lambda x: os.path.basename(x))
    autocrop_dict = {}
    for row in df_crop.itertuples():
        autocrop_dict[row.base_filename] = {
            "x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2, "score": row.score
        }

    print("2. Loading and processing Detector Parquet...")
    # Parquet 로드 및 saliency 0.2 이상 필터링
    df_det = pd.read_parquet(parquet_path)
    df_det = df_det[df_det['saliency'] >= 0.2]
    
    # 클래스명 매핑 적용 (정의되지 않은 ID는 class_ID 형태로 유지)
    df_det['class_name'] = df_det['id_class'].map(lambda x: class_mapping.get(x, f"class_{x}"))
    
    # Parquet의 image 경로에서 파일명(basename) 추출
    df_det['base_filename'] = df_det['image'].apply(lambda x: os.path.basename(x))
    
    print("3. Aggregating objects per image...")
    # 이미지 단위로 딕셔너리 리스트로 묶기
    def make_obj(row):
        return {
            "class_id": row.id_class,
            "class_name": row.class_name,
            "box_cx": round(row.box_cx, 4),
            "box_cy": round(row.box_cy, 4),
            "box_w": round(row.box_w, 4),
            "box_h": round(row.box_h, 4)
        }
    
    df_det['obj_dict'] = df_det.apply(make_obj, axis=1)
    
    # 파일명(basename)을 기준으로 대용량 GroupBy 진행
    grouped_det = df_det.groupby('base_filename')['obj_dict'].apply(list).reset_index()

    print("4. Writing to JSONL...")
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for row in grouped_det.itertuples():
            base_fname = row.base_filename
            detector_objects = row.obj_dict
            
            # Auto crop 정보 가져오기 (없으면 None)
            autocrop_top1 = autocrop_dict.get(base_fname, None)
            
            # 2. 최종 JSON 형태 구성 (teacher_answer 제거)
            record = {
                "image": base_fname, 
                "split": "val",
                "sample_id": os.path.splitext(base_fname)[0],
                "detector_objects": detector_objects,
                "autocrop_top1": autocrop_top1
            }
            
            # 한 줄씩 바로바로 쓰기 (메모리 절약)
            f_out.write(json.dumps(record) + "\n")

    print(f"Done! Saved to {output_jsonl}")

# 실행 예시
# generate_dataset_jsonl(
#     parquet_path='Detector.parquet', 
#     json_path='auto_crop.json', 
#     output_jsonl='output.jsonl'
# )



# 실행 예시
generate_dataset_jsonl(
    parquet_path='data/coco/train/auto_crop_train2017.parquet', 
    json_path='data/coco/train/coco_annotations_train_onnx_gem.json', 
    output_jsonl='data/coco/train/coco_train_det_autocrop.jsonl'
)


'''
python scripts/generate_teacher_labels.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input data/coco/output_val.jsonl \
  --output data/coco/output_val_with_teacher.jsonl \
  --root-dir /home/km_rhino.kim/Documents/1_GIT_REPOS/2_EX_GITS/auto-crop/coco/val2017 \
  --temperature 0.0

'''

