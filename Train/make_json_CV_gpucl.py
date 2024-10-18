import os
import json
import glob


def get_index_from_pathclass_by_filename(lst):
    return sorted(range(len(lst)), key=lambda k: os.path.split(lst[k])[1])

if __name__ == "__main__":
    LABEL_NAMES = {
                "bolus": 1,
                "nank": 2,
                "zetu": 3,
                "skull": 4,
                "mandible": 5,
                "cervical_spine": 6,
                "hyoid_bone": 7,
                "thyroid_cartilag": 8,
                #"epiglottis": 9,
                #"airway": 10,
                #"nasopharynx": 11,
                #"oropharynx": 12,
                #"hypopharynx": 13,
                #"main": 14,
                #"patch": 15,
                #"syokudo": 16,
                #"jointo_chuinto_cloased": 17,
                }



    LABEL_COLORS = {
                    0: (0, 0, 0),       # 背景色（黒）
                    1: (255, 0, 0),     # クラス1の色（赤）
                    2: (0, 255, 0),     # クラス2の色（緑）
                    3: (0, 0, 255),     # クラス3の色（青）
                    4: (255, 255, 0),   # クラス4の色（黄色）
                    5: (255, 0, 255),   # クラス5の色（マゼンタ）
                    6: (0, 255, 255),   # クラス6の色（水色）
                    7: (128, 0, 0),     # クラス7の色（暗赤色）
                    8: (0, 128, 0),     # クラス8の色（暗緑色）
                    9: (0, 0, 128),     # クラス9の色（暗青色）
                    10: (128, 128, 0),  # クラス10の色（オリーブ色）
                    11: (128, 0, 128),  # クラス11の色（パープル）
                    12: (0, 128, 128),  # クラス12の色（ティール）
                    13: (255, 165, 0),  # クラス13の色（オレンジ）
                    14: (75, 0, 130),   # クラス14の色（インディゴ）
                    15: (238, 130, 238),# クラス15の色（バイオレット）
                    16: (139, 69, 19),  # クラス16の色（茶色）
                    17: (60, 179, 113)  # クラス17の色（ミディアムシースグリーン）
                    }


    cace_list = ["ENGE_2_ENGE_2_","enge81_","enge167","enge255","enge256"]

    file_path = []
    fn_keys = ("image", "label") 
    images_root = "/data03/user/ogura/nnUnet/datasets/nnUnet_raw/Dataset012_Swallowing-8structures/imagesTr"
    labels_root = "/data03/user/ogura/nnUnet/datasets/nnUnet_raw/Dataset012_Swallowing-8structures/labelsTr"
    image_paths = glob.glob(images_root+"/*.mha")
    label_paths = glob.glob(labels_root+"/*.mha")
    
    image_indices = get_index_from_pathclass_by_filename(image_paths)
    label_indices = get_index_from_pathclass_by_filename(label_paths)
    image_paths = [image_paths[k] for k in image_indices]
    label_paths = [label_paths[k] for k in label_indices]
    print(image_paths)
    print(label_paths)

    for case_idx in range(5):
        dataset_json = {
            "labels": {
                "background":0,
                "bolus": 1,
                "nank": 2,
                "zetu": 3,
                "skull": 4,
                "mandible": 5,
                "cervical_spine": 6,
                "hyoid_bone": 7,
                "thyroid_cartilag": 8
                #"epiglottis": 9,
                #"airway": 10,
                #"nasopharynx": 11,
                #"oropharynx": 12,
                #"hypopharynx": 13,
                #"main": 14,
                #"patch": 15,
                #"syokudo": 16,
                #"jointo_chuinto_cloased": 17,
            },
            "tensorImageSize": "3D",
            "training": [],
            "validation": []
        }
        
        
                
        for image_filepath, label_filepath in zip(image_paths,label_paths):
                
            if cace_list[case_idx] in image_filepath:
                dataset_json["validation"].append({"image":image_filepath,"label":label_filepath})
            else:
                dataset_json["training"].append({"image":image_filepath,"label":label_filepath})
                #file_path.append({"image":image_filepath,"label":label_filepath})

        datasets = f'../dataset_json_list/Dataset012_Swallowing-8structures_faststor/dataset_val_{cace_list[case_idx]}.json'
        with open(datasets, 'w') as outfile:
            json.dump(dataset_json, outfile)