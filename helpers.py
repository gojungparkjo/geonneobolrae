# Helper functions

import os
import glob # 폴더에서 이미지를 불러오기 위한 라이브러리
import matplotlib.image as mpimg



# 이 함수는 이미지와 라벨을 불러오고 리스트에 추가
# 리스트는 모든 이미지와 관련 라벨이 포함
# 예를 들어 데이터를 불러온 후 im_list[0][:]가 리스트의 첫번째 이미지-라벨 쌍
def load_dataset(image_dir):

    im_list = []
    image_types = ["red", "yellow", "green"]

    for im_type in image_types:
        
        # glob는 "image_dir/im_type/*" 경로에 있는 모든 이미지를 읽음
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # 이미지에서 읽음
            im = mpimg.imread(file)
            
            # 이미지가 존재하는지/이미지가 올바르게 읽혔는지 확인
            if not im is None:
                # 이미지와 이미지의 타입(색상)을 리스트에 추가
                im_list.append((im, im_type))

    return im_list


