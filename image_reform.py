import cv2
import numpy as np
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# new.width = h*sin(theta) + w*cos(theta)
# new.height= h*cos(theta) + w*cos(theta)

# | a  b  (1-a)*centre.x-b*centre.y+(new.width/2-centre.x)  |
# |-b  a  b*centre.x+(1-a)*centre.y+(new.height/2-centre.y) |

def rotate_img(origin_img, theta):
    (h, w) = origin_img.shape[:2]
    (cX, cY) = (w//2, h//2)
    
    M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    
    sin = np.abs(M[0, 0])
    cos = np.abs(M[0, 1])
    
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))
    
    rotated_img = cv2.warpAffine(origin_img, M, (nW, nH))
    
    return rotated_img

def rotate_points(x_points, y_points, img_cX, img_cY, img_h, img_w, theta):
    new_x = []
    new_y = []
    
    # make transformation matrix
    M = cv2.getRotationMatrix2D((img_cX, img_cY), theta, 1.0)
    
    for i in range(len(x_points)):
        x, y = x_points[i], y_points[i]
        v = [x, y, 1.0]
        calculated = np.dot(M, v)
        
        new_x.append(calculated[0])
        new_y.append(calculated[1])
        
    return new_x, new_y

# json format
"""
{
"img_294.jpeg29490":{
    "fileref":"",
    "size":29490,
    "filename":"img_294.jpeg",
    "base64_img_data":"",
    "file_attributes":{},
    "regions":{
        "0":{
            "shape_attributes":{
                "name":"polygon",
                "all_points_x":[300,280,427,447,300],
                "all_points_y":[304,336,316,294,304]
            },
            "region_attributes":{"car":"1"}
            },
        }
    }
}
"""

if __name__ =='__main__':
    
    # input original data's directory .json file
    JSON_DIR = "./own_data/train/via_region_data.json"
    IMG_DATA_DIR = "./own_data/train/"
    SAVE_DIR = "./own_data/train/"
    
    THETA = -60
    
    json_data = open(JSON_DIR).read()
    data = json.loads(json_data)
    
    # new json file
    NEW_JSON = {}
    
    # set large number...
    IMG_CNT = 2288
    
    # read image file
    for i, f in enumerate(list(data.items())):
        # rotated image name setting
        IMG_CNT += 1
        img_format = f[1]["filename"].split('.')[1]
        new_img_name = "img_" + str(IMG_CNT) + "." + str(img_format)

        # read image and shape
        origin_img = cv2.imread(IMG_DATA_DIR + f[1]["filename"], cv2.IMREAD_UNCHANGED)
        print("filename = ", f[1]["filename"], '\n')
        (img_h, img_w) = origin_img.shape[:2]
        (img_cX, img_cY) = (img_w//2, img_h//2)
        
        # roate origin image
        rotated_img = rotate_img(origin_img, THETA)
        
        # rotated_image save & get rotated image size(Byte)
        cv2.imwrite(SAVE_DIR + new_img_name, rotated_img)
        file_size = os.path.getsize(SAVE_DIR + new_img_name)

        # ================================ test show ======================================
        # image show in plt
        # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(18, 24))
        # plt.tight_layout()
        
        # ax1.imshow(origin_img, aspect='equal')
        # ax2.imshow(rotated_img, aspect='equal')
        
        # ax1.axis('off')
        # ax2.axis('off')
        # =======================================================================

        # one file dictionary for region json file
        obj_dict = {}
        
        NEW_JSON[new_img_name + str(file_size)] = obj_dict
        
        obj_dict["fileref"] = ""
        obj_dict["size"] = file_size
        obj_dict["filename"] = new_img_name
        obj_dict["base64_img_data"] = ""
        obj_dict["file_attributes"] = {}
        obj_dict["regions"] = f[1]["regions"]
        
        # rotate each polygon in every image
        for j, polygon in enumerate(f[1]["regions"].values()):
            x_points = polygon["shape_attributes"]["all_points_x"]
            y_points = polygon["shape_attributes"]["all_points_y"]
            
            # one polygon
            rotated_x, rotated_y = rotate_points(x_points, y_points, img_cX, img_cY, img_h, img_w, THETA)
            
            # change shape data to rotated_points
            obj_dict["regions"][str(j)]["shape_attributes"]["all_points_x"] = rotated_x
            obj_dict["regions"][str(j)]["shape_attributes"]["all_points_y"] = rotated_y
            # for rotated points checking =============================
            # temp_o = []
            # temp_n = []
            # for k in range(len(x_points)):
                # temp_o.append([x_points[k], y_points[k]])
                # temp_n.append([rotated_x[k], rotated_y[k]])
                
            # ax1.add_patch(mpatches.Polygon(temp_o, lw=3.0, fill=False, color='green'))
            # ax2.add_patch(mpatches.Polygon(temp_n, lw=3.0, fill=False, color='green'))
            #=============================================================
        
        # print("================\n",NEW_JSON,"==========================\n\n\n\n\n")
    with open(SAVE_DIR + 'via_region_data'+str(THETA)+'.json', 'w') as f:
        json.dump(NEW_JSON, f, ensure_ascii=False)
            
            
