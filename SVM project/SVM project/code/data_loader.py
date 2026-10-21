import os
def get_image_mask_pairs(folder):
    files = os.listdir(folder)    
    images = []
    masks = []        
    for f in files:        
        if "_mask" not in f:
            img_path = os.path.join(folder, f)                        
# Corresponding mask file name
            mask_name = f.replace(".png", "_mask.png")
            mask_path = os.path.join(folder, mask_name)                        
            if os.path.exists(mask_path):
                images.append(img_path)
                masks.append(mask_path)            
            else:                
                print("can not find mask:", mask_name)        
    return images, masks

if __name__ == "__main__":
    train_folder = "../EWS-Dataset/train"
    images, masks = get_image_mask_pairs(train_folder)

    print("Number of pictures:", len(images))
    print("mask numbrs:", len(masks))
    print("example:")
    print(images[0])
    print(masks[0])
    