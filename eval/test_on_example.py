# %%
# change the path
import os
import imageio
import cv2
import torch
import numpy as np
from einops import rearrange
import yaml
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
from tqdm import tqdm
import matplotlib.image as mpimg
import scipy.spatial.distance as sci_dis
from utils.map import process_image, dilate_tensor
import utils.distance_estimation as dis
from utils.common import intersect2d
from utils.display import makr_robot_loc
from utils.create_training_data import get_pixel_number_from_name
from model.semantic_anticipator import (   
    SemAnticipator,
    SemAnticipationWrapper,
)
import time
from config.default import Config as CN
import json

# set up the parameters
with open("config/default.yaml", "r") as yamlfile:
    config_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")
print(config_yaml)

config = CN(config_yaml)
IF_IM = config.SEMANTIC_ANTICIPATOR.IF_IM
device = config.SEMANTIC_ANTICIPATOR.device


map_size = config.SEMANTIC_ANTICIPATOR.map_size
map_scale = config.SEMANTIC_ANTICIPATOR.map_scale
projection = dis.GTEgoMap(map_size=map_size,map_scale=map_scale)

imgh = config.SEMANTIC_ANTICIPATOR.imgh
imgw = config.SEMANTIC_ANTICIPATOR.imgw

# font type
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 0.5
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 1


img_mean = config.IMG_NORMALIZATION.res18_img_mean
img_std = config.IMG_NORMALIZATION.res18_img_std    
sem_model = SemAnticipationWrapper(
SemAnticipator(config), map_size, (imgh, imgw)
).to(device)
checkpoint = torch.load(config.SEMANTIC_ANTICIPATOR.ckpt,map_location=device)
sem_model.load_state_dict(checkpoint['model_state_dict'])
sem_model.eval()  
dataset_path = config.SEMANTIC_ANTICIPATOR.dataset

x_matrix = (np.arange(map_size) - np.int32(map_size/2)-1)[np.newaxis]*map_scale*(np.ones(map_size)[:,np.newaxis])
x_tensor = torch.tensor(x_matrix[np.newaxis,np.newaxis]).to(device)
z_matrix = -(map_size - np.arange(map_size)-1)[:,np.newaxis]*np.ones(map_size)[np.newaxis,:]*map_scale
z_tensor = torch.tensor(z_matrix[np.newaxis,np.newaxis]).to(device)

clip = '0000/'
meta_file_location = dataset_path + '/stats_info/' + clip 
pose_camera = np.load(meta_file_location + 'camera_spec.npy',allow_pickle=True).tolist()
pose_ped =  np.load(meta_file_location + 'peds_infos.npy',allow_pickle=True).tolist()

location_camera = pose_camera['position']
location_ped =  pose_ped['positions']
frame_list = sorted(os.listdir(dataset_path + '/object_id/'  + clip ))

img_mean = config.IMG_NORMALIZATION.res18_img_mean
img_std = config.IMG_NORMALIZATION.res18_img_std      

with open(dataset_path+'/stats_info/' + clip + 'semantic_id_to_name.json') as f:
    id_to_name = json.load(f)
clip_result = []   

for frame in tqdm(frame_list,desc="Frame"):
    
    fram_num = int(frame.split('.')[0]) 
    #print(fram_num)

    ground_truth = np.uint8(cv2.imread(dataset_path + '/object_id/'+ clip + frame)[:,:,0])
    
    depth_img = cv2.imread(dataset_path + '/depth/'+ clip + frame)
    depth_array = depth_img[:,:,0]/255.
    
    ground_truth_points = {}
    gt_mask = ground_truth > 0
    object_ids = np.unique(ground_truth)
    
    img_path = dataset_path + '/rgb/' + clip  + frame.split('.')[0]+'.jpg'
    bgr_image = cv2.imread(img_path)
    rbg_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv2.resize(rbg_image, (rbg_image.shape[1]//2, rbg_image.shape[0]//2), interpolation = cv2.INTER_AREA)
    
    mask_file_path = dataset_path + '/3dc_anno/' + clip + frame[1:]
    mask_img = cv2.imread(mask_file_path)
    mask = np.sum(mask_img,axis=2) > 0
    # validation object:
    valid_object_ids = []
    for object_id in object_ids[1:]:
        location = location_ped[object_id-1][fram_num]
        dis_gt_2 = location[0]*location[0] + location[2]*location[2]
        dis_gt = np.sqrt(dis_gt_2)
        object_mask =  ground_truth == object_id
        object_name = id_to_name[str(object_id)]
        pixel_th = get_pixel_number_from_name(object_name)
        if dis_gt > 1:
            beta = 0.5
        if dis_gt < 1:
            beta = 0.2
            
        if np.sum(object_mask)*dis_gt_2 > beta*pixel_th:
            valid_object_ids.append(object_id)



    if np.all(mask == False):
        gt_mask = ground_truth > 0
        if np.all(mask == False) and np.all(gt_mask == False):
            fig = plt.figure(figsize=(10,10))
            plt.xlim([0, map_size])
            plt.ylim([0, map_size])
            plt.tight_layout()
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            makr_robot_loc(map_size)
            plt.savefig("middle.png", bbox_inches = 'tight',pad_inches = 0)
            plt.close(fig)
            img = cv2.imread('middle.png')
            clip_result.append(img)
        else:
            img = cv2.imread('middle.png')
            ground_truth_points = {}
            fig = plt.figure(figsize=(10,10))
            plt.xlim([0, map_size])
            plt.ylim([0, map_size])
            plt.tight_layout()
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            for object_id in object_ids[1:]:
                mask = ground_truth == object_id
                ego_map_gt_object = projection.get_observation(depth_array*mask)
                points = np.array(np.where(ego_map_gt_object[:,:,0] > 0)).T
                if points.shape[0] > 0:
                    clf = LocalOutlierFactor(n_neighbors=int(points.shape[0]/20), contamination=0.1)
                    y_pred = clf.fit_predict(points)
                    
                    plt.xlim([0, map_size])
                    plt.ylim([0, map_size])
                    plt.scatter(points[:,1],points[:,0], color="k", s=1.0)
                    plt.scatter(points[y_pred > 0,1],points[y_pred > 0,0], color="r", s=100)
                    
                    ground_truth_points[object_id] = points[y_pred>0]
                    orignal_points = points[y_pred>0]
                    orignal_points_img = np.copy(orignal_points)
                    orignal_points_img[:,0] = map_size - orignal_points[:,0]
                    cluster_centre = np.average(orignal_points_img, axis=0)
                    org = (int(cluster_centre[1]/map_size*img.shape[1]) + 50,int(cluster_centre[0]/map_size*img.shape[0]))
                    image = cv2.putText(img, f'Error: N/A', org, font, 
                                    fontScale, color, thickness, cv2.LINE_AA)
            makr_robot_loc(map_size)
            plt.savefig("middle.png", bbox_inches = 'tight',pad_inches = 0)
            plt.close(fig)
            img = cv2.imread('middle.png')
            clip_result.append(img)
        continue
    
    ego_map_gt = projection.get_observation(depth_array*mask)
    points_sensor = np.array(np.where(ego_map_gt[:,:,0] > 0)).T
    dilation_kernel = np.ones((5, 5))
    dilation_mask =  cv2.dilate(ego_map_gt[:,:,0], dilation_kernel, iterations=4,
            ).astype(np.float32)
    
    if IF_IM:
        x = {}
        rbg_image_bt = torch.tensor(rearrange(rbg_image, "h w c -> () h w c"))
        x["rgb"] = process_image(rbg_image_bt, img_mean, img_std).to(device)
        ego_map_gt_bt = torch.FloatTensor(rearrange(ego_map_gt[:,:,0], "h w -> () () h w ")/255.).to(device)
        location_mask = (dilate_tensor(ego_map_gt_bt,7,iterations=3) > 0 )
        x["ego_dyn"] = ego_map_gt_bt
        mytime = time.time()
        pre = sem_model.forward(x)
        mytime2 = time.time()
        result = pre['sem_estimate'][0,0].cpu().detach().numpy()*dilation_mask
        ego_map_gt = result[:,:,np.newaxis]
        
    points = np.array(np.where(ego_map_gt[:,:,0] > 0.3)).T    
    if points.shape[0] < 2:
        img = cv2.imread('middle.png')
        ground_truth_points = {}
        fig = plt.figure(figsize=(10,10))
        plt.xlim([0, map_size])
        plt.ylim([0, map_size])
        plt.tight_layout()
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        for object_id in object_ids[1:]:
            mask = ground_truth == object_id
            ego_map_gt_object = projection.get_observation(depth_array*mask)
            points = np.array(np.where(ego_map_gt_object[:,:,0] > 0)).T
            if points.shape[0] > 0:
                clf = LocalOutlierFactor(n_neighbors=int(points.shape[0]/20), contamination=0.1)
                y_pred = clf.fit_predict(points)
                
                plt.xlim([0, map_size])
                plt.ylim([0, map_size])
                plt.scatter(points[:,1],points[:,0], color="k", s=1.0)
                plt.scatter(points[y_pred > 0,1],points[y_pred > 0,0], color="r", s=100)
                
                ground_truth_points[object_id] = points[y_pred>0]
                orignal_points = points[y_pred>0]
                orignal_points_img = np.copy(orignal_points)
                orignal_points_img[:,0] = map_size - orignal_points[:,0]
                cluster_centre = np.average(orignal_points_img, axis=0)
                org = (int(cluster_centre[1]/map_size*img.shape[1]) + 50,int(cluster_centre[0]/map_size*img.shape[0]))
                image = cv2.putText(img, f'Error: N/A', org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)
        makr_robot_loc(map_size)
        plt.savefig("middle.png", bbox_inches = 'tight',pad_inches = 0)
        plt.close(fig)
        img = cv2.imread('middle.png')
        clip_result.append(img)
        continue
    
    clf1 = LocalOutlierFactor(n_neighbors=np.min([int(points.shape[0]/5+1),50]), contamination=0.3)
    y_pred = clf1.fit_predict(points)
    clustering = DBSCAN(eps=3, min_samples=2).fit(points[y_pred > 0])
    relativ_average_cors = {} 
    valid_clusters = []
    for lable in np.unique(clustering.labels_):
        o_cluster_points = points[y_pred > 0][np.where(clustering.labels_ == lable)]
        if o_cluster_points.shape[0] > 2:
            valid_clusters.append(lable)
            shift_centre = []
            cluster_points = points[y_pred > 0][np.where(clustering.labels_ == lable)] - [map_size, int(map_size/2)+1]
            cluster_points = cluster_points*map_scale
            cluster_centre = np.average(cluster_points, axis=0)
            if IF_IM:
                clusters_points_img = torch.zeros((map_size, map_size)).to(device)
                clusters_points_img[(o_cluster_points[:,0].tolist(),o_cluster_points[:,1].tolist())] = 1
                clusters_points_img = (dilate_tensor(clusters_points_img[None,None],5,1))[:,:]
                dis_esti_bt = clusters_points_img*pre['sem_estimate'][0,0]
                x_pt = (torch.sum(x_tensor*dis_esti_bt)/(torch.sum(dis_esti_bt))).cpu().detach().numpy()
                z_pt = (torch.sum(z_tensor*dis_esti_bt)/(torch.sum(dis_esti_bt))).cpu().detach().numpy()
            else:
                x_pt = cluster_centre[1]
                z_pt = cluster_centre[0]
            relativ_average_cors[lable] = {
                    'orignal_points':o_cluster_points,
                    'cluster_points':cluster_points,
                    'cluster_centre':[x_pt, z_pt],
                    'cluster_centre_no_shift':shift_centre,
                    }
            
    # calculate points of the ground truth object
    object_ids = np.unique(ground_truth)
    ground_truth_points = {}
    fig = plt.figure(figsize=(10,10))
    plt.xlim([0, map_size])
    plt.ylim([0, map_size])
    plt.scatter(points_sensor[:,1],map_size - points_sensor[:,0], c='r', s=0.3, alpha=0.2)
    plt.scatter(points[y_pred > 0,1],map_size - points[y_pred > 0,0], c='b', s=0.8, alpha=0.3)
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for object_id in object_ids[1:]:
        #show the masked depth image
        #print('show the masked depth image')
        mask = ground_truth == object_id
        #plt.imshow(depth_array*mask)
        #print(np.sum(depth_array*mask)/np.sum(mask)*10)
        ego_map_gt_object = projection.get_observation(depth_array*mask)
        points = np.array(np.where(ego_map_gt_object[:,:,0] > 0)).T
        if points.shape[0] > 0:
            clf = LocalOutlierFactor(n_neighbors=int(points.shape[0]/20), contamination=0.1)
            y_pred = clf.fit_predict(points)
            #plt.scatter(points[y_pred > 0,1],points[y_pred > 0,0], color="r", s=1.0)
            
            ground_truth_points[object_id] = points[y_pred>0]
    makr_robot_loc(map_size)
    plt.savefig("middle.png", bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)
    
    id2lable = {}
    for object_id in ground_truth_points.keys():
        best_match = {
                    'lable' : -1,
                    'points_match':0
                        } # -1 no match, 0 pints match
        for lable in np.unique(clustering.labels_):
            points_pre = relativ_average_cors[lable]['orignal_points']
            points_gt = ground_truth_points[object_id]
            points_matched = intersect2d(points_pre,points_gt)
            #print(points_matched.shape)
            if points_matched.shape[0] > best_match['points_match']:
                best_match['lable'] = lable
                best_match['points_match'] = points_matched.shape[0]
        id2lable[object_id] = best_match['lable']

    eval_record = []
    for id in id2lable.keys():
        if id2lable[id] == -1:
            eval_record.append(np.array([-1,0,0]))
        else:
            location = location_ped[id-1][fram_num]
            distance_gt = np.sqrt(location[0]*location[0] + location[2]*location[2])
            dis_est = np.sqrt(np.sum(np.square(relativ_average_cors[id2lable[id]]['cluster_centre'])))
            eval_record.append(np.array([1,dis_est,distance_gt]))
            
    img = cv2.imread('middle.png')
    for key in relativ_average_cors.keys():
        if key < 0:
            continue
        orignal_points = relativ_average_cors[key]['orignal_points']
        orignal_points_img = np.copy(orignal_points)
        orignal_points_img[:,0] = map_size - orignal_points[:,0]

        cluster_centre = np.average(orignal_points_img, axis=0)
        cluster_centre = (int(cluster_centre[1]/map_size*img.shape[1]),int((map_size - cluster_centre[0])/map_size*img.shape[0]))
        distance_ct = sci_dis.cdist(orignal_points_img,[np.average(orignal_points_img, axis=0)])
        r = int(max(distance_ct)*img.shape[0]/map_size)
        color = (255, 111, 0)
        img = cv2.circle(img, cluster_centre, r, color, 2)

    for index,key in enumerate(ground_truth_points.keys()):
        if key < 0:
            continue
        orignal_points = ground_truth_points[key]
        orignal_points_img = np.copy(orignal_points)
        orignal_points_img[:,0] = map_size - orignal_points[:,0]
        cluster_centre = np.average(orignal_points_img, axis=0)
        org = (int(cluster_centre[1]/map_size*img.shape[1]) + 50,int((map_size - cluster_centre[0])/map_size*img.shape[0]))


        # Using cv2.putText() method
        if eval_record[index][0] > 0:
            error = np.abs(eval_record[index][1] - eval_record[index][2])
            img = cv2.putText(img, f'Error: {error:.3f} E-Dist: {eval_record[index][1]:.3f}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        else:
            if key in valid_object_ids:
                r = 5
                color = (255, 255, 0)
                img = cv2.circle(img, org, r, color, 2)
                img = cv2.putText(img, f'Error: N/A', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
    clip_result.append(img)

print(len(clip_result))
writer = imageio.get_writer('test.mp4', fps=24)
for i in tqdm(range(len(clip_result))):
    writer.append_data(clip_result[i])
writer.close()
        
# %%
