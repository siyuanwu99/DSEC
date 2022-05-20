import yaml
import torch

def extract_projmat(path='./cam_to_cam.yaml'):
    with open(path) as file:
        documents = yaml.load(file,Loader=yaml.FullLoader)
        # print(documents['disparity_to_depth']['cams_03'])

        if torch.cuda.is_available():
            Q=torch.tensor(documents['disparity_to_depth']['cams_03']).cuda()
        else:
            Q=torch.tensor(documents['disparity_to_depth']['cams_03'])
        return Q

if __name__ == '__main__':
    Q=extract_projmat()
    print(Q)