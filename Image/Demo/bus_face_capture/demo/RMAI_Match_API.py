import lap
import numpy as np
from sympy import capture

class MatchApi():
    """
    MatchApi
    """
    def __init__(self):
        # option
        self.option_init()

        # param
        self.param_init()


    def option_init(self):
        self.wait_frame_threshold = 200
        self.match_threshold = 3 * 1000


    def param_init(self):
        self.wait_frame_num = 0
        self.face_capture_list = []
        self.p3d_capture_list = []


    def run(self, face_capture_dict, p3d_capture_dict, end_bool=False):
        
        # init
        capture_bool = False
        match_dict = {}

        if len(face_capture_dict.keys()):
            capture_bool = True
            for key, face_capture_idy in face_capture_dict.items():
                self.face_capture_list.append(face_capture_idy)
        
        if len(p3d_capture_dict.keys()):
            capture_bool = True
            for key, p3d_capture_idy in p3d_capture_dict.items():
                self.p3d_capture_list.append(p3d_capture_idy)
        
        if capture_bool:
            self.wait_frame_num = 0
        else:
            self.wait_frame_num += 1

        if end_bool or self.wait_frame_num >= self.wait_frame_threshold:

            if not ( len(self.face_capture_list) or len( self.p3d_capture_list )):
                
                return match_dict

            dists = time_distance(self.face_capture_list, self.p3d_capture_list)
            matches, u_face, u_p3d = linear_assignment(dists, thresh=self.match_threshold)

            for iface, ip3d in matches:

                # bbox_capture_dict
                bbox_capture_dict = {}
                bbox_capture_dict['face_id'] = 0
                bbox_capture_dict['face_loc'] = []
                bbox_capture_dict['face_frame_idx'] = []
                bbox_capture_dict['face_img'] = 0

                bbox_capture_dict['p3d_id'] = 0
                bbox_capture_dict['p3d_loc'] = []
                bbox_capture_dict['p3d_frame_idx'] = []
                bbox_capture_dict['p3d_img'] = 0

                bbox_capture_dict['face_id'] = self.face_capture_list[iface]['id']
                bbox_capture_dict['face_loc'] = self.face_capture_list[iface]['loc']
                bbox_capture_dict['face_frame_idx'] = self.face_capture_list[iface]['frame_idx']
                bbox_capture_dict['face_img'] = self.face_capture_list[iface]['img']

                bbox_capture_dict['p3d_id'] = self.p3d_capture_list[ip3d]['id']
                bbox_capture_dict['p3d_loc'] = self.p3d_capture_list[ip3d]['loc']
                bbox_capture_dict['p3d_frame_idx'] = self.p3d_capture_list[ip3d]['frame_idx']
                bbox_capture_dict['p3d_img'] = self.p3d_capture_list[ip3d]['img']

                match_dict[bbox_capture_dict['p3d_id']] = bbox_capture_dict

            self.wait_frame_num = 0
            self.face_capture_list = []
            self.p3d_capture_list = []

        return match_dict      


def time_distance(face_capture_list, p3d_capture_list):

    times_cost = np.zeros((len(face_capture_list), len(p3d_capture_list)), dtype=np.float)

    for idx in range(len(face_capture_list)):

        for idy in range(len(p3d_capture_list)):
            
            times_cost[idx][idy] = abs(face_capture_list[idx]['frame_idx'] - p3d_capture_list[idy]['frame_idx'])
    
    return times_cost


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b