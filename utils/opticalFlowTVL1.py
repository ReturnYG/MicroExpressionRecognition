import os
from glob import glob
import cv2
import numpy as np


def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    print(frames)
    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=1):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def compute_TVL1_my(prev, curr, bound=1):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    hsv = np.zeros_like(curr)
    hsv[..., 1] = 255
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    # flow = (flow + bound) * (255.0 / (2 * bound))
    # flow = np.round(flow).astype(int)
    # flow[flow >= 255] = 255
    # flow[flow <= 0] = 0

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def save_flow(video_flows, flow_path):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return


if __name__ == '__main__':
    video_path = '/Users/returnyg/Desktop/作图用样本/EP02_01f'  # 预先提取好的视频帧
    save_path = '/Users/returnyg/Desktop/作图用样本/optical flow'  # 保存光流的路径
    extract_flow(video_path, save_path)
    u = cv2.imread('/Users/returnyg/Desktop/作图用样本/optical flow/u/000000.jpg')
    v = cv2.imread('/Users/returnyg/Desktop/作图用样本/optical flow/v/000000.jpg')
    uv = cv2.addWeighted(u, 0.5, v, 0.5, 0)
    cv2.imshow("u", np.asarray(u, dtype=np.uint8))
    cv2.imshow("v", np.asarray(v, dtype=np.uint8))
    cv2.imshow("uv", np.asarray(uv, dtype=np.uint8))
    cv2.waitKey(0)

