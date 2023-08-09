"""
运行环境准备
1. 操作系统：Windows 10 x64 / Ubuntu 20.04 下测试通过
2. python 3.8
3. 需要用 pip 安装的第三方依赖
3.1 av==10.0.0
3.2 opencv-python-headless~=4.5
3.3 tqdm==4.65.0
使用方法:
1. 运行 python3 NorthAmerica_1448_h264_demo.py --help 将会打印出可用的命令行参数及其注释
2. 运行 python3 NorthAmerica_1448_h264_demo.py {h264文件} --summary 打印出 h264 文件所包含的通道，以及每个通道包含的 ID 列表
   如: python3 NorthAmerica_1448_h264_demo.py D:/test.h264 --summary
3. 运行 python3 NorthAmerica_1448_h264_demo.py {h264文件} --channel={channel} --id ... 进行绘制。结果会保存在当前目录下的 result.mp4
   如: python3 NorthAmerica_1448_h264_demo.py D:/test.h264 --channel=0 --id 21 23 25
       以上命令表示：将 test.h264 文件中第 0 通道里面包含 id 号为 21、23、25 的车辆进行绘制
"""

import argparse
import io

import av
import cv2
import ctypes
import numpy as np
import itertools
import tqdm
from collections import namedtuple, defaultdict
from typing import BinaryIO, List


# MetaHeader 元数据头:
# |-- 帧类型 --|-- 视频帧长度 --|-- 校验码 --|-- 扩展数据长度 --|-- 扩展数据数量 --|
# |   32bit   |     24bit    |   8bit    |      24bit     |      8bit     |
class MetaHeader(ctypes.LittleEndianStructure):
    MASK_FRAME_TYPE_CHANNEL = 0x000000FF
    MASK_FRAME_TYPE_TYPE = 0xFFFFFF00
    FRAME_TYPE_I = 0x636432
    FRAME_TYPE_P = 0x636433
    _pack_ = 1
    _fields_ = [
        ("frame_type", ctypes.c_uint, 32),
        ("frame_len", ctypes.c_uint, 24),
        ("stream_exam", ctypes.c_uint, 8),
        ("extend_len", ctypes.c_uint, 24),
        ("extend_count", ctypes.c_uint, 8)
    ]


# InfoTypeHeader 扩展信息头:
# |-- 扩展数据项类型 --|-- 扩展数据项长度(包含本信息头) --|
# |       8bit      |             24bit            |
class InfoTypeHeader(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("info_type", ctypes.c_uint, 8),
        ("info_len", ctypes.c_uint, 24)
    ]


# AIInfoHead AI 算法信息元数据:
# |-- 保留数据 --|-- 算法帧时间 --|-- AI 帧叠加大小 --|
# |   32bit    |     64bit    |      不定长       |
class AIInfoHead(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("reserve", ctypes.c_uint, 32),
        ("pts", ctypes.c_ulonglong, 64),
        ("ai_info_len", ctypes.c_uint, 32)
    ]


VideoFrameInfo = namedtuple("VideoFrameInfo",
                            field_names=["channel", "pts", "frame_type", "frame_pos"])
AIFrameInfo = namedtuple("AIFrameInfo",
                         field_names=["channel", "pts", "ai_info"])


class H264File(object):
    def __init__(self, filename: str):
        self._filename = filename
        self._decoder_map = {}
        self._track_x_coord = 10
        self._video_frame_meta = defaultdict(list)
        self._ai_frame_meta = defaultdict(list)
        self._pre_parse()
        # 扔掉最开始的 P 帧，否则无法解码
        for c in self._video_frame_meta:
            for i, frame in enumerate(self._video_frame_meta[c]):
                if frame.frame_type == MetaHeader.FRAME_TYPE_I:
                    self._video_frame_meta[c] = self._video_frame_meta[c][i:]
                    break

    def _pre_parse(self):
        frame_idx = 0
        with open(self._filename, "rb") as fp:
            # 每帧数据格式
            # 其中 AI 信息包含在扩展数据中，码流数据为实际的视频帧数据
            # |-- MetaHeader --|-- 扩展数据 --|-- 码流数据 --|
            # |     12Byte     |    不定长    |    不定长   |
            while True:
                # 找到最近的帧起始位置
                flag, extra_offset = H264File.move_fileptr_to_framehead(fp)
                if not flag:
                    break
                header_raw = fp.read(12)  # 获取帧头元数据
                header = MetaHeader.from_buffer(bytearray(header_raw))
                # 根据帧头元数据获取扩展数据
                extend_data_raw = fp.read(header.extend_len)
                # 根据帧头元数据获取码流数据
                frame_data_start = fp.tell()
                frame_data_raw = fp.read(header.frame_len)
                # 已经到达文件末尾，读取的数据可能不全
                if len(frame_data_raw) != header.frame_len:
                    break
                frame_data_end = fp.tell()
                frame_pos = (frame_data_start, frame_data_end - frame_data_start)
                channel = header.frame_type & MetaHeader.MASK_FRAME_TYPE_CHANNEL
                frame_type = (header.frame_type & MetaHeader.MASK_FRAME_TYPE_TYPE) >> 8

                curr = 0
                tracks = []
                frame_pts = None
                # extend_data_raw 中保存有 extend_count 个数据项，因此迭代着去获取
                for _ in range(header.extend_count):
                    info_type_header = InfoTypeHeader.from_buffer(
                        bytearray(extend_data_raw[curr:curr + ctypes.sizeof(InfoTypeHeader)]))
                    # 13 为 AI 算法信息, 跳过其他类型的处理
                    if info_type_header.info_type == 13:
                        ai_curr = curr + ctypes.sizeof(InfoTypeHeader)
                        # AI 信息元数据
                        ai_info_header = AIInfoHead.from_buffer(bytearray(
                            extend_data_raw[ai_curr:ai_curr + ctypes.sizeof(AIInfoHead)]))
                        ai_curr += ctypes.sizeof(AIInfoHead)
                        ai_pts = ai_info_header.pts
                        # AI 信息原始数据
                        ai_info_raw = bytearray(extend_data_raw[ai_curr: ai_curr + ai_info_header.ai_info_len])
                        # 跳过前 21 个字节，都是一些额外信息没有用
                        ai_info_raw = ai_info_raw[21:]
                        idx = 0
                        # |-- AI 信息类型 --|-- AI 信息个数 --|-- AI 信息总字节 --|-- AI 信息数据 --|
                        # |      1Byte    |      1Byte     |       4Byte     |      不定长     |
                        while idx < len(ai_info_raw):
                            # AI 叠加信息类型
                            data_type = int.from_bytes(ai_info_raw[idx:idx + 1], byteorder="little", signed=False)
                            idx += 1
                            # AI 叠加信息个数
                            data_count = int.from_bytes(ai_info_raw[idx:idx + 1], byteorder="little", signed=False)
                            idx += 1
                            # AI 叠加信息总长度
                            data_len = int.from_bytes(ai_info_raw[idx: idx + 4], byteorder="little", signed=False)
                            idx += 4
                            # 只需要关注 data_type == 4 的情况，即字符串数据
                            if data_type == 4:
                                # 字符数据的排列:
                                # |--------- xy 坐标数组 -------|------ reserve -----|-- 实际的文本数据 --|
                                # |  2 * 2 * data_count Byte   |  data_count Byte  |       不定长      |
                                # 其中坐标数据类型为 uint16, 因此一个坐标使用两个 uint16, 即一个坐标占用 4 字节
                                xy_coords = np.frombuffer(ai_info_raw[idx:idx + 2 * 2 * data_count],
                                                          dtype=np.uint16).reshape(-1, 2)
                                idx += 2 * 2 * data_count
                                # 跳过 reserve
                                idx += data_count
                                # 取出当前帧的所有字符串
                                str_len = data_len - (2 * 2 * data_count + data_count)
                                all_strs = [s.decode("ascii") for s in ai_info_raw[idx:idx + str_len].split(b'\x00')]
                                # 只提取 x 坐标等于一个固定值的字符串
                                tracks = [[int(i) for i in s.split(",")] for (coord, s) in zip(xy_coords, all_strs) if
                                          coord[0] == self._track_x_coord]
                                self._ai_frame_meta[channel].append(AIFrameInfo(channel, ai_pts, tracks))
                                break
                            idx += data_len
                    # 3 为视频帧时间信息
                    elif info_type_header.info_type == 3:
                        time_curr = curr + ctypes.sizeof(InfoTypeHeader)
                        frame_pts = int.from_bytes(extend_data_raw[time_curr:time_curr + 8], byteorder="little",
                                                   signed=False)
                    # 根据 info_len 进行指针跳跃
                    curr += info_type_header.info_len
                frame_idx += 1
                self._video_frame_meta[channel].append(VideoFrameInfo(channel, frame_pts, frame_type, frame_pos))

    def summary(self):
        print(f"{self._filename} summary:")
        print(f"channels: {list(self._video_frame_meta.keys())}")
        for c in self._video_frame_meta:
            frames = self._video_frame_meta[c]
            ai_frames = self._ai_frame_meta[c]
            print(f"channel[{c}]:")
            print(f"\tframe count: {len(frames)}")
            print(
                f"\tunique id(s): {list(set(list(itertools.chain.from_iterable([[s[0] for s in frame.ai_info] for frame in ai_frames]))))}")

    def draw(self, channel, id_list: List[int]):
        assert channel in self._video_frame_meta, "h264 文件不包含选择的 channel, 请运行 --summary 进行检查"
        assert len(id_list) > 0, "输入 id 为空, 请检查 --id 参数"
        frames: List[VideoFrameInfo] = self._video_frame_meta[channel]
        ai_frames: List[AIFrameInfo] = self._ai_frame_meta[channel]
        # buffer = io.BytesIO()
        # code_ctx = av.open(buffer, format="h264", mode="r")
        code_ctx = av.CodecContext.create("h264", "r")
        cap = None
        with open(self._filename, "rb") as fp:
            for frame in tqdm.tqdm(frames, desc="drawing..."):
                if frame.frame_type not in (MetaHeader.FRAME_TYPE_I, MetaHeader.FRAME_TYPE_P):
                    continue
                frame_start, frame_size = frame.frame_pos
                fp.seek(frame_start, io.SEEK_SET)
                chunk = fp.read(frame_size)
                packets = code_ctx.parse(chunk)
                for packet in packets:
                    av_frames = code_ctx.decode(packet)
                    for av_frame in av_frames:
                        img = av_frame.to_ndarray(format="bgr24")
                        # 找到和视频 PTS 最接近的 AI 帧（且视频 pts 大于 AI pts）
                        # 每次重复循环，效率较低
                        if frame.pts < min(ai_frames, key=lambda f: f.pts).pts:
                            ai_info = []
                        else:
                            ai_info = min(ai_frames, key=lambda f: abs(f.pts - frame.pts)).ai_info
                        requested_info = [s for s in ai_info if s[0] in id_list]
                        if cap is None:
                            h, w = img.shape[:2]
                            cap = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (w, h))
                        for info in requested_info:
                            cv2.rectangle(img, info[1:3], info[3:5], color=(255, 0, 0), thickness=2)
                            cv2.putText(img, f"id: {info[0]}", info[3:5], cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                        color=(255, 0, 0), thickness=2)
                        cap.write(img)
        if cap is not None:
            cap.release()

    @staticmethod
    def move_fileptr_to_framehead(fr: BinaryIO):
        # 移动流指针到下一个帧头位置
        offset = 0
        per = fr.read(1)
        cur = fr.read(1)
        net = fr.read(1)
        pos = fr.read(1)

        while True:
            if (cur == b'2' or cur == b'3') and (net == b'd') and (pos == b'c'):
                fr.seek(-4, 1)  # 往前移动4个字节
                return True, offset
            per = cur
            cur = net
            net = pos
            pos = fr.read(1)
            offset += 1
            if len(pos) == 0 and len(fr.read()) == 0:
                break
        return False, offset


def main(args):
    filename = args.video
    h264_file = H264File(filename)
    if args.summary:
        h264_file.summary()
        exit(0)
    h264_file.draw(args.channel, args.id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="h264 文件路径")
    parser.add_argument("-c", "--channel", type=int, default=0, help="选择需要输出的通道")
    parser.add_argument("-i", "--id", type=int, default=None, nargs="+", help="需要绘制的 id (可以选择多个)")
    parser.add_argument("-s", "--summary", action="store_true", help="打印出文件概览信息")
    opts = parser.parse_args()
    if not opts.summary:
        assert opts.id is not None, "需要设置 --id 参数"
    main(opts)
