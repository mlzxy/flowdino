import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
import os.path as osp
from tqdm import tqdm
import ujson as json
from numpy import linalg as LA
from glob import glob

from dataset.helper import CacheOpen
from dataset.optical_flow import unpack_flow_to_float


def read_folder(folder, ext='.jpg', limit=10, index=0):
    frames = glob(osp.join(folder, '*' + ext))
    flows = glob(osp.join(folder, '*.tiff'))
    assert len(frames) > 0 and len(flows) > 0 and len(frames) == len(flows) + 1
    frames = sort_frames(frames)[index:limit]
    flows = sort_frames(flows)[index:max(len(frames)-1, 1)]
    load = CacheOpen()
    frames = torch.from_numpy(np.array([load(f) for f in frames])).permute(0, 3, 1, 2)
    flows = torch.from_numpy(np.array([unpack_flow_to_float(load(f)) for f in flows]))
    return frames, flows


def sort_frames(paths):
    paths = list(set(paths))
    return sorted(paths, key=lambda x: int(osp.basename(x).split('.')[0]))



class VideoDataset(Dataset):
    """
    schema for the json
    {
        path: string # root path of all files
        videos: [
            {
                path: string # this path works as id

                # the id of the frame indicate frame order
                # assume sorted in ascending order
                # if T frames, then we have T-1 flows, respectively

                # normally, the absoluate path for each frame is the joining of root path + video path + frame id
                # as well as flows

                # but if any path is given as an absolute path, then the preceding path is ignored
                # support we have a video path as "/video" and root path as "/home", then we will ignore the root path 
                # as well as frame id

                # flows are stored as LA image
                frames: string[]
                flows: string[]
            }
        ]
    }
    """

    def __init__(self, json_files, flow=False, cache_size=-1, transform=None, return_norm_stats=False, normalize_flow=False):
        self._load = CacheOpen(size=cache_size)
        self.flow = flow
        self.flow_size = 1
        if not isinstance(json_files, (list, tuple)):
            json_files = [json_files]
        meta_list = [json.loads(Path(json_file).read_text()) for json_file in json_files]
        self.videos = [{**info, 'parent': js['path']} for js in meta_list for info in js['videos'] if len(info['frames']) >= self.flow_size]
        self.return_norm_stats = return_norm_stats
        self.normalize_flow = normalize_flow

        def default_transform(frame, flow=None):
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            if flow is None:
                return frame 
            else:
                return frame, torch.from_numpy(flow).permute(2, 0, 1)

        self.transform = transform or default_transform

    def __len__(self):
        return len(self.videos) 

    def __getitem__(self, index):
        video = self.videos[index]
        prefix = osp.join(video['parent'], video['path'])
        ind = random.randint(0, max(len(video['frames']) - 1 - self.flow_size, 0))
        frame_path = video['frames'][ind]
        frame_array = self._load(osp.join(prefix, frame_path))

        if self.flow:
            flow_path = video['flows'][ind]
            flow = unpack_flow_to_float(self._load(osp.join(prefix, flow_path)).view('int32'))
            if self.normalize_flow:
                flow = flow - flow.reshape(2, -1).mean(axis=1)[:, None, None]

            if self.return_norm_stats:
                max_norm = LA.norm(flow, axis=0).flatten().max().reshape(1)

            flow_array = flow.transpose(1, 2, 0)
            frame_array, flow_array = self.transform(frame_array, flow_array)

            if isinstance(frame_array, (list, tuple)):
                return frame_array + flow_array + [torch.from_numpy(max_norm), ]
            else:
                if self.return_norm_stats:
                    return frame_array, flow_array, torch.from_numpy(max_norm)
                else:
                    return frame_array, flow_array
        else:
            return self.transform(frame_array, None)


    @staticmethod
    def organize(jsons_dir, output_json_file, use_absolute_video_path=False, output_indent=True):
        """
        merge multiple existing json files (in the given dir) into a single one and ensure a clean state 
        (will perform numeric sort in frame ids)

        all these jsons are assumed to have the same root path, otherwise, set `use_absolute_video_path` to True
        """
        jsons = []

        for jd in jsons_dir.split(','):
            if jd.endswith('.json'):
                jsons.append(jd)
            else:
                jsons += glob(osp.join(jd, "*.json"))

        root = ''
        videos = {}

        for js in tqdm(jsons, desc="collecting all jsons"):
            info = json.loads(Path(js).read_text())

            if not use_absolute_video_path:
                if not root:
                    root = info['path']
                else:
                    assert info['path'] == root

            for video in info['videos']:
                if use_absolute_video_path:
                    vpath = osp.join(video['path'], video['path'])
                else:
                    vpath = video['path']

                if vpath not in videos:
                    videos[vpath] = {
                        'frames': [],
                        'flows': []
                    }

                videos[vpath]['frames'] += video.get('frames', [])
                videos[vpath]['flows'] += video.get('flows', [])
                if video.get('delete', False):
                    videos[vpath]['delete'] = True
        
        videos = {k:v for k,v in videos.items() if not v.get('delete', False)}

        for vpath in tqdm(list(videos.keys()), desc="post-processing"):
            if len(videos[vpath]['flows']) > 0:
                assert len(videos[vpath]['flows']) == len(
                    videos[vpath]['frames']) - 1
                videos[vpath]['flows'] = sort_frames(videos[vpath]['flows'])
            videos[vpath]['frames'] = sort_frames(videos[vpath]['frames'])

        videos = [dict(path=vpath, **vinfo) for vpath, vinfo in videos.items()]
        print(f'{len(videos)} videos found.')
        Path(output_json_file).write_text(json.dumps(
            dict(path=root, videos=videos), indent=4 if output_indent else None))

