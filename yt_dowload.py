import os
import glob
import argparse
import yt_dlp

parser = argparse.ArgumentParser(description='YouTube video downloader parameters.')
parser.add_argument('--download_dir', required=True, help='target directory to save downloaded videos')
parser.add_argument('--youtube_idlist', required=True, help='a .txt file saving urls to all videos')
# parser.add_argument('--img_dir', help='target directory to save downsampled')
# parser.add_argument('--img_ext', default='jpg', help='image extension')
parser.add_argument('--downsample_rate', default=3.0, type=float, help='downsample rate')

args = parser.parse_args()

DOWNLOAD_DIR = args.download_dir 


# Download videos
if not os.path.isdir(DOWNLOAD_DIR):
    print("The indicated download directory does not exist!")
    print("Directory made!")
    os.makedirs(DOWNLOAD_DIR)

'''Download videos'''
ydl_opt = {'outtmpl': DOWNLOAD_DIR + '%(id)s.%(ext)s',
           'format': 'mp4',
           'ignoreerros': True,
           'skipdownload': True}
ydl = yt_dlp.YoutubeDL(ydl_opt)

url_list = []
with open(args.youtube_idlist,'r') as fid:
    for line in fid.readlines():
        vid = line.rstrip()
        url = "https://www.youtube.com/watch?v=" + vid
        print("url:"+url)
        url_list.append(url)

try:
    ydl.download(url_list)
except:
    print('errors ocurred!')
print("Download finished!")

all_videos = sorted(glob.glob(DOWNLOAD_DIR + '*.mp4'))
print("Number of videos: ", len(all_videos))