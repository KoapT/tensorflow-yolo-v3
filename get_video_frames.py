import cv2
import os
from tqdm import tqdm
'''
Watch the videos via opencv, and save every <interval> frame.
'''
source_path = '/home/tk/Desktop/20190919record'
video_list = []
for root, dirs, files in os.walk(source_path):
    for i in files:
        if i.endswith('.dav'):
            video_list.append(os.path.join(root,i))

for videopath in tqdm(video_list):
    video_size = os.path.getsize(videopath)/(1000*1000)
    if video_size > 600:
        interval = 400
    else:
        interval = 200
    outputpath = os.path.join('data',videopath.split('/')[-1][:-4])
    try:
        os.mkdir(outputpath)
    except:
        pass
    cap = cv2.VideoCapture(videopath)
    time = 0
    while True:
        ret, frame = cap.read()
        if ret:
            time += 1
            if "PTZ" not in videopath:
                frame = frame[::-1,:,:]
                h,w = frame.shape[:2]
                sections = []
                for row in range(2):
                    for col in range(4):
                        sections.append(frame[row*(h//2):row*(h//2)+h//2,col*(w//4):col*(w//4)+w//4,:])

                for i in range(8):
                    cv2.namedWindow('frame_{}'.format(i), 0)
                    cv2.resizeWindow('frame_{}'.format(i), 502, 450);
                    cv2.imshow('frame_{}'.format(i), sections[i])
            #     if time%25==0:
            #         cv2.imwrite('data/videoframe/{}_{:d}.jpg'.format(time,i), sections[i])
            else:
                cv2.imshow('frame', frame)

            if time%interval ==0:
                cv2.imwrite(os.path.join(outputpath,'{:d}.jpg'.format(time//interval)), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print('Captures of video {} has been saved in {}'.format(videopath.split('/')[-1],outputpath))

cap.release()
cv2.destroyAllWindows()