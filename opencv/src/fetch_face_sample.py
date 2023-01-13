# -----fetch_face_sample.py-----
# -----获取人脸-----

import cv2

# 调用笔记本内置摄像头，参数为0，如果有其他的摄像头可以调整参数为1,2
cap = cv2.VideoCapture(0)
# 调用人脸分类器，要根据实际路径调整
face_detector = cv2.CascadeClassifier(r'../dataset/haarcascades/haarcascade_frontalface_default.xml')
# 为即将录入的脸标记一个id
face_id = input('\nUser dataset input, Look at the camera and wait ...\n>>')
# sampleNum用来计数样本数目
count = 0

while True:
    # 从摄像头读取图片
    success, img = cap.read()
    # 转为灰度图片，减少计算量，提高识别度
    # 图像灰度化的目的：彩色图像中的每个像素的颜色由R，G，B三个分量决定，而每个分量中可取值0-255，
    # 这样一个像素点可以有1600多万（256*256*256=1677256）的颜色的变化范围。而灰度图像是R，G，B三个分量相同的一种特殊的彩色图像，
    # 其中一个像素点的变化范围为256种，所以在数字图像处理中一般将各种格式的图像转化为灰度图像以使后续的图像的计算量少一些。
    # 灰度图像的描述与彩色图像一样仍然反映了整副图像的整体和局部的色度和高亮等级的分布和特征。
    if success is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        break
    # 检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
    # 其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # 框选人脸，for循环保证一个能检测的实时动态视频流
    for (x, y, w, h) in faces:
        # xy为左上角的坐标,w为宽，h为高，用rectangle为人脸标记画框
        cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
        # 成功框选则样本数增加
        count += 1
        # 保存图像，把灰度图片看成二维数组来检测人脸区域
        # (这里预先在项目下建立了dataset/raw的文件夹，当然也可以设置为其他路径或者调用数据库)
        cv2.imwrite("../dataset/raw/user." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])
        # 显示图片
        cv2.imshow('image', img)
        # 保持画面的连续。waitkey方法可以绑定按键保证画面的收放，通过q键退出摄像
    k = cv2.waitKey(1)
    if k == '27':
        break
        # 或者得到800个样本后退出摄像，这里可以根据实际情况修改数据量，实际测试后800张的效果是比较理想的
    elif count >= 800:
        break

# 关闭摄像头，释放资源
cap.release()
cv2.destroyAllWindows()
