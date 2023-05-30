# -- coding: utf8 -
import cv2
import sqlite3
# Describe: 训练识别器



# 导入openCV自带的人脸检测配置文件
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(1)

# 创建LBP人脸识别器
rec = cv2.face.LBPHFaceRecognizer_create()

# 加载训练结果至人脸识别器
rec.read('recognizer\\trainningData.yml')
id = 0

def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM people WHERE id = " + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


# 显示在边框旁边的文字属性
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,2)
#cv2.putText（img，text，（x，y+h），cv2.FONT_HERSHEY_SIMPLEX， 1，（255，255，255））
#font = cv2.putText(cv2.FONT_HERSHEY_SIMPLEX,2,1,0,2)
font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_scale = 2
font_thickness = 1
line_type = cv2.LINE_AA

# 创建字体对象
font = cv2.putText(
    img=None,
    text='',
    org=(0, 0),
    fontFace=font_face,
    fontScale=font_scale,
    color=(0, 0, 0),
    thickness=font_thickness,
    lineType=line_type
)
while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        profile = getProfile(id)
        '''if (profile != None):
            # 显示检测到的人脸对应的人名
            cv2.cv.PutText(cv2.cv.fromarray(img), str(round(conf,2)), (x, y + h), font, 255)
            cv2.cv.PutText(cv2.cv.fromarray(img), 'name:' + str(profile[1]), (x, y + h + 30), font, 255)
            cv2.cv.PutText(cv2.cv.fromarray(img), 'age:' + str(profile[2]), (x, y + h + 60), font, 255)
        else:
            cv2.cv.PutText(cv2.cv.fromarray(img), str('unkonw'), (x, y + h + 60), font, 255)'''


        font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        font_thickness = 1
        line_type = cv2.LINE_AA
        color = (255, 255, 255)

        if profile is not None:
            conf_text = f"Confidence: {round(conf, 2)}"
            name_text = f"Name: {profile[1]}"
            age_text = f"Age: {profile[2]}"

            cv2.putText(img, conf_text, (x, y + h), font_face, font_scale, color, font_thickness, line_type)
            cv2.putText(img, name_text, (x, y + h + 30), font_face, font_scale, color, font_thickness, line_type)
            cv2.putText(img, age_text, (x, y + h + 60), font_face, font_scale, color, font_thickness, line_type)
        else:
            cv2.putText(img, 'unknown', (x, y + h + 60), font_face, font_scale, color, font_thickness, line_type)

    cv2.imshow("Face",img);
    # 按Q键退出识别程序
    if(cv2.waitKey(2) == ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
