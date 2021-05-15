from fastapi import FastAPI, File, Form, UploadFile
import datetime, cv2
import numpy as np
import tensorflow as tf
import pickle
from skimage.feature import hog

def normallize(img):
    norm = np.zeros((800,800))
    final = cv2.normalize(img,  norm, 0, 255, cv2.NORM_MINMAX)
    return final

def clahe(img_in):
    lab = cv2.cvtColor(img_in,cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    lab = cv2.merge((L2,A,B))
    im_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return im_clahe

def resize(img_in):
    w_real, h_real, _ = np.array(img_in).shape
    # Contour -> หาเส้นรอบวง
    im_gr = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)  # grayscale
    _, im_thr = cv2.threshold(im_gr, 10, 255, 0)
    # หาเส้นเค้าโครง
    contour, _ = cv2.findContours(im_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # max contour คือเอาเฉพาะเส้นโค้งที่ใหญ่สุด
    max_contour = np.argmax([len(i) for i in contour])
    # con_t = cv2.drawContours(im_rgb,contour,max_contour, (0,200,0), 5)
    con = img_in
    x, y, w, h = cv2.boundingRect(contour[max_contour])
    # print(w,h)
    if w >= h:
        w_cat = w
        h_plus = w - h
        con = con[y:y + w, x:x + w]
        if w > w_real:
            img_plus = np.zeros([int(h_plus / 2), w_cat, 3], dtype=np.uint8)
            img_plus.fill(0)
            # print(con.shape, img_plus.shape)
            con = np.concatenate((img_plus[:None], con, img_plus[:None]), axis=0)  # 0 short edge, 1 long edge
    else:
        w_cat = h - w
        h_plus = h
        con = con[y:y + h, x:x + h]
    # Resize
    im_resized = cv2.resize(con, (224, 224))
    return im_resized

def model_dl(img_in):
    class_names = ['glaucoma', 'normal', 'other']
    predictions = model.predict(np.array([img_in]))[0]
    argmax = np.argmax(predictions)
    return class_names[argmax], predictions[argmax]

model_path = "model_dl/model_4.h5"
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])


app = FastAPI()

@app.get("/")
async def root():
    return {"Greeting": "Hello, World!"}

@app.post("/api/fundus")
async def uploadImage(nonce: str = Form(None, title="Random String"), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, 1)
    img_resize = resize(img)
    img_nor = normallize(img_resize)
    img_clahe = clahe(img_nor)

    class_out, class_cfd = model_dl(img_clahe)

    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_cfd),
        "currentTime": datetime.datetime.now(),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
            "re_size": dict(zip(["height", "width", "channels"], img_resize.shape))
        }
    }
