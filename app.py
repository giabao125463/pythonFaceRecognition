import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import face_recognition
import detect
from detect import h
from flask_cors import CORS

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def checkface(cmnd, avatar):
    try:
        cmnd_encoding = face_recognition.face_encodings(cmnd)[0]
        avatar_encoding = face_recognition.face_encodings(avatar)[0]
    except IndexError:
        return "<p>I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...</p>"

    cmnd_faces = [
        cmnd_encoding
    ]

    result = face_recognition.compare_faces(cmnd_faces, avatar_encoding, 0.6)

    return result[0]
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'cmnd' not in request.files:
            return "<h1>khong thay file cmnd</h1><p>day la get</p>"

        file = request.files['cmnd']
        file2 = request.files['cmndBack']
        file3 = request.files['avatar']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return "<h1>khong thay file</h1><p>day la get</p>"
            
        if file and allowed_file(file.filename):
            front = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], front))

            back = secure_filename(file2.filename)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], back))

            avatar = secure_filename(file3.filename)
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], avatar))
        
        cmnd, hovaten, ngaysinh, nguyenquan = detect.CMNDFront(app.config['UPLOAD_FOLDER'] + "/"+ front)
        ngaycap = detect.CMNDBack(app.config['UPLOAD_FOLDER'] + "/"+ back)

        cmndImg = face_recognition.load_image_file(app.config['UPLOAD_FOLDER'] + "/"+ front)
        avatarImg = face_recognition.load_image_file(app.config['UPLOAD_FOLDER'] + "/"+ avatar)
        result = checkface(cmndImg, avatarImg)
        if (result):
            result = 'Nhận dạng đúng người.'
        else:
            result = 'Không trùng khớp khuôn mặt và CMND.'

        return "<p>CMND: " + cmnd + "</p><p>Ho va ten: " + hovaten + "</p><p>ngay sinh: " + ngaysinh + "</p><p>nguyen quan: " + nguyenquan + "</p><p>ngay cap: " + ngaycap + "</p><p>Xác thực: " + result + "</p>"
    return "<h1>Distant Reading Archive</h1><p>day la get</p>"

app.run('0.0.0.0')