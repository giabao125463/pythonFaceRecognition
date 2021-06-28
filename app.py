import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import face_recognition

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def checkface(cmnd, avatar):
    try:
        cmnd_encoding = face_recognition.face_encodings(cmnd)[0]
        avatar_encoding = face_recognition.face_encodings(avatar)[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()

    cmnd_faces = [
        cmnd_encoding
    ]

    result = face_recognition.compare_faces(cmnd_faces, avatar_encoding, 0.4)

    return result[0]
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        file2 = request.files['file2']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filename2 = secure_filename(file2.filename)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        cmnd = face_recognition.load_image_file(app.config['UPLOAD_FOLDER'] + "/"+ filename)
        avatar = face_recognition.load_image_file(app.config['UPLOAD_FOLDER'] + "/"+ filename2)
        result = checkface(cmnd, avatar)

        return "<h1>"+ result +"</h1><p>day la get</p>"
    return "<h1>Distant Reading Archive</h1><p>day la get</p>"

app.run()