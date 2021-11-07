import face_recognition

bills_image = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_image_encoding = face_recognition.face_encodings(bills_image)[0]

unknown_image = face_recognition.load_image_file(
    './img/unknown/d-trump.jpg')
unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [bill_image_encoding], unknown_image_encoding)

if results[0]:
    print('This is Bill Gates')
else:
    print('This isnt Bill Gates')