import face_recognition

image = face_recognition.load_image_file('./img/groups/team2.jpg')
face_location = face_recognition.face_locations(image)

print('There are {len(face_location)} faces in the image')