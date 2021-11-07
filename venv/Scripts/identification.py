import face_recognition
from PIL import Image, ImageDraw

bills_image = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_image_encoding = face_recognition.face_encodings(bills_image)[0]

steves_image = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_image_encoding = face_recognition.face_encodings(steves_image)[0]

elons_image = face_recognition.load_image_file('./img/known/Elon Musk.jpg')
elon_image_encoding = face_recognition.face_encodings(elons_image)[0]

known_face_encodings = [
  bill_image_encoding,
  steve_image_encoding,
  elon_image_encoding
]

known_face_names = [
  "Bill Gates",
  "Steve Jobs",
  "Elon Musk"
]

test_image = face_recognition.load_image_file('./img/groups/bill-steve-elon.jpg')

face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

pil_image = Image.fromarray(test_image)

draw = ImageDraw.Draw(pil_image)

for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown face"

  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]  
  
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# shows image
pil_image.show()