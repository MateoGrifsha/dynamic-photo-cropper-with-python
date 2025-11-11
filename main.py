import os
import random

import cv2
from datetime import datetime
from hsemotion.facial_emotions import HSEmotionRecognizer
from exif import Image
from PIL import Image as PImage

#gets all the image filepaths
def getAllPaths(initialPath):
    filePaths = []
    for root, _, files in os.walk(initialPath):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return filePaths


if __name__ == "__main__":
    # model that recognizes faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # emotion recognizer
    model_name = 'enet_b0_8_best_afew'
    fer = HSEmotionRecognizer(model_name=model_name, device='cpu')  # device is cpu or gpu
    count = 0
    faceCount = 0
    #enumerate bcs we need the index item, iterates through all the items in ./images
    for idx, item in enumerate(getAllPaths("./images")):
        # read the image
        img = cv2.imread(os.path.abspath(item))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # unique naming convention
        d = datetime.now()
        # name = f"{d.second}{d.microsecond}{idx}{count}"

        # iterates over detected faces and draw the approximate head region
        for (x, y, w, h) in faces:
            # approx where the face would be on average
            extension_factor_top = 0.7  # Extend 70% above the face top
            extension_factor_bottom = 0.1  # Extend 10% below the face bottom

            y1 = int(y - h * extension_factor_top)
            h1 = int(h + h * (extension_factor_top + extension_factor_bottom))
            x1 = int(x - w * 0.1)  # Extend sides slightly
            w1 = int(w + w * 0.2)

            # makes sure the coordinates are within img boundaries
            y1 = max(0, y1)
            x1 = max(0, x1)

            #original image is set to the cropped size of the head
            face_img = img[y1:y1 + h1, x1:x1 + w1]

            emotion, score = fer.predict_emotions(face_img, logits=True)
            if(emotion == "Anger"):
                cv2.imwrite(f'./cropped/anger/{count}{faceCount}.jpeg', face_img)
            elif(emotion == "Disgust"):
                cv2.imwrite(f'./cropped/disgust/{count}{faceCount}.jpeg', face_img)
            elif (emotion == "Fear"):
                cv2.imwrite(f'./cropped/fear/{count}{faceCount}.jpeg', face_img)
            elif (emotion == "Happiness"):
                cv2.imwrite(f'./cropped/happy/{count}{faceCount}.jpeg', face_img)
            elif (emotion == "Sad"):
                cv2.imwrite(f'./cropped/sad/{count}{faceCount}.jpeg', face_img)
            elif (emotion == "Surprise"):
                cv2.imwrite(f'./cropped/surprise/{count}{faceCount}.jpeg', face_img)
            elif(emotion == "Neutral"):
                cv2.imwrite(f'./cropped/neutral/{count}{faceCount}.jpeg', face_img)
            faceCount += 1
            #close all processes
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        count += 1

#-----------SAVES METADATA IN ONE OF THE DEFAULT FIELDS ------------- uses exif library
            # with open(f"./cropped/{name}.jpeg", "rb") as file:
            #     print(file)
            #     exif_img = Image(file)
            # 
            # exif_img.model = emotion
            # 
            # with open(f"./cropped/{name}.jpeg", "wb") as file:
            #     file.write(exif_img.get_file())


#------------- SAVING METADATA IN CUSTOM FIELD (UserComment). Doesn't show on Right Click -> Properties--------------
            # finalImg = PImage.open(f"./cropped/{name}.jpeg")
            #
            # # Get existing EXIF data or create a new one
            # exif_dict = piexif.load(finalImg.info["exif"]) if "exif" in finalImg.info else {"0th": {}, "Exif": {}, "GPS": {},
            #                                                                       "Interop": {}, "1st": {},
            #                                                                       "thumbnail": None}
            #
            # # Add or modify custom metadata (e.g., UserComment tag)
            # # The UserComment tag is often used for custom data, and can store JSON strings
            # exif_dict["Exif"][piexif.ExifIFD.UserComment] = f"Emotion: {emotion}".encode("utf-8")
            #
            # # Convert the dictionary back to EXIF data
            # exif_bytes = piexif.dump(exif_dict)
            #
            # # Save the image with the updated EXIF data
            # finalImg.save(f"./cropped/{name}.jpeg", exif=exif_bytes)
            #
            # loaded_exif = piexif.load(f"./cropped/{name}.jpeg")
            # user_comment = loaded_exif["Exif"].get(piexif.ExifIFD.UserComment, b"").decode("utf-8", errors="ignore")
            # print("Stored EXIF Comment:", user_comment)