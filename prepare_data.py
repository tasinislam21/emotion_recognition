import shutil
import pandas as pd

id = 0
df = pd.DataFrame(columns=['image','emotion'])
for i in range(19):
    for emotion in ["Anger.jpg", "Contempt.jpg", "Disgust.jpg", "Fear.jpg", "Happy.jpg", "Neutral.jpg", "Sad.jpg", "Surprised.jpg"]:
        src = "dataset_raw/images/{}/{}".format(i, emotion)
        dst = "dataset/images/{}.jpg".format(id)
        shutil.copy(src, dst)
        new_row = {'image': "{}.jpg".format(id), 'emotion': emotion[:-4]}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        id +=1

df.to_csv("dataset/label.csv", index=False)