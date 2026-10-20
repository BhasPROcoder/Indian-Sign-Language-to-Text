# Indian Sign Language A-Z prototype

This Python prototype recognises one alphabet sign at a time. It uses grayscale
HOG features and a linear SVM, as proposed in the presentation. The full
dataset ZIP has **10,400 images: 400 for each letter A-Z**. The script reads the
ZIP directly, without extracting it, and uses at most 400 images per letter
during training. The included model was trained on all 10,400 images.
The large dataset ZIP is kept locally and is not uploaded to GitHub; the
included model is enough to run the webcam prototype.

## Setup

1. Download this repository from GitHub (**Code → Download ZIP**) and extract it,
   or use the existing project folder on your computer.
2. If you want to retrain, put `Indian_Sign_Language_Small.zip` in the
   same folder. The included model already works without the dataset ZIP.
3. Open that folder in VS Code (or open a terminal in the folder) and install
   the packages:

```powershell
python -m pip install -r requirements.txt
```

## Live webcam

In VS Code, open `sign_to_text.py` and click **Run Python File**. Running the
script without arguments starts the webcam with the included model. You can
also use the integrated terminal:

```powershell
python sign_to_text.py
```

Place the complete hand sign anywhere in the camera frame, with both hands when
needed. The detector checks the whole frame and a smaller box follows the
detected hands for letter recognition. The program makes a prediction every
second frame and votes across three recent predictions. When it sees no hand it
shows **No hand**; for a weak letter match it shows **Uncertain** and no letter.
Press **Space** to add the displayed letter,
**Backspace** to delete a letter, **C** to clear the text, or **Q** to quit.
If camera 0 is unavailable, add `--camera 1`.

The hand detector is included in `models/hand_landmarker.task`. Keep that folder
beside `sign_to_text.py`. A letter is displayed only when a hand is found, the
best model score is at least -0.25, and it leads the next letter by at least
0.15. If a real sign repeatedly shows **Uncertain**, try `--min-score -0.4`.
If wrong letters appear, increase `--min-score` or `--min-margin` instead.
The training images have coloured landmark lines that do not appear
in the webcam, so live predictions can be less reliable than the dataset
evaluation. This is a classroom prototype.

## Retrain on the full ZIP (optional)

```powershell
python sign_to_text.py train
```

This creates a model and an evaluation report in `results`. The default cap is
400 images per letter. The training split is 70% training, 15% validation and
15% testing. Training uses only the dataset ZIP; webcam frames are never added.

To predict a saved image instead of opening the webcam:

```powershell
python sign_to_text.py predict --model results\isl_hog_svm.joblib --image "C:\path\to\sign.jpg"
```

## Simple improvements without enlarging the dataset

- Use an even, plain background and keep the whole hand visible, including near
  the screen edges. A hand partly outside the camera frame can still be missed.
- Increase `--votes` to 5 if the letter flickers. Change `--scan-every` to 1
  for more frequent predictions or 3 for lower CPU use.
- Use the `results/confusion_matrix.png` file to see which letters the model
  confuses. For a more honest evaluation, test on a different signer or session.
- Keep your face and other people out of the camera frame when possible so the
  letter classifier focuses on the sign.

The dataset source identified in the presentation is Preksha Palva's Indian
Sign Language dataset on Kaggle. Keep this attribution if you share the work.

