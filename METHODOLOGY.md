# Methodology

1. How to run it
First, we can set up the environment 
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python
Dataset: Sampled-YD-Object-Detection-2 (999 labeled images, Roboflow). This can be downloaded with the Roboflow API, with the API key. Next put it in an .env file. .env is git-ignored and never committed.

python train.py      # baseline: YOLOv8n, 40 epochs, imgsz=512, default augmentation
python train_v2.py   # custom augmentation (degrees=20, hsv_h=0.025, hsv_s=0.5,
                     # hsv_v=0.65, translate=0.15, erasing=0.25), 40 epochs
Shipped weights: runs/detect/runs/baseline_yolov8n/weights/best.pt (as the baseline won)
python infer.py --input <image_folder> --output <out_folder> [--conf 0.5]
For every input image this saves an annotated copy plus one .txt file with one line per detection:
class_id x_center y_center width height confidence (all normalized).
Class IDs follow data.yaml: 0 = bottle, 1 = mallet. They also include the confidence scores of the AI in brackets showing how confident the model is with its prediction (omitting >25% confidence level predictions). 




2. Thought process

Preprocessing & Augmentation 
I noticed that the bottles in the images appear to be represented in numerous angles, at different orientations. Some appear tilted horizontally, and some are tilted at acute angles. Therefore I used random rotation presenting the bottle at all possible angles, to ensure overfitting does not take place with the data and it actually learns what a bottle is.
I felt that the bottle images had noisy and grainy images. Paired with a cluttered and non-plain background, this made it harder to differentiate the bottle. Appearing to be washed out, with dark lighting and harsh shadows. For these cases, I utilized the random saturation and brightness jitters to ensure all possible visual compositions of the bottle were recognised. 
In the mallet images, there was only one specific mallet used that was in an orange colour. I used the random hue shift augmentation technique here to represent the mallet in different colours to make sure that the model does not memorize a mallet for its distinctive colour only. 
Since often times, the bottle and the mallet both were not in the centre of the frame, I utilized the random translate (shift) augmentation to move the x and y position of the objects and ensure the model does not think that the centre is the key focus point while determining the object. 
I recognised that the mallet was in some cases covered or placed near the background clutter, thus I decided to use the random erasing augmentation technique allowing random parts of the image to be cut out completely, further forcing the model to internalize all parts of these objects. 
Metric Analysis
This project included two models that I utilized for training, with different augmentation parameters. For the baseline version I used the base YOLOv8 values, which produced the following results. The bottle’s precision value was approximately 96% which indicated that in a given image, the model was able to detect and classify the bottle extremely well. While looking at the recall metric, the bottle’s metric was 81%. This indicated that the model was unable to correctly determine the exact position of all real bottles around 19% of the time. Moreover, looking at the mAP50-95 metric, the bottle received a low score of 57% meaning that although the bottle was detected, the accuracy and precision of the box placement for the detected bottle was flawed. Pairing this with the prediction images from the base YOLOv8 model, it can be seen that in the images there were extra boxes around the bottle, boxes in place there was a simple background and boxes that were too big and not precise enough. In the same way, although the precision and recall of the mallet was in total higher in comparison to the bottle, the mAP50-95 metric was 65% similar to the bottle - the model had difficulty placing the box on the exact precise area of the mallet. While connecting this to the images, in the event of increased random background clutter it was more likely to falsely predict a bottle rather than a mallet. Showing that it was wrongly more confident to predict a bottle in my viewpoint. 


In the second run, I trained the model on custom augmentation parameters of the following: 
degrees = 20 
hsv_h = 0.025
hsv_s = 0.5 
hsv_v = 0.65 
translate = 0.15 
erasing = 0.25 
Based on this, the metrics saw a significant deterioration across numerous factors like precision, recall and the mA50-95 metrics. The metrics for the bottle and mallet for the precision score declined by roughly 9% and 4% for bottle and mallet respectively. In terms of the precision score, the bottle’s recall went down by 4% further but the recall for mallet increased by roughly 2%. This could indicate that the custom augmentation allowed for the model to become used to an increased orientations and representations of mallets, allowing it to recognise multiple at once in the right positions. Nevertheless the custom augmentation caused a decline in the mAP50-95 metrics steeply indicating that the box placements and positional accuracy faced a decrease. 


The metric analysis for both of these versions displays that the custom augmentation with 40 epochs does not improve the outcome of the training for the model, and instead it deteriorates the models predictions. This could be due to the increase in the variation in aspects like translation, saturation and brightness jitters, rotation, and random erasing. I feel that although these factors prevent overfitting, it is possible that the 40 epochs that the model is trained on - is in fact not enough for the model to actually internalize the classification features for mallets and bottles. Therefore, the base version should be submitted from my perspective.  


Error Log

In the image bottle_double_recall.jpg there is a zoomed in large image of a mallet, however the system wrongly predicts it to be a bottle. The mallet is presumed to be two bottles. The box as well is extremely large and inaccurate along with the overall prediction class. I feel this happened because the handle part of the mallet on a larger scale looks like the main body of the bottle. 
The image double_bottle_overlap.jpg displays one bottle being covered by two boxes, overlapping one another. Here although the prediction is correct, the mAP50-95 score is compromised as it presumes there are two bottles together, when in reality there is only one. Neither boxes completely cover the bottle. It could have also happened due to the dark lighting in the background, and rocky background producing many harsh shadows.
The image bottle_false_pred.jpg showcases the model correctly locating the mallet with high confidence, and precise boxes. However the model predicts a whole human as a bottle with decent confidence (0.57). In my perspective this could have only happened due to the upright posture of the man and how his stance resembles a straight bottle in some way. I think that overall, the model’s over confidence with bottle predictions led to the model predicting bottles a lot more. 
In the image titled mallet_overlap.jpg the model accurately predicts that the object is a mallet however there are an underminable number of boxes created as it predicts at least 5-6 mallets to be in that one overlapping spot. This could have occurred due to the mallet being partially covered by grass, and simultaneously blending with the background causing the model to confuse the location and size of the actual mallet. 


Phone video Test 

I filmed a bottle on a chair, and moved around in different directions to display different orientations of the bottle. The lighting was natural and had some natural shadows falling onto the object. I did this in a 18-19 seconds video on my phone.
In these frames I realized that the model is unable to properly detect objects. Most of the images the model did not detect anything, and there was no observation of a bottle or even a mallet. The model was able to recognize that it is a bottle in some photos, but with medium to high confidence (not too high of a confidence level). The boxes are often too big, and overcompensate for the bottle as well.
The images frame_0002copy.jpeg and frame_0003copy.jpeg are examples that the model works and detects the bottle, however here the box is way too large and does not precisely detect the exact bottle location. The pros of the test were that it did not wrongly interpret the bottle to be a mallet a single time. But in frame_0014copy.jpeg the photo shows how the model completely missed the bottle, and felt there was no object in the frame to detect as well.
3. Known limitations
First of all, the 999 images for the training of the model acted as a limitation as the dataset was relatively small, and therefore the model may have not been able to generalize past the training images. This is possibly what caused the model to overfit, and thus show great images for training images, but fail during the phone test. The model was only able to perfect the types of bottles displayed to it with the 999 images. Next I would like to train the model based on a larger variety of bottles and mallets, and basically increase the dataset. 
For the custom augmentation version, it is possible that the model was undertrained with only 40 epochs, given that the custom augmentation had harsher preprocessing techniques that in the long run could have trained the model better. Next I would probably increase the number of epochs that the model trains on, as I feel that it would allow the model to adjust to a larger variety of objects, and also a variety of orientations of them. 
Another key limitation that I identified was that the model utilized was only YOLOv8, the later updated and bigger variants could have localized better and have had a greater success rate. Next I would probably try the same training and testing process on more recent models. 
Furthermore, I feel that I would probably add a variety of tests, at least 2-3 unique ones instead of just doing the phone test. As I just got access to one room and one type of object. Maybe if there were a couple more testing environments - the process could have in all become more realistic and representative of the truth of the model. 

