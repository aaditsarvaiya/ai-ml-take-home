# AI Usage Log

AI Usage Log

1. 
What I asked: I asked Muse AI how to do the initial environment set up (including venv, ultralytics install, and Roboflow download.
What I kept vs. rewrote, and why: From this I kept all of the code that I needed to put in my terminal, as it was just the basic environment set up that I needed to do. I rewrote one of the lines, because I realized that the line was not working for some reason, and there was some syntax error. 
What the AI got wrong that I had to catch: For the initial environment set up on my terminal, there was nothing except some syntax error that the AI got wrong. So I only caught the syntax mistake as it displayed that as an error. 
How I verified it (for example, checked the metric by hand, ran it on data it hadn't seen): I ensured that the model actually ran and displayed on the next line .venv, which indicated that it had been set up.

2. 
What I asked: I asked Muse AI for help with the code of the following files: train.py, infer.py (partially) and extract_frames.py.
What I kept vs. rewrote, and why: From this I kept the code exactly the same for extract_frames.py because I did not have anything to change from that, and it was perfectly accurate to extract all of the frames needed. I had to rewrite a bit of the infer.py file because there were some issues with the arrays, as the append function did not work appropriately some of the times. I ended up modifying the train.py file for my custom augmented trainV2 file later on. So I ended up using the same skeleton from train.py but a different set of parameters to train the model on. 
What the AI got wrong that I had to catch: The AI mainly only got the array part wrong, as it was not working with the append function correctly with the array where the boxes would be placed. 
How I verified it (for example, checked the metric by hand, ran it on data it hadn't seen): I verified it, because I checked the terminal’s log initially when it did not run and gave an error message. However once I made the change, since it was running I presumed that the error had gone, and then when I went into the infer file I was able to visually see the final data and thus verify it worked. 

3. 
What I asked: I asked about the confusion matrix, and how it is to be read. I also asked what the metric definitions meant for each of the runs. 
What I kept vs. rewrote, and why: From this, it gave me a basic understanding of what the confusion matrix, and metric definitions displayed. So based on that, I completely wrote the justifications for the metric analysis by myself. It did not actually give me any descriptions or analysis points, as I had just asked to understand certain concepts. I did this, as I felt that the justifications and actual design choices must be made by me, and needed a critical human mind. 
What the AI got wrong that I had to catch: For this the AI had gone one small part of the confusion matrix where it initially gave me an understanding of how it shows false predictions but it made a small matrix, as it assumed that the confusion matrix for the custom augmentation training run. So I had to manually check and catch the AI for that aspect. 
How I verified it (for example, checked the metric by hand, ran it on data it hadn't seen): I verified this while closely looking at the numbers, and checking the confusion matrix points by hand for both of the runs (base and custom augmentation ones). I hand checked and ensured that the custom augmentation matrix had more false positives, and thus verified that the right matrix was being recognised now. 

4. 
What I asked: I asked about the methodology steps in the methodology document (step 1), and what all was done basically. 
What I kept vs. rewrote, and why: I kept the overall pieces of code that the AI had written as it included basic set up and mandatory information to be included for each user. However I rewrote the descriptions it had given for most of the steps, as I felt that it would be better to include actual human written information rather than information that I myself cannot directly restate. 
What the AI got wrong that I had to catch: The AI did not get anything wrong that I had to catch, mainly the only thing I “caught” is its language, being way too technical or overly jargon-filled and complex. 
How I verified it (for example, checked the metric by hand, ran it on data it hadn't seen): I verified this by trying to re-read and mentally follow the same instruction written there. I was able to verify that it was overly confusing/complex if I was unable to register where to go from there next. Next, I prompted the AI to simplify the language and then I rewrote it myself - and checked if I was able to mentally follow now to finally verify it. 



