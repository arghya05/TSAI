# Submission for Week 11

- [Problem Statement](#problem-statement)
- [OpenCV Yolo](#part-1)
- [Results](#results)
- [Part 2](#part-2)
- [Problem Statement of Assignment B](#problem-statement-of-assignment-b)
- [COCO DATASET](#coco-dataset)
- [Refrences](#references)
- [Team Members](#team-members)



# OpenCV Yolo

**Problem Statement:**

- OpenCV YoloV3: [SOURCE](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
     1. Run this above code on your laptop or Colab. 
     2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
     3. Run this image through the code above. 
     4. Upload the link to GitHub implementation of this
     5. Upload the annotated image by YOLO. 
 
- Link to [Notebook](https://github.com/vivek-a81/EVA6/blob/main/Session11/S11P1.ipynb)


Results
------------
**Testing On Our Images:**
<p float="center">
 <img src="images/out1.png" alt="drawing" width="550">
</p>
<p float="center">
 <img src="images/out_neha.jpg" alt="drawing" width="550">
</p>

**Testing On More Images:**

<p float="center">
 <img src="images/out3.png" alt="drawing">
 <img src="images/out4.png" alt="drawing" width="950">
 <img src="images/out2.png" alt="drawing">
</p>


# Part 2
**Problem Statement:**

1. Training Custom Dataset on Colab for YoloV3
     1. Refer to this Colab File:  [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
     2. Refer to this GitHub [Repo](https://github.com/theschoolofai/YoloV3)
     3. Download this dataset (Links to an external site.). This was annotated by EVA5 Students. Collect and add 25 images for the following 4 classes into the dataset shared:
           1. class names are in custom.names file. 
           2. you must follow exact rules to make sure that you can train the model. Steps are explained in the README.md file on github repo link above.
           3. Once you add your additional 100 images, train the model         
2. Once done:
      1. [Download](https://www.y2mate.com/en57) a very small (~10-30sec) video from youtube which shows your classes. 
      2. Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to extract frames from the video. 
      3. Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
      4. Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
      python detect.py --conf-three 0.3 --output output_folder_name
      5. Use  ffmpeg  to convert the files in your output folder to video
      6. Upload the video to YouTube. 
      7. Also run the model on 16 images that you have collected (4 for each class)

# COCO DATASET
The sample_coco.txt given for this assignment looks like this: id: 0, height: 330, width: 1093, bbox:[69, 464, 312, 175], id: 1, height: 782, width: 439, bbox:[359, 292, 83, 199], id: 3, height: 645, width: 831, bbox:[297, 312, 267, 167], id: 34, height: 943, width: 608, bbox:[275, 112, 319, 290], id: 20, height: 593, width: 857, bbox:[71, 368, 146, 147], id: 61, height: 587, width: 745, bbox:[177, 463, 68, 302],

Here is is the class id, followed by the height and width of the image. the bounding box (bbox). The COCO dataset defines the bounding box as x,y, width, height, where x and y is the vertex closer to origin.

<p float="center">
  <img src="Assignment B/images/anchorboxes.png" alt="drawing" width="650">
</p>


<p float="center">
  <img src="Assignment B/images/excel_ss.png" alt="drawing" width="650">
</p>

References
------------------------

* https://github.com/theschoolofai/YoloV3
* https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/


Team Members
------------------------

Neha Mittal, Vivek Chaudhary
