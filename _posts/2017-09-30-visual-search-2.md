---
layout: post
title:  "Visual Search using Deep Learning (pt. 1) - Model Architecture selection & Data preparation"
date:   2017-09-30 12:12:12 +0530
---

Now that we have decided out task (**Visual Similarity**) and our application (**Apparel Recommendation**) we will begin our project. 

### First step

Before beginning any task, it is very important to take time and do a thorough **literature review**. I do feel like a hypocrite saying this as I often find myself making this mistake of not conducting a thorough survey and jumping to implementing a solution only to find myself starting again a week later when I find a paper presenting a better solution.

For visual search, I shortlisted a method described in a paper titled ["Learning Fine-grained Image Similarity with Deep Ranking"](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf))


<object data="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf">
        This browser does not support PDFs. Please download the PDF to view it: <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf">Download PDF</a>.</p>
    </embed>
</object>

<br>
### Quick Summary:
The paper describes a three-way Siamese network which accepts three images (henceforth called a training triplet) in a go viz.

1. Input image
2. Positive image (An image visually similar to the input image)
3. Negative image (An image visually dissimilar to the input image than the positive image)

The individual network in the 3-way siamese network is a ```VGG-16 + a shallow Convolutional Neural Network```. The model individually passes each image of the aforementioned 3 through this network (same weights) and generates an embedding for each ie. output of the last dense of both networks. 

Loss for a single pass of a single  training triplet in the network is calculated as follows : 

- Let ```p```,```p+``` and ```p-``` be individual images of the training triplet.
- Let ```f``` be the embedding function that will convert an image to its embedding vector.
- Distance ```D``` between any 2 embedding vectors is calculated by taking squared differences between them.
<br>
      ``` Loss = max{0, g + D(f(p),f(p+)) - D(f(p),f(p-))} ```
<br>
where ```g``` is the gap parameter to regularie the distance of the 2 image pairs. 


Thus, in simple words, the model tries to generate an embedding for an image such that the distance between the image embedding and positive image embedding is less than the distance between the image embedding and negative image embedding.

### Training set

>Wait wait wait ... So is this Supervised, you ask? 

>Do I need to make a million training triplets to use this? 

**Yes and no !**

Let me explain.....

Yes you need labelled data for this model to train and you're gonna need a lot of it........but there is a smart way of generating these triplets. You could use traditional image similarity algorithms to generate these triplets. 
In my case, I made use of image meta-data that was available. Let me show you my dataset.<br><br>
```python
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import random
import os
```


```python
path = "../datasets/whole/"
```

Printing first 5 rows of image meta-data


```python
df = pd.read_csv(path+"csv/sample_set.csv",sep='\t')
df.head()
```
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_category</th>
      <th>_color</th>
      <th>_id</th>
      <th>_gender</th>
      <th>_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dress-material-menu</td>
      <td>Green</td>
      <td>1915297</td>
      <td>f</td>
      <td>dress-material-menu/1915297_Green_0.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dress-material-menu</td>
      <td>Green</td>
      <td>1915297</td>
      <td>f</td>
      <td>dress-material-menu/1915297_Green_1.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dress-material-menu</td>
      <td>Green</td>
      <td>1915297</td>
      <td>f</td>
      <td>dress-material-menu/1915297_Green_2.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dress-material-menu</td>
      <td>Green</td>
      <td>1915297</td>
      <td>f</td>
      <td>dress-material-menu/1915297_Green_3.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dress-material-menu</td>
      <td>White</td>
      <td>1845835</td>
      <td>f</td>
      <td>dress-material-menu/1845835_White_0.jpg</td>
    </tr>
  </tbody>
</table>
</div>

<br>
Let me tell you what these columns are.

- ```_name``` is the name-path to the actual image file
- ```_color``` is the color of the apparel.
- ```_category``` is the category of the product in the image viz. shirt, trouser, etc.
- ```_gender``` is the gender that uses the product
- ```_id``` is the identification number of the product. As you can see from the above data, I was provided with multiple images of the same product.

Some statistics to get a feel of the metadata provided :

```python
"""
Counting categories
"""
categories =  list(df._category.unique())
count = []
for c in categories:
    count.append((df["_category"]==c).sum())
category_probability = count/np.sum(count)
print("Number of categories : "+str(len(category_probability)))
```

    Number of categories : 25

```python
"""
Counting colors
"""
colors = list(df._color.unique())
count = []
for c in colors:
    count.append((df["_color"]==c).sum())
color_probability = count/np.sum(count)
print("Number of colors : "+str(len(color_probability)))
```

    Number of colors : 47

<br><br>

I decided to sample images randomly according to their category frequency. 
Let me describe the the type of triplets I decided to come up with :

- For a image, similar image is another image of the same product while the dissimilar is any other random image from the same category as selected image. (in-class negative)

- For a image, similar image is another image of the same product while the dissimilar is any other random image from a category other than that of the selected image. (out-of-class negative)

- For a image, similar image is another image of the same category and having same color while the dissimilar is any other random image from the same category as selected image. (in-class negative)

- For a image, similar image is another image of the same category and having same color while the dissimilar is any other random image from a category other than that of the selected image. (out-of-class negative)

Down below is a highly inefficient triplet sampling function that generates the aforementioned 4 types of triplets :

```python
def triplet_data_generator(count,dest):
    """
    Generates triplet
    
    Parameters
        count - int
            number of triplets to be generated
        dest - str
            location to write triplets
    Returns  
        writes to a csv in destination
        every element in csv is index of product in sample_set.csv 
  
    """
  
    triplet = pd.DataFrame(columns=("q","p","n"))
    for i in range(count):
        
        """Selecting query sample"""
        sample_category = np.random.choice(categories,p=category_probability) # select category
        sample_color    = np.random.choice(colors    ,p=color_probability   ) # select color
        temp = df[(df._category == sample_category) & (df._color==sample_color)]
        try:
            q_row = temp.sample()
            q_img = q_row.index.values[0]
        except:
            continue
            
        """Selecting positive sample"""   
        # select in-class(30%) negative or out-of-class(70%) negative
        negative_type = True if random.random() < 0.3 else False 

        if(negative_type):
            # in-class negative
            temp = df[(df._category == sample_category) & (df._color!=sample_color)]
            try:
                n_row = temp.sample()
                n_img = n_row.index.values[0]
            except:
                continue
            pass
        else:
            #out-of-class negative
            temp = df[(df._category != sample_category) ]
            try:
                n_row = temp.sample()
                n_img = n_row.index.values[0]
            except:
                continue
            pass
        """Selecting negative sample"""
        positive_type = True if random.random() < 0.1 else False
        # select different product with same color or different image of same product for positive

        if(positive_type):
            # different product same color
            temp = df[(df._category == sample_category) & (df._color==sample_color)]
            try:
                p_row = temp.sample()
                p_img = p_row.index.values[0]
            except:
                continue
            pass
        else:
            # same product different image
            temp = df[(df._id == list(q_row["_id"])[0])]
            try:
                p_row = temp.sample()
                p_img = p_row.index.values[0]
            except:
                continue
            pass
        """Insert in dataframe"""
        triplet.loc[i] = [q_img,p_img,n_img]
        if(i%5000==0):
            print(str(i)+" completed")
            triplet.to_csv(dest,sep='\t',index=False)
            
    triplet.to_csv(dest,sep='\t',index=False)
```


```python
triplet_data_generator(800000,path+"csv/triplets.csv")
```


```python
temp = pd.read_csv(path+"csv/triplets.csv",sep='\t')
temp.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>q</th>
      <th>p</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>125610</td>
      <td>125612</td>
      <td>664796</td>
    </tr>
    <tr>
      <th>1</th>
      <td>409528</td>
      <td>409526</td>
      <td>336219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>95883</td>
      <td>95883</td>
      <td>48374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>658843</td>
      <td>658842</td>
      <td>270175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>520477</td>
      <td>520476</td>
      <td>127710</td>
    </tr>
  </tbody>
</table>
</div>

<br>
Each row in this csv contains a triplet. Every number is the index of a corresponding image in the previous csv.

Let us visualize a few of them now :


```python
path2i = path + "images/"
for index, row in temp.iterrows():
    
    
    i1,i2,i3 = row["q"],row["p"],row["n"]
    
    i1 = list(df.loc[[i1]]["_name"])[0]
    i2 = list(df.loc[[i2]]["_name"])[0]
    i3 = list(df.loc[[i3]]["_name"])[0]
   
    
    try:
        ii1 = Image.open(path2i+i1)
        ii2 = Image.open(path2i+i2)
        ii3 = Image.open(path2i+i3)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.set_xticks([]) 
        ax1.set_yticks([])
        ax1.imshow(ii1)
        
        
        ax2 = fig.add_subplot(1,3,2)
        ax2.set_xticks([]) 
        ax2.set_yticks([])
        ax2.imshow(ii2)
        
        
        ax3 = fig.add_subplot(1,3,3)
        ax3.set_xticks([]) 
        ax3.set_yticks([])
        ax3.imshow(ii3)
        plt.show()
    except:
        continue
  
    
    if(index == 100):
        break
    pass
```

I will be printing 100 triplets for you to see how easily I managed to generate high quality labelled data. 
If you reach the comments section down below, let me know if you know a better sampling strategy or a better implementation of my current sampling strategy.
We will see the implementation of the network in the next post.


![png](/resources/triplet_generator_files/triplet_generator_11_0.png)



![png](/resources/triplet_generator_files/triplet_generator_11_1.png)



![png](/resources/triplet_generator_files/triplet_generator_11_2.png)



![png](/resources/triplet_generator_files/triplet_generator_11_3.png)



![png](/resources/triplet_generator_files/triplet_generator_11_4.png)



![png](/resources/triplet_generator_files/triplet_generator_11_5.png)



![png](/resources/triplet_generator_files/triplet_generator_11_6.png)



![png](/resources/triplet_generator_files/triplet_generator_11_7.png)



![png](/resources/triplet_generator_files/triplet_generator_11_8.png)



![png](/resources/triplet_generator_files/triplet_generator_11_9.png)



![png](/resources/triplet_generator_files/triplet_generator_11_10.png)



![png](/resources/triplet_generator_files/triplet_generator_11_11.png)



![png](/resources/triplet_generator_files/triplet_generator_11_12.png)



![png](/resources/triplet_generator_files/triplet_generator_11_13.png)



![png](/resources/triplet_generator_files/triplet_generator_11_14.png)



![png](/resources/triplet_generator_files/triplet_generator_11_15.png)



![png](/resources/triplet_generator_files/triplet_generator_11_16.png)



![png](/resources/triplet_generator_files/triplet_generator_11_17.png)



![png](/resources/triplet_generator_files/triplet_generator_11_18.png)



![png](/resources/triplet_generator_files/triplet_generator_11_19.png)



![png](/resources/triplet_generator_files/triplet_generator_11_20.png)



![png](/resources/triplet_generator_files/triplet_generator_11_21.png)



![png](/resources/triplet_generator_files/triplet_generator_11_22.png)



![png](/resources/triplet_generator_files/triplet_generator_11_23.png)



![png](/resources/triplet_generator_files/triplet_generator_11_24.png)



![png](/resources/triplet_generator_files/triplet_generator_11_25.png)



![png](/resources/triplet_generator_files/triplet_generator_11_26.png)



![png](/resources/triplet_generator_files/triplet_generator_11_27.png)



![png](/resources/triplet_generator_files/triplet_generator_11_28.png)



![png](/resources/triplet_generator_files/triplet_generator_11_29.png)



![png](/resources/triplet_generator_files/triplet_generator_11_30.png)



![png](/resources/triplet_generator_files/triplet_generator_11_31.png)



![png](/resources/triplet_generator_files/triplet_generator_11_32.png)



![png](/resources/triplet_generator_files/triplet_generator_11_33.png)



![png](/resources/triplet_generator_files/triplet_generator_11_34.png)



![png](/resources/triplet_generator_files/triplet_generator_11_35.png)



![png](/resources/triplet_generator_files/triplet_generator_11_36.png)



![png](/resources/triplet_generator_files/triplet_generator_11_37.png)



![png](/resources/triplet_generator_files/triplet_generator_11_38.png)



![png](/resources/triplet_generator_files/triplet_generator_11_39.png)



![png](/resources/triplet_generator_files/triplet_generator_11_40.png)



![png](/resources/triplet_generator_files/triplet_generator_11_41.png)



![png](/resources/triplet_generator_files/triplet_generator_11_42.png)



![png](/resources/triplet_generator_files/triplet_generator_11_43.png)



![png](/resources/triplet_generator_files/triplet_generator_11_44.png)



![png](/resources/triplet_generator_files/triplet_generator_11_45.png)



![png](/resources/triplet_generator_files/triplet_generator_11_46.png)



![png](/resources/triplet_generator_files/triplet_generator_11_47.png)



![png](/resources/triplet_generator_files/triplet_generator_11_48.png)



![png](/resources/triplet_generator_files/triplet_generator_11_49.png)



![png](/resources/triplet_generator_files/triplet_generator_11_50.png)



![png](/resources/triplet_generator_files/triplet_generator_11_51.png)



![png](/resources/triplet_generator_files/triplet_generator_11_52.png)



![png](/resources/triplet_generator_files/triplet_generator_11_53.png)



![png](/resources/triplet_generator_files/triplet_generator_11_54.png)



![png](/resources/triplet_generator_files/triplet_generator_11_55.png)



![png](/resources/triplet_generator_files/triplet_generator_11_56.png)



![png](/resources/triplet_generator_files/triplet_generator_11_57.png)



![png](/resources/triplet_generator_files/triplet_generator_11_58.png)



![png](/resources/triplet_generator_files/triplet_generator_11_59.png)



![png](/resources/triplet_generator_files/triplet_generator_11_60.png)



![png](/resources/triplet_generator_files/triplet_generator_11_61.png)



![png](/resources/triplet_generator_files/triplet_generator_11_62.png)



![png](/resources/triplet_generator_files/triplet_generator_11_63.png)



![png](/resources/triplet_generator_files/triplet_generator_11_64.png)



![png](/resources/triplet_generator_files/triplet_generator_11_65.png)



![png](/resources/triplet_generator_files/triplet_generator_11_66.png)



![png](/resources/triplet_generator_files/triplet_generator_11_67.png)



![png](/resources/triplet_generator_files/triplet_generator_11_68.png)



![png](/resources/triplet_generator_files/triplet_generator_11_69.png)



![png](/resources/triplet_generator_files/triplet_generator_11_70.png)



![png](/resources/triplet_generator_files/triplet_generator_11_71.png)



![png](/resources/triplet_generator_files/triplet_generator_11_72.png)



![png](/resources/triplet_generator_files/triplet_generator_11_73.png)



![png](/resources/triplet_generator_files/triplet_generator_11_74.png)



![png](/resources/triplet_generator_files/triplet_generator_11_75.png)



![png](/resources/triplet_generator_files/triplet_generator_11_76.png)



![png](/resources/triplet_generator_files/triplet_generator_11_77.png)



![png](/resources/triplet_generator_files/triplet_generator_11_78.png)



![png](/resources/triplet_generator_files/triplet_generator_11_79.png)



![png](/resources/triplet_generator_files/triplet_generator_11_80.png)



![png](/resources/triplet_generator_files/triplet_generator_11_81.png)



![png](/resources/triplet_generator_files/triplet_generator_11_82.png)



![png](/resources/triplet_generator_files/triplet_generator_11_83.png)



![png](/resources/triplet_generator_files/triplet_generator_11_84.png)



![png](/resources/triplet_generator_files/triplet_generator_11_85.png)



![png](/resources/triplet_generator_files/triplet_generator_11_86.png)



![png](/resources/triplet_generator_files/triplet_generator_11_87.png)



![png](/resources/triplet_generator_files/triplet_generator_11_88.png)



![png](/resources/triplet_generator_files/triplet_generator_11_89.png)



![png](/resources/triplet_generator_files/triplet_generator_11_90.png)



![png](/resources/triplet_generator_files/triplet_generator_11_91.png)



![png](/resources/triplet_generator_files/triplet_generator_11_92.png)



![png](/resources/triplet_generator_files/triplet_generator_11_93.png)



![png](/resources/triplet_generator_files/triplet_generator_11_94.png)



![png](/resources/triplet_generator_files/triplet_generator_11_95.png)



![png](/resources/triplet_generator_files/triplet_generator_11_96.png)



![png](/resources/triplet_generator_files/triplet_generator_11_97.png)



![png](/resources/triplet_generator_files/triplet_generator_11_98.png)



![png](/resources/triplet_generator_files/triplet_generator_11_99.png)



![png](/resources/triplet_generator_files/triplet_generator_11_100.png)

