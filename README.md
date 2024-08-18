# Image Based Gem Stone Classification Using Machine Learning Algorithms

This repository shows the project related to image based gem stone classification using ML algorithms and it cintains all files and documents used for this project for your further reference.

## Introduction 

Accurate gemstone classification is crucial to the gem and jewelry trade as the identification is an important in first step for evaluation and appraisal of any gem. Currently, the identity of a gemstone is determined using a combination of visual observation and spectro-chemical analysis. Gemmologists detect visual characteristics, such as color, transparency, luster, fractures, cleavages, inclusions, pleochroism, to facilitate the separation of gemstones. 

Identification of all gem characteristics is still difficult and time consuming, and not all laboratories have access to sophisticated instruments. In recent years, computers and algorithms have evolved significantly, and image processing and computer vision tasks are common place in many areas. In geological sciences, computer vision algorithms have been developed for classifying mineral grains, gems, soil and rocks.

This project represent  image  based classification of **2476** training and testing images and **340** images for prediction divided into **85** categories of gem stones.

## Objectives

* Design, development, implementation and evaluation of different machine learning type algorithms, for Image based Gemstones classification for 87 Gem classes.
* Feature Extraction of Gemstone Image Color Statistics Utilizing Advanced Image Embedding Techniques for Enhanced Model Input Representation.
* Comprehensive Evaluation of Classification Algorithms through Confusion Matrix Analysis and Key Performance Metrics
* Class-Wise Performance Assessment using Confusion Matrix and Detailed Metrics
* Comparative Analysis of Classification Algorithms via Receiver Operating Characteristic (ROC) Curves and Area Under the Curve (AUC) Metrics
* Training and testing Time Evaluation for Performance Assessment.
* Identification and Selection of the Optimal Machine Learning Model for High-Precision Image-Based Gemstone Classification Using Comprehensive Performance and Efficiency Metrics.

## Materials and Methods

### Software Used 

Following shows the software tools and libraries used in project:

* **Development Environment**: **Visual Studio Code (v1.75.0)** configured with **Python (v3.10.8)** for scripting and integration tasks.
* **Image Preprocessing**: Utilized the [**Rembg library (v2.0.30)**](https://pypi.org/project/rembg/) for efficient background removal from gemstone images, enhancing the quality of input data for machine learning models.
* **Data Mining and Machine Learning**: Employed [**Orange Data Mining (v3.34.1)**](https://orangedatamining.com/getting-started/) as the primary platform for the classification of gemstone images, integrating multiple machine learning algorithms for model training and evaluation.
* [**SV2 Library (v1.5)**](https://pypi.org/project/sv2/): Integrated for additional preprocessing or feature extraction tasks, contributing to the overall workflow of image-based gemstone classification.

### Methodology 

Following shows the proposed methodology used or project
![Methodology](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/methodology.png)
 
Hence this project is done by Orange software, above proposed methodology is utilized. Following shows the visual workflow construction done by Orange according to the proposed methodology.

![Workflow](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/workflow.png)

#### 1.	Data Acquisition

Data Source: Kaggle - Gemstones Images [Link](https://www.kaggle.com/datasets/lsind18/gemstones-image)  

This dataset contains 3,577 images of different gemstones. The training and testing set includes 3,284 images grouped into 87 gem classes for prediction, and 363 images grouped into 87 gem classes for testing. All images are in various sizes, in .jpg format, and have different backgrounds.

Following shows the images of gemstones that collected by data sources. All gemstones are categorized according to colors.

![Gemstones](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/Gemstone%20images.png)
 
#### 2.	Image Preprocessing

#### Gem Class Classification

The original Kaggle dataset contained 87 gemstone categories. However, some categories were removed to improve the accuracy of the algorithms. Specifically, the "Garnet Red" category was excluded due to significant overlap with other categories, including Almandine, Pyrope, Rhodolite, and Spessartite. The "Moonstone" category was also eliminated because the images within this category displayed a wide range of colors, such as orange, white, and yellow, which introduced unwanted variability and could negatively impact the performance of the algorithms. after eliminating these two classes 85 classes are used for this project.

![Gem class removal](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/gemclass%20selection.png)

#### Background Removal

To prepare the images for analysis, background removal was performed to ensure that only the gemstones themselves were analyzed, reducing noise and improving model accuracy. This process was carried out using Visual Studio Code (v1.75.0) along with Python (v3.10.8).
The images were accessed from the local drive using the SV2 Library (v1.5). For the actual background removal, the Rembg Library (v2.0.30) was employed. This library is specifically designed for image segmentation and effectively isolates the subject. in this case, the gemstones from their backgrounds. By removing the backgrounds, the dataset was standardized, making the images more uniform and better suited for machine learning algorithms.

![Code](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/code.png) 

Following shows the gemstone images before and after image preprocessing

![Background removal](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/background%20removal.png)

#### Discarding Poorly Segmented Images

After performing background segmentation, a quality check was conducted to ensure that the images were properly processed. Poorly segmented images, where the gemstones were not accurately separated from the background, were identified and discarded. These included images where significant portions of the gemstone were missing or where remnants of the background were still visible. Additionally, images containing gem clusters—where multiple gemstones were present in a single image were also discarded. Such images could confuse the algorithm, as they introduce ambiguity in identifying and classifying individual gemstones.

![Poor segment removal](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/discard%20poorly%20segment%20images.png)

By eliminating these poorly segmented images, the dataset was refined to include only high-quality, well-segmented images, ensuring more reliable and accurate predictions by the machine learning model.

#### Image Dataset for Feature Extraction

After the image preprocessing steps, including gem class selection, background removal, and discarding poorly segmented images, the dataset was refined and organized for feature extraction. 

The final dataset for training and testing consists of 2,476 images, which are grouped into 85 distinct gem classes. This reduction from the original dataset is due to the removal of categories such as "Garnet Red" and "Moonstone," as well as the exclusion of poorly segmented images. The high-quality images that remain are now well-suited for training the machine learning models to accurately recognize and classify gemstones.

For prediction, a separate set of 340 images has been prepared, also grouped into the same 85 gem classes. This prediction set is used to evaluate the model's performance, ensuring that the trained algorithms can generalize well to new, unseen images. The consistency in the number of gem classes between the training/testing set and the prediction set ensures a balanced and representative evaluation of the model's accuracy.

#### 3.	Feature Extraction Using Image Embedding

Feature extraction was performed using the **Image Embedding widget** in Orange, a data mining and machine learning software suite. This process was crucial for transforming the raw images into a format that could be effectively used by machine learning models.

**Embedder – SqueezeNet**

The feature extraction was conducted through a deep neural network model known as SqueezeNet. SqueezeNet is a lightweight convolutional neural network (CNN) architecture designed for efficient image classification. Despite its compact size, it maintains a high level of accuracy, making it well-suited for tasks that require deep learning on large datasets with limited computational resources.

![Feature extraction](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/feature%20extraction.png)

**Input – List of Images**

The input to the embedding process was a list of preprocessed images from the dataset. These images, which had undergone background removal and quality checks, were fed into the SqueezeNet model

**Output – Images Represented as Vectors**

The output of this process was a set of feature vectors. Each image was represented by a vector of numbers, typically referred to as an embedding. These vectors are a compact, numerical representation of the images, capturing essential features such as shapes, textures, and colors that are crucial for distinguishing between different gemstone classes.

The vectorized form of the images enables the machine learning models to process and analyze the data more efficiently. By reducing the complex visual information into a set of meaningful numerical features, the embeddings make it possible to apply various machine learning algorithms to tasks such as classification, clustering, or prediction, with improved accuracy and speed.

![feature extraction](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/result-feature%20extraction.png)

#### 4.	Data Pre-processing 

**Outlier Detection**

To further refine the dataset, outlier detection was performed using the **Outlier widget** in Orange, a powerful tool for identifying data points that deviate significantly from the rest of the dataset. Detecting and handling outliers is a critical step in data preprocessing, as outliers can skew the results and reduce the performance of machine learning models.

**Method – Total Outlier Factor**

The method used for outlier detection was the Total Outlier Factor (TOF), which is based on the concept of local density. This approach calculates the density of each data point by analyzing its relationship with the k-nearest neighbors. Essentially, TOF assesses how isolated a particular data point is in relation to its neighbors. Points that have significantly lower local density compared to their neighbors are flagged as outliers. This method is particularly effective in identifying instances that do not conform to the expected distribution of the data.

![outlier](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/outliers.png)

**Input – Produced Dataset by Image Embedding**

The input for the outlier detection process was the dataset produced by the Image Embedding widget, where each image was represented as a vector of numerical features. This vectorized dataset was analyzed to detect anomalies that could negatively impact the training of the machine learning models.

**Output – Outliers and Inliers**

The output of this process was a classification of the dataset into outliers and inliers.

**Outliers:** These are instances that were scored as outliers by the TOF method. Outliers often represent data points that are significantly different from the majority of the dataset, possibly due to errors in data processing, unusual characteristics, or noise. In the context of this gemstone dataset, outliers might include images that were incorrectly segmented, mislabeled, or had unusual visual features not representative of their class.

**Inliers:** These are instances that were not scored as outliers, meaning they fit well within the expected distribution of the data. Inliers represent the "normal" data points that are consistent with the patterns seen in the majority of the dataset.

![outliersinliers](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/outlier%20data.png)




















