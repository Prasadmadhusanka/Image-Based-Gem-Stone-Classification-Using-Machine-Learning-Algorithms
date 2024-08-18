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

Data Source: [Kaggle - Gemstones Images](https://www.kaggle.com/datasets/lsind18/gemstones-image)  

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

![outliers vis](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/result-outliers.png)

By identifying and separating outliers from inliers, the dataset was further refined to ensure that only high-quality, representative images were used for model training. This step helps in building more robust and accurate machine learning models by minimizing the impact of anomalies in the data.

**Normalization**

**Method – Continuize Widget**

Normalization was performed using the **Continuize widget** in Orange, a tool designed to preprocess and scale numerical data. Normalization is an essential step in data preprocessing, particularly when preparing data for machine learning models. It transforms data into a standardized range, making it easier to compare and process.

**Input – Inliers Dataset**

The input to the normalization process was the dataset consisting of inliers—the data points identified as fitting well within the expected distribution and not flagged as outliers. These inliers had already been transformed into numerical feature vectors through image embedding.

**Output – Transformed Normalized Dataset**

The output of the normalization process was a transformed dataset with values scaled to a range of 0 to 1. This normalization ensures that all features contribute equally to the analysis and that the data is on a comparable scale. Specifically:

Range 0-1: Each feature in the dataset is adjusted so that its values fall within the range of 0 to 1. This scaling helps in reducing the impact of differences in feature magnitudes, which can be crucial for algorithms sensitive to the scale of input data.

![Normalization](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/normalization.png)
 
The normalized dataset is now ready for use in machine learning models, as normalization improves the efficiency and accuracy of training algorithms by ensuring that each feature contributes proportionally and that the data is uniformly scaled. Flowing shows the data range before and after the normalization 

![normalization result](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/result-normalization.png)

#### 5.	Classifiers Construction

Supervised Machine Learning Algorithms

To classify the gemstone images, several supervised machine learning algorithms were employed. Each algorithm was selected for its ability to handle the features extracted from the images and provide accurate classification results. The following classifiers were used logistic regressing using **logistic regression widget**, SVM using **SVM widget**, random forest using **random forest widget**, and neural network using **neural network widget**.

![models](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/models.png)
 
**Logistic Regression**

Logistic Regression provides probabilistic predictions for each class based on a non-linear transformation of the input features. In the context of multi-class classification, the algorithm uses the softmax function to extend binary logistic regression to multiple classes. The softmax function transforms the raw model outputs (logits) into probabilities by exponentiating and normalizing them, ensuring that the sum of the probabilities for all classes equals one.

**Regularization Parameter (C)**
The parameter 𝐶 in Logistic Regression is the inverse of the regularization strength. Regularization is used to prevent overfitting by penalizing excessively complex models. A lower value of C increases the regularization strength, leading to a simpler model with reduced variance but potentially higher bias. Conversely, a higher value of  C decreases regularization, allowing the model to fit the training data more closely but with a higher risk of overfitting.

**Support Vector Machine (SVM)**

The Support Vector Machine is a powerful classification algorithm that works by finding the hyperplane that best separates the classes in the feature space. SVM is known for its effectiveness in high-dimensional spaces and its ability to handle complex relationships between features through kernel functions.

To address multi-class classification problems, a strategy known as "one-vs-rest" or "one-vs-all" is employed. In this approach, multiple binary SVM classifiers are constructed, each designed to distinguish one class from all other classes. This results in a series of binary classifiers that collectively handle the multi-class problem.

**Optimized Parameters**

Three key parameters were optimized to enhance the performance of the SVM

**Kernel Type**

The kernel function determines how the input features are transformed into a higher-dimensional space to enable linear separation of classes. Common kernel types include linear, polynomial, and radial basis function (RBF) kernels. The choice of kernel impacts the SVM’s ability to model complex relationships between features.

**Regularization Parameter (C)**

The parameter c  controls the trade-off between achieving a low error on the training data and minimizing the model's complexity. A lower value of increases regularization, which encourages a simpler model by penalizing large margin violations. A higher value of 𝐶n decreases regularization, allowing the model to fit the training data more closely but with a greater risk of overfitting.

**Kernel Coefficient (Gamma)**

The gamma parameter is specific to certain kernel functions, such as the RBF kernel. It controls the influence of a single training example. A high gamma value means that the influence of a training example is limited to a smaller region, leading to a more complex decision boundary, while a low gamma value means that the influence is broader, leading to a smoother decision boundary.

By optimizing these parameters, the SVM classifier was tuned to achieve the best performance in classifying the gemstone images, balancing between fitting the training data accurately and maintaining generalization to new, unseen data.

![SVM](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/SVM.png)

**Random Forest**

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the model of the classes (classification) of the individual trees. It is well-regarded for its robustness and accuracy, as it reduces the risk of over fitting by averaging the results from multiple trees. This approach reduces the correlation between the trees, ensuring that each tree in the forest is different and that the overall model benefits from a diverse set of predictions. At each node within a tree, the data is recursively partitioned based on the feature that results in the most homogeneous collection of samples. This feature is selected from a randomly chosen subset of all available features, further reducing the correlation between trees and enhancing the model’s robustness.

**Optimized Parameters**

Three key parameters were optimized to improve the performance of the Random Forest model

**Number of Estimators (Trees)**

This parameter determines the number of trees in the forest. A higher number of trees generally improves the model’s accuracy and stability, as it reduces variance. However, it also increases computational cost. The optimal number of trees strikes a balance between performance and efficiency.

**Maximum Depth of the Tree**

The maximum depth controls how deep each tree can grow. Deeper trees can capture more complex patterns in the data but are more prone to overfitting. By optimizing this parameter, the model can effectively capture important patterns without becoming too complex.

**Minimum Number of Samples Required in Each Leaf Node**

This parameter sets the minimum number of samples that must be present in a leaf node. Increasing this number can prevent the model from creating overly specific splits, which can lead to overfitting. Optimizing this parameter helps in controlling the tree’s complexity and ensuring that the model generalizes well to new data.

By carefully tuning these parameters, the Random Forest model was optimized to achieve a strong balance between accuracy, robustness, and generalization, making it well-suited for the classification of gemstone images.

![Random forest](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/random%20forest.png)

**Artificial Neural Network – Multi-Layer Perceptron (MLP)**

The activation of neurons in the hidden layers of the MLP neural network was implemented using the ReLU (Rectified Linear Unit) function. ReLU is a popular activation function that introduces non-linearity into the model, allowing the network to learn complex patterns in the data by outputting the input directly if it is positive, and zero otherwise. This helps in mitigating the vanishing gradient problem and speeds up the training process.

The MLP neural network was trained using the "Back propagation" algorithm, a widely used method for optimizing neural networks. During training, the network calculates the error by comparing the difference between the predicted output and the desired output. This error is typically measured using the mean squared error (MSE), which is calculated as the square of the difference between the predicted and actual values. The back propagation algorithm then propagates this error backward through the network, adjusting the weights of the connections in order to minimize the error over time, leading to improved model accuracy.

![ANN](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/nural%20net%20work.png)

#### 6. Evaluation of Classifiers 

**Model Performance Evaluation on Training and Test Data**

After constructing the machine learning models, their performance was evaluated using the Test & Score widget in Orange. This evaluation was performed on both the training and test datasets to assess how well the models learned from the data and how they performed on unseen data.

The following metrics were measured:

**1. AUC (Area Under the Curve):** AUC measures the ability of the model to distinguish between classes. It represents the area under the ROC curve, with higher values indicating better performance in distinguishing between classes.

**2. Accuracy:** Accuracy represents the proportion of correctly classified instances out of the total instances. It provides an overall measure of the model's correctness.
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

**3. Precision:** Precision measures the proportion of true positive predictions out of all positive predictions made by the model. It indicates the accuracy of positive predictions.
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**4. Sensitivity (Recall or True Positive Rate):** Sensitivity measures the proportion of actual positives that were correctly identified by the model. It reflects the model's ability to capture positive instances.
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**5. F1-Score:** The F1-Score is the harmonic mean of Precision and Sensitivity. It provides a balanced measure of the model's accuracy, particularly useful when there is an uneven class distribution.
$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**6. Specificity:** Specificity measures the proportion of actual negatives that were correctly identified by the model. It complements Sensitivity by indicating how well the model avoids false positives.
$$
\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}
$$

**7. Training and Testing Time:** The time taken by the model to train on the data and to make predictions on the test data was also recorded. This metric is important for understanding the computational efficiency of the model.

**Prediction on Unseen Gemstone Images**

To further validate the models, predictions were made on a separate set of 340 unseen gemstone images, grouped into 85 gem classes. For each gem class, 4 images were used to ensure a representative sample of the class. The predictions were carried out using the Predictions widget in Orange.

The same performance metrics, AUC, Accuracy, Precision, Sensitivity, F1-Score, and Specificity were measured on this unseen dataset to evaluate how well the models generalized to new, previously unseen data. This step is crucial for understanding the model's real-world applicability, as it tests the model's ability to accurately classify gemstones it has not encountered during training. 

By comparing the performance metrics on the training, test, and unseen datasets, a comprehensive assessment of the model's accuracy, robustness, and generalization ability was obtained.

![prediction](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/prediction.png)

#### 7.	Results and discussion 

**Using train and test data set**

![result train](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/overall_1.png)

When considering the overall results for the training and test datasets, Logistic Regression achieved the highest values across multiple performance metrics, including AUC, accuracy, precision, sensitivity, F1-score, and specificity. This indicates that Logistic Regression was the most effective model for classifying the gemstone images.

Following Logistic Regression, the Artificial Neural Network (ANN) with a Multi-Layer Perceptron (MLP) architecture ranked second, demonstrating strong performance across the same metrics. The Support Vector Machine (SVM) model came in third, also showing good results but slightly lower than those of Logistic Regression and ANN-MLP.

This ranking reflects the models' ability to accurately classify gemstones and their generalization to new, unseen data.

![time](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/time_1.png)

When considering the training and testing times, the experiments were conducted on a machine with an Intel(R) Core(TM) i5-2430M CPU @ 2.40GHz. The training and testing times, measured in seconds, provided insight into the computational efficiency of each model.

**Training Time:**

Random Forest showed the quickest training time. This efficiency is due to its parallel nature, where multiple decision trees are trained simultaneously. The simplicity of each individual tree and the use of bootstrapped samples contribute to its fast training speed. Support Vector Machine (SVM) came next in terms of training time.  Artificial Neural Network (ANN-MLP)  took longer. Neural networks typically require more time to train due to the iterative nature of back propagation and the optimization of weights across multiple layers. Logistic Regression had the longest training time among the models. This is likely due to the optimization process involved in finding the best-fitting model, especially when regularization is applied, which can increase computational complexity.

**Testing Time:**
Random Forest  also demonstrated the fastest testing time. Once trained, predicting with Random Forest is quick because it simply involves aggregating the outputs of the decision trees. Logistic Regression  had the second-fastest testing time. Artificial Neural Network (ANN-MLP)  ranked third in testing time. Support Vector Machine (SVM)  had the longest testing time. 

This analysis shows that while Random Forest is the most efficient in both training and testing, Logistic Regression, despite being slower to train, offers relatively fast prediction times. ANN and SVM, while powerful, require more time for both training and testing, reflecting their complexity.
 
**Using prediction data set**

![overall2](https://github.com/Prasadmadhusanka/Image-Based-Gem-Stone-Classification-Using-Machine-Learning-Algorithms/blob/main/images/overall_2.png)

When considering the performance metrics AUC, accuracy, precision, F1-score, sensitivity, and specificity on the prediction dataset, Logistic Regression secured the first place. It consistently achieved the highest scores across all these metrics, demonstrating its superior ability to generalize to new, unseen data. Following Logistic Regression, the Artificial Neural Network (ANN-MLP) ranked second, showing strong performance but slightly below that of Logistic Regression. The Support Vector Machine (SVM) came in third, performing well but not as strongly as Logistic Regression and ANN. Lastly, Random Forest ranked fourth in terms of these metrics, indicating that while it was efficient in terms of training and testing time, it did not perform as well as the other models in terms of prediction accuracy and other performance measures.

#### 8.	Conclusion

The **Random Forest** algorithm demonstrated the fastest training and testing times compared to other machine learning algorithms. However, it exhibited low performance across key metrics such as accuracy, precision, sensitivity, F1-score, and specificity, all of which were below 50%. As a result, Random Forest is not suitable for image-based gemstone classification.

On the other hand, **Logistic Regression**, **SVM**, and **ANN-MLP** algorithms exceeded 60% in all performance measures, making them more effective for this task. While Logistic Regression showed strong performance, it required twice as much time for training and testing compared to SVM and ANN-MLP.

Among the top three models, the **ANN-MLP** algorithm performed particularly well in classifying the Emerald gem category, while Logistic Regression excelled in classifying the Zircon gem category. This trend was consistent across other gem classes as well.

Considering both performance and speed, a combination of **SVM** and **ANN-MLP** provided the best overall results. Therefore, these algorithms were selected and implemented in the prototype (a portable electronic system) as the classification algorithms for image-based gemstone classification.





































