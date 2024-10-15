<script lang="ts">
  // You can add any specific logic or data here for Subheading 1
</script>

<div class="container">
  <h1>Skin Cancer Detection</h1>

  <h2>Goal</h2>
  <p>
    The goal of the project is to enhance the detection of skin cancer by
    leveraging advanced Machine Learning (ML) and Deep Learning (DL) models. The
    primary focus is on classifying skin lesions as benign or malignant, thereby
    supporting dermatologists with AI-driven tools to increase diagnostic
    accuracy and aid in early detection of melanoma.
  </p>
  <br />
  <h2>Used Tools</h2>
  <ul>
    <li>
      Dataset: Melanoma Cancer Image Dataset (13,929 high-resolution images)
    </li>
    <li>
      Data Pre-processing: Python (pytorch's torchvision for augmentation), Hash
      value filtering for duplicates, RGB anomaly detection, image resizing, SVD
      for compression
    </li>
    <li>Model Training: TensorFlow, sklearn</li>
    <li>Computational Resource: NVidia A100 GPU</li>
  </ul>

  <h2>Used ML Algorithms</h2>
  <ul>
    <li>
      <strong>Random Forest (RF):</strong> An ensemble learning method that constructs
      multiple decision trees to improve classification accuracy. Used as a baseline
      model in the project with 100 estimators.
    </li>
    <li>
      <strong>Linear Neural Network (LNN):</strong> A simple deep learning model
      with a linear structure, mainly used for binary classification tasks. The model
      includes a flattening layer and a dense output layer optimized using the Adam
      optimizer.
    </li>
    <li>
      <strong>Convolutional Neural Networks (CNNs):</strong> Two CNN
      architectures were used:
      <ul>
        <li>
          CNN 1: A simple CNN with one convolutional layer, max pooling, and a
          dense layer.
        </li>
        <li>
          CNN 2: A more complex CNN with additional convolutional and pooling
          layers, along with dropout layers to prevent overfitting.
        </li>
      </ul>
    </li>
  </ul>
</div>

<style>
  .container {
    padding-bottom: 100px;
  }
</style>
