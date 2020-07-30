# Project Write-Up


I Have used Faster_rcnn_inception_v2_coco_2018_01_28
http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

I converted the model using the Model optimizer
## Explaining Custom Layers

Since Custom layers are used based on the framework (Since I have used the Tensorflow model). 

I had to 
* Regsiter custom layers as extentions for Model Optimizer 
* Replace subraphs which were unsupported with different subgraphs
* Registering definite sub-graphs of the model as those that should be offloaded to TensorFlow during inference.


Some of the potential reasons for handling custom layers are 

* Without handling custom layers Model optimizer cant conver the model to IR
* They are used to handle unsuppoted layers
* The Model optimiser clasifies any unknown layers as Custim layers

## Comparing Model Performance

Comparing the two models i.e. ssd_inception_v2_coco and faster_rcnn_inception_v2_coco in terms of latency and memory, several insights were drawn. It could be clearly seen that the Latency (microseconds) and Memory (Mb) decreases in case of OpenVINO as compared to plain Tensorflow model which is very useful in case of OpenVINO applications.

| Model/Framework                             | Latency (microseconds)            | Memory (Mb) |
| -----------------------------------         |:---------------------------------:| -------:|
| ssd_inception_v2_coco (plain TF)            | 229                               | 538    |
| ssd_inception_v2_coco (OpenVINO)            | 150                               | 329    |
| faster_rcnn_inception_v2_coco (plain TF)    | 1279                              | 562    |
| faster_rcnn_inception_v2_coco (OpenVINO)    | 891                              | 281    |


The difference between model accuracy pre- and post-conversion was :
* The model accuracy before convertion is slightly higher than the model which is converted into IR but it totally overrules the faster speed given for detection with respect to the bulky models.
* The size of the model pre- and post-conversion was :
  * The size of the model after converting into IR was smaller than the model before convertion.
When a model is converted into IR It'll convert into .xml file which contains model architecture and .bin file which contains weights and baises.
* The average inference time of the model pre- and post-conversion was :
  * The average inference time of the model after converting to IR is shorter than the model before converting.


## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

* Shoping malls for billing counters
* SmartLights that go off when no human is present
* Object detection alarms

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

* Lighting of the room plays a key role but most of the object detection models depend on change in the gradients (Gradient decent) which can handle weaker lighting aswell
* Model acuracy is key as any human size object could be easily detected as a human hence ancuracy and the training size of the model play a key role in this
* Camera focal length gives good focus when we have a larger focal lenght
* In case of the model being deployed in a crowdy area there has to be a model that detects in that use case.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD Mobilenet
  - [Model Source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - I converted the model to an Intermediate Representation with the following arguments
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - The model was insufficient for the app because it wasn't pretty accurate while doing inference. Here's an image showing mis classification of the model:


  - I tried to improve the model for the app by using some transfer learning techniques, I tried to retrain few of the model layers with some additional data but that did not work too well for this use case.
  
- Model 2: SSD Inception V2]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```
  
  - The model was insufficient for the app because it had pretty high latency in making predictions ~155 ms whereas the model I now use just takes ~40 ms. It made accurate predictions but due to a very huge tradeoff in inference time, the model could not be used.
  - I tried to improve the model for the app by reducing the precision of weights, however this had a very huge impact on the accuracy.

- Model 3: SSD Coco MobileNet V1
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments

```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - The model was insufficient for the app because it had a very low inference accuracy. I particularly observed a trend that it was unable to identify people with their back facing towards the camera making this model unusable.

