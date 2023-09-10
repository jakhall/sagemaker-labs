# Sagemaker ML Pipelines BlazingText Lab


## Introduction

Sagemaker Pipelines is a CI/CD solution for ML projects. It allows management of various workflows using custom or pre-trained ML models. 

Watch the following video to get started creating a new project, training a simple model and deploying it on an endpoint. It will walkthrough setting up the abalone sample project that will be used for the BlazingText model.

https://youtu.be/Hvz2GGU3Z8g?si=JcNnMp5ORxGzG0Ge

You can read more about BlazingText here, along with other models provided by Sagemaker:

https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html 

## Lab Instructions

<br>

1. If it hasn't been done already, create a local clone of the ModelBuild repository from the Projects > Repositories menu. You can then access the directory from the "File Browser" menu on the far left side.

 <br>  
   
![Screenshot_1png](https://github.com/jakhall/sagemaker-labs/assets/23346546/c3291e10-227d-42e6-9ea7-7c69b62d5447)

<br>

2. Copy and paste the contents of pipelines folder within this repo into the pipelines folder of the ModelBuild directory.  

<br>

![Screenshot_2](https://github.com/jakhall/sagemaker-labs/assets/23346546/7987c279-1abf-4675-8e37-e0752d1144b3)

<br>

3. Search for "s3" service within AWS and look for an existing bucket, it should have "Sagemaker" within the name. Within this bucket place the contents of the s3 folder within this repo. Don't worry if there are other files within the bucket, just make sure the data folder is at the root location. If there are multiple buckets with "Sagemaker" in the name, place the data folder in all of them. 

<br>
   
![Screenshot_4](https://github.com/jakhall/sagemaker-labs/assets/23346546/99fe7cce-4e2c-4082-beeb-92a37f4f72ba)

<br>

4. We now want to commit out local changes to the ModelBuild repository. We have to go to the Git menu on the left hand side. From here we have to stage the changes, this should be "pipeline.py", "preprocessing.py" and "sentiment-analysis.ipynb". Then press commit and lastly the publish button on the top right of the left-hand side menu. This will trigger the build and execution of the associated pipeline. 

<br>

![Screenshot_3 5](https://github.com/jakhall/sagemaker-labs/assets/23346546/ea290a88-3a13-4cb2-a540-b55830967146)

<br>

5. It will take roughly 10 minutes for the pipeline to build and execute. We can monitor the progress from "CodePipeline" service within AWS, this is helpful for troubleshooting issues in the build.

<br>

![Screenshot_4 5](https://github.com/jakhall/sagemaker-labs/assets/23346546/0975db52-cb5a-412f-b3d5-2b90816bd75d)

<br>

6. If the build and execution is successful we should be able to see the following from the project > piplelines tab. The video should demonstrate how to access this view. 

<br>

![Screenshot_5](https://github.com/jakhall/sagemaker-labs/assets/23346546/73938e49-9af5-4ef0-952b-395ca4229b49)

<br>

7. Our model has now been trained, and is waiting for review to deploy to an endpoint. To trigger this go to the Projects > Model Groups tab and select the model. Different versions of our model will appear here - there should be two. The first is the original abolone model, and the second is our BlazingText model. We need to approve the most recent version, this will trigger the deployment process. This should take several minutes to complete. In the sameway as before we can monitor the endpoint deployment from "CodePipeline" and "CloudFormation" within AWS. If deployment was successful "staging" should appear next to the version we approved.

<br>

![Screenshot_5 5](https://github.com/jakhall/sagemaker-labs/assets/23346546/106d753e-55fc-4118-aaea-b434687f5daf)

<br>

8. We should also be able to check on our endpoint from the main sagemaker page under Inference > Endpoints. Alternatively from within Sagemaker studio, under Deployments > Endpoints. They should show the same information. (Take note of the endpoint name, as we will use it for invocation later). 

<br>

![Screenshot_6](https://github.com/jakhall/sagemaker-labs/assets/23346546/7a482dd5-73ba-4ec0-8e28-a7dbaa81fa96)


![Screenshot_6 5](https://github.com/jakhall/sagemaker-labs/assets/23346546/73ba8537-f601-4c24-85db-71ec41f1f4ec)

<br>

9. If we select our endpoint from the Sagemaker studio menu, it will display additional information and allow us to test inference by sending in some example queries. Here's a couple examples to try:

{"instances": ["The worst guitar strings, very bad, don't buy."]}

{"instances": ["This is a great keyboard, keys are high quality!"]}

(Note: The model was trained on reviews ranging from 1 - 5 stars e.g. An output of _label_1 is a 1 star review and an output of _label_5 is a 5-star review).

<br>

![Screenshot_7](https://github.com/jakhall/sagemaker-labs/assets/23346546/861b28c0-14ce-4fab-9555-8da842498ee1)

<br>

10. The last thing to do is apply the endpoint to analyse script data. This is where we can use the "sentiment-analysis.ipynb" script we added to the pipeline directory earlier. For it to work, we need to input the name of s3 bucket we placed our data into (the csv and fdx file) and the name of the deployed endpoint. We should then be able to run each block of code sequentially, to run the model against lines of our script loaded from s3. 

<br>

![Screenshot_8](https://github.com/jakhall/sagemaker-labs/assets/23346546/cd4b276f-a003-48b5-a57c-aa48956e595f)

<br>

11. Well done if you made it this far, we've completed the lab. Hopefully, you have an understanding of how we can create/modify these pipelines in order to train and deploy models using AWS Sagemaker. You may notice the outputs are far from accurate using this model, the original data was reviews of musical instruments and don't apply well to the scripts. If you want, try to modify the preprocessing.py file to train the model on better data, we can also add additional steps to the pipeline for evaluation before deploying. When you're finished feel free to delete the endpoint, or set the status of all model versions to "rejected".

##Outro 

If you have questions or issues with any of the above, feel free to contact me: jak.hall@3gi.co.uk.
     

