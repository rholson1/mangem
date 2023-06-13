# MANGEM
MANGEM (Multimodal Analysis of Neuronal Gene expression, Electrophysiology, and Morphology) is available for use
at https://ctc.waisman.wisc.edu/mangem. 

MANGEM provides a step-by-step accessible and user-friendly interface 
to machine-learning alignment methods of neuronal multi-modal data while enabling real-time visualization of 
characteristics of raw and aligned cells. It can be run asynchronously for large-scale data alignment, 
provides users with various downstream analyses of aligned cells and visualizes the analytic results such as 
identifying multi-modal cell clusters of cells and detecting correlated genes with electrophysiological and 
morphological features.

## Deployment on Amazon AWS with Elastic Beanstalk
We deployed MANGEM on AWS using Elastic Beanstalk.  Documentation on deploying Python applications using AWS Elastic 
Beanstalk is available in the AWS Elastic Beanstalk Developer Guide: 
https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-apps.html

The steps to deploy MANGEM to AWS using Elastic Beanstalk are roughly:
1. Create environment in EBS.
https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/using-features.managing.html
2. Create an application bundle.
If you have cloned this repository, the application bundle to be deployed can be created from your local
working directory using git:
```
git archive -v -o application_bundle.zip --format=zip HEAD
```
3. Deploy application.
The application can be deployed by uploading the application bundle created in the previous step 
(`application_bundle.zip`) in the AWS Elastic Beanstalk console.  Deployment will automatically install and configure
MANGEM and its dependencies, including gunicorn, Redis, and Celery.

