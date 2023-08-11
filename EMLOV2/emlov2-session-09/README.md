# Deployment on Serverless

# TOC
- [TOC](#toc)
- [Assignment](#Assignment)
- [AWS Lambda](#aws-lambda)
- [Results on Postman](#results-on-postman)
- [Front End Deployment](#front-end-deployment)

# Assignment

- Deploy CIFAR10 Model on AWS Lambda and share link to your Lambda
- Deploy Frontend on Vercel and share link to your deployment

# AWS Lambda
response to events and automatically manages the compute resources required by that code. It is a service offered by Amazon Web Services (AWS) that runs your code in response to various events, such as changes to data in an Amazon S3 bucket or an Amazon DynamoDB table.

With Lambda, you can run your code for virtually any type of application or backend service, all with zero administration. You can set up your code to automatically trigger from other AWS services or call it directly from any web or mobile application.

Lambda functions are executed in response to specific events, such as a change to data in an S3 bucket, a request to an API Gateway endpoint, or a message published to an Amazon Simple Notification Service (SNS) topic. When an event occurs, Lambda executes the function, processes the event, and returns the results.

Lambda automatically scales your applications in response to incoming request traffic. You can use Lambda to build a variety of applications, including serverless backends for web, mobile, and IoT applications, data processing and analytics pipelines, and automation workflows.
![](images/p4.png)


- We have uploaded the image to ECR and create a lambda function that invokes the image.
![](images/p5.jpg)

- Cloud Formation Resources
![](images/p6.png)

# Results on Postman

```bash
https://w3oru842si.execute-api.ap-south-1.amazonaws.com/dev/inference
```
![](images/p2.jpg)

- Postman Results

![](images/p3.jpg)

- CloudWatch Logs
![](images/p7.png)

# Front End Deployment

![](images/p4.jpg)

The link to open -
```bash
http://65.1.129.80:3000/
```
