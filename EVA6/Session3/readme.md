Assignment

1. Write a neural network that can:
   1. take 2 inputs:
      1. an image from MNIST dataset, and
      2. a random number between 0 and 9
         2.and gives two outputs:
      3. the "number" that was represented by the MNIST image, and
      4. the "sum" of this number with the random number that was generated and sent as the input to the network

![network](https://user-images.githubusercontent.com/84603388/119204883-4cc83180-bab4-11eb-8827-512642628da5.png)

```
3. you can mix fully connected layers and convolution layers
4. you can use one-hot encoding to represent the random number input as well as the "summed" output.
```

2. Your code MUST be:
   1. well documented (via readme file on github and comments in the code)
   2. must mention the data representation
   3. must mention your data generation strategy
   4. must mention how you have combined the two inputs
   5. must mention how you are evaluating your results
   6. must mention "what" results you finally got and how did you evaluate your results
   7. must mention what loss function you picked and why!
   8. training MUST happen on the GPU

Architecture:

![model_arch](https://user-images.githubusercontent.com/84603388/119204998-99137180-bab4-11eb-9242-b2cfe80613fa.png)

Trainig log file:

![Screenshot (225)](https://user-images.githubusercontent.com/84603388/119217911-1366e480-bafb-11eb-9816-637a9789ddfa.png)

Final output:
![output](https://user-images.githubusercontent.com/84603388/119204920-65d0e280-bab4-11eb-9efb-c1a305c68e75.png)
