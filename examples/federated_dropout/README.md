# HeteroFL
## Setup
To run this example, you need to download this directory to your system. You can do so by running the following command in the shell or git bash:
``` shell
git clone --depth=1 https://github.com/QVQZZZ/HeFlwr.git \
&& mv HeFlwr/examples/federated_dropout . \
&& rm -rf HeFlwr \
&& cd heterofl
```
If you have multiple devices installed with HeFlwr and wish to run federated learning across multiple devices, you need to run the above command on the other devices as well.

To run this example, you need to modify the `server_address` in the `client{N}.py` file and the client IP in `strategy.py`.

## Running
Choose one device you like and run `python server.py` to make it act as a federated learning server.

Then, on any device, run `python client{N}.py` to make it a federated learning client.
You need to replace {N} with {1|2|3|4}, where the number controls the neural network's device training retention rate at p={0.25/0.5/0.75/1.0}, with different numbers representing different running loads.

## Results
You can get the training process loss and acc data on the device terminal running the server.

A `federated_dropout_test_log.txt` file will be generated in the running directory of each device, which records the device load information during the training process.