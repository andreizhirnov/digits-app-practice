# An application used to recognize a handwritten digit

This repository includes:
- The estimates of an NN model in the ONNX format. This model was trained using PyTorch on the MNIST dataset.
- A Streamlit app that can process handwritten digits to guess which digit has been drawn and will commit the results to a Postgres database
- A Dockefile and a docker-compose file that can be used to create a Docker image of the app and run it together with a database

A live version of the app can be accessed here: [http://116.202.108.139:80/](http://116.202.108.139:80/)
 
