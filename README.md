*LSTM is a type of recurrent neural network (RNN) that is designed to handle sequential data and solve the problem of long-term dependencies in deep learning.

*It is widely used for tasks like speech recognition, text generation, time series forecasting, and chatbots.

Why LSTM?
 *Regular RNNs struggle with remembering information from long sequences due to the vanishing gradient problem.
 *LSTM solves this by using a memory cell and three special gates:

      -Forget Gate – Decides what information to discard.
      -Input Gate – Decides what new information to store.
      -Output Gate – Decides what to output.

LSTM Architecture
  *Each LSTM cell consists of:

      -Cell state (Ct) → Stores long-term memory.
      -Hidden state (Ht) → Short-term memory used for output.
      -Gates (Forget, Input, Output) → Control data flow.

Note: TensorFlow Version is 1.9
                (OR)
     pip install tensorflow==2.12.0
 
    
