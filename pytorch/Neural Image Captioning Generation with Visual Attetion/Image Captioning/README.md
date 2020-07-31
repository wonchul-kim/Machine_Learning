





This **image capationing** repository is based on the paper, [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf) and it is a process to convert image to sequential natural language descripting the image. The general architecture is composed of two parts: encoder and decoder. Normally, for image captioning, the encoder is CNNs and the decoder is RNNs.

#### Training

The encoder is CNN extracting feature map according to the input image. The decoder is LSTM network trained to generate target texts from the source texts and the feature map. For example of source texts and target texts, if the image description is "Giraffes standing next to each other", the source sequence is a list containing ('<start>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other') and the target sequence is a list containing ('Giraffes', 'standing', 'next', 'to', 'each', 'other', '<end>'). 
</end>
</start>
    
#### Test

The encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using `encoder.eval()`. For the decoder part, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a `for-loop`.

