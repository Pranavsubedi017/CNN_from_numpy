import numpy as np
from numpy.lib.stride_tricks import as_strided

class Convolve():
  def __init__(self,activation=None,pooling=None,filter_size=(3,3),num_filters=1,stride=1,padding=0):

    self.filter_size=filter_size
    self.filter_height,self.filter_width=self.filter_size
    self.num_filters=num_filters
    self.stride=stride
    self.padding=padding
    self.pool=pooling
    self.activation=activation
    self.t = 0
   
  def get_patches(self,input_array,backward=False):
    if backward==True:
      self.padding2=self.padding
    else:
      self.padding2=0  

    self.input_array=input_array
    
    self.batch_size,self.height,self.width,self.channel=self.input_array.shape
    self.output_height = int((self.height +2*self.padding2 - self.filter_height)/self.stride) + 1
    self.output_width = int((self.width +2*self.padding2  - self.filter_width)/self.stride) + 1
    self.new_shape = (self.batch_size, self.output_height, self.output_width, self.filter_height, self.filter_width, self.channel)
    self.new_strides = (self.input_array.strides[0], self.stride * self.input_array.strides[1], self.stride * self.input_array.strides[2],
                  self.input_array.strides[1], self.input_array.strides[2], self.input_array.strides[3])
    self.patches = as_strided(self.input_array, self.new_shape, self.new_strides)
    #print(patches.shape)
    return self.patches

  def forward(self,input_array):
    self.input_array=input_array
    #print(f'input array ko shape{self.input_array.shape}')
    if self.padding > 0:
      self.input_array_padded = np.pad(self.input_array, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
    else:
      self.input_array_padded=self.input_array
    self.patches=self.get_patches(self.input_array_padded)
    
    #print(f'patches ko shape{self.patches.shape}')
    self.patches=self.patches.reshape(self.patches.shape[0],self.patches.shape[1],self.patches.shape[2],-1)
    #print(f'patches{self.patches[0][0][0]}')
    #print(self.patches.shape)
    self.filter = np.random.randn(self.filter_height,self.filter_width,self.channel,self.num_filters)
    self.filter=self.filter.reshape(-1,self.num_filters)
    #print(self.filter.shape)
    self.output_array=np.tensordot(self.patches,self.filter,axes=([3],[0]))
    self.patches=self.patches.reshape(self.batch_size,self.output_height,self.output_width,-1)
    #print(f'output array{self.output_array.shape}')
    self.output_array=self.output_array.reshape(self.batch_size,self.output_height,self.output_width,self.num_filters)
    #print(f'output array{self.output_array.shape}')
    if self.pool:
      self.output_array=self.pool.forward(self.output_array)
    if self.activation:
          self.output_array=self.activation.forward(self.output_array)
    #print(f'convolve forward{self.output_array.shape}')

    return self.output_array

  def l2(self):
        return np.sum(self.filter ** 2)

  def backward(self, gradient):
    self.gradient=gradient

    if self.activation:
      self.gradient=self.activation.backward(self.gradient)

    if self.pool:
      self.gradient=self.pool.backward(self.gradient)
    #print(f'convolve backward ma pako{self.gradient.shape}')
    # print(f'patches shape RESHAPED{self.patches.reshape(self.batch_size, self.output_height, self.output_width, self.filter_height, self.filter_width, self.channel).shape}')
    # self.patches=self.patches.reshape(self.batch_size, self.output_height, self.output_width, self.filter_height, self.filter_width, self.channel)
    # self.gradient=self.gradient.reshape(self.batch_size,self.output_height,self.output_width,self.channel,-1)
    #print(f'convolve backward{self.gradient.shape}')
   
    self.filter_grad = np.tensordot(self.patches.transpose(3,0, 1, 2), self.gradient, axes=([1,2,3], [0, 1, 2]))
    #print(f'filter grad{self.filter_grad.shape}')
    # Reshape the filter gradient to match the original filter dimensions
    self.filter_grad = self.filter_grad.reshape(self.filter_height, self.filter_width,self.channel, self.num_filters)
    #print(f'filter grad reshape{self.filter_grad.shape}')
    # Calculate gradient for the input
    #gradient_input = np.zeros_like(self.input_array)
    self.flipped_filter = np.flip(self.filter, axis=(0, 1))
    self.flipped_filter=self.flipped_filter.reshape(-1,self.channel)
    #print(f'flipped filter{self.flipped_filter.shape}')
    self.gradient_patches = self.get_patches(self.gradient,backward=True)
    #print(f'gradient patches{self.gradient_patches.shape}')
    self.gradient_patches=self.gradient_patches.reshape(self.gradient_patches.shape[0],self.gradient_patches.shape[1],self.gradient_patches.shape[2],-1)
    #print(f'gradient patches222222222{self.gradient_patches.shape}')
    self.gradient_input = np.tensordot(self.gradient_patches, self.flipped_filter, axes=([3], [0]))
    #print(f'gradient input{self.gradient_input.shape}')
    # return self.gradient_input
    # for i in range(self.output_height):
    #     for j in range(self.output_width):
    #         patch_gradient = np.tensordot(gradient[:, i, j, :], self.filter.T, axes=(1, 0))
    #         patch_gradient = patch_gradient.reshape(self.batch_size, self.filter_height, self.filter_width, self.channel)
    #         gradient_input[:, i*self.stride:i*self.stride+self.filter_height, j*self.stride:j*self.stride+self.filter_width, :] += patch_gradient

    # if self.padding > 0:
    #     print(f'gradient input padding{self.gradient_input.shape}')
    #     self.gradient_input = self.gradient_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
    #     print(f'gradient input padding{self.gradient_input.shape}')

    # else:
    #     self.gradient_input=self.gradient_input

    return self.gradient_input



  def calculate(self, optimizer):


    self.sdw = np.zeros_like(self.filter_grad)
    


    self.vdw =  np.zeros_like(self.filter_grad)



    if optimizer == 'adam':
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        sdw = beta2 * self.sdw + (1 - beta2) * (self.filter_grad ** 2)
        self.sdw = sdw

        vdw = beta1 * self.vdw + (1 - beta1) * self.filter_grad
        self.vdw = vdw

        # Bias correction for adam optimizer for the starting difference while using exponantially weighted average
        sdw_corrected = self.sdw / (1 - beta2 ** self.t)

        vdw_corrected = self.vdw / (1 - beta1 ** self.t)


        self.sdw_corrected = sdw_corrected

        self.vdw_corrected = vdw_corrected


  def update(self, learning_rate, optimizer):
    if optimizer == 'adam':
        self.filter_grad -= learning_rate * self.vdw_corrected / (np.sqrt(self.sdw_corrected) + 1e-8)

    else:
        self.filter -= learning_rate * self.filter_grad



class MaxPool():
  def __init__(self,pool_size=(2,2),stride=2):

    self.pool_size=pool_size
    self.pool_height,self.pool_width=self.pool_size
    self.stride=stride


  def get_patches(self,input_array):
    self.input_array=input_array
    self.batch_size,self.height,self.width,self.channel=self.input_array.shape
    self.output_height = int((self.height - self.pool_height)/self.stride) + 1
    self.output_width = int((self.width - self.pool_width)/self.stride) + 1
    self.new_shape = (self.batch_size, self.output_height, self.output_width, self.pool_height, self.pool_width, self.channel)
    self.new_strides = (self.input_array.strides[0], self.stride * self.input_array.strides[1], self.stride * self.input_array.strides[2],
    self.input_array.strides[1], self.input_array.strides[2], self.input_array.strides[3])
    self.patches = as_strided(self.input_array, self.new_shape, self.new_strides)

    return self.patches

  def forward(self,input_array):
    self.input_array=input_array

    self.patches2=self.get_patches(self.input_array)
    #print(f'patches2{self.patches2.shape}')
    self.patches2=self.patches2.reshape(self.patches.shape[0],self.patches.shape[1],self.patches.shape[2],self.pool_height*self.pool_width,-1)
    #print(f'patches2{self.patches2.shape}')
    # print(f'patches{self.patches[0][0][0]}')
    #print(f'patches ko shape{self.patches2.shape}')


    self.output=np.max(self.patches2,axis=3)
    #print(f'output of pool{self.output.shape}')
    # self.output=self.output.reshape(self.batch_size,self.output_height,self.output_width,self.channel)
    self.max_indices=np.argmax(self.patches2,axis=3)
    #print(f'maxpool forward{self.output.shape}')

    return self.output

  # def l2(self):
  #       return np.sum(self.output ** 2)

  def backward(self, gradient):
          maxpool_gradient=np.zeros_like(self.input_array)
          self.maxpool_gradient=maxpool_gradient
          gradient=gradient.flatten()
          max_indices=self.max_indices.reshape(-1)


          batch_indices,height_indices,width_indices,channel_indices= np.indices((self.batch_size,self.output_height,self.output_width,self.channel))
          indexes= ( batch_indices.flatten(),
                   (height_indices.flatten()*self.stride).reshape(-1) + max_indices //self.pool_width,
                   (width_indices.flatten()*self.stride).reshape(-1)+max_indices % self.pool_width,

                   channel_indices.flatten() )
          self.maxpool_gradient[indexes]+=gradient.flatten()


          self.maxpool_gradient=self.maxpool_gradient.reshape(self.batch_size,self.height,self.width,self.channel)
          #print(f'maxpool backward{self.maxpool_gradient.shape}')
          return self.maxpool_gradient

class Flatten():
  def __init__(self):
    pass

  def forward(self,input_array):
    self.input_array=input_array
    self.batch_size,self.height,self.width,self.channel=self.input_array.shape
    self.new_shape=(self.batch_size,self.height*self.width*self.channel)
    self.output_array=self.input_array.reshape(self.new_shape)
    #print(f'flatten forward{self.output_array.shape}')
    return self.output_array

  def l2(self):
        return 1

  def backward(self, gradient):
        self.gradient=gradient
        #print(f'flatten backward{self.gradient.reshape(self.input_array.shape).shape}')
        return self.gradient.reshape(self.input_array.shape)


  def calculate(self, optimizer):
    pass

  def update(self, learning_rate, optimizer):
   pass
'''
input_array=np.random.randn(2000,50,50,3)
print(input_array.shape)
filter_size=(3,3)
layer4=Convolve(input_array,filter_size,30,2,1)
output_array=layer4.forward()
print(output_array.shape)
layer5=Max_Pool(output_array)
output_array2=layer5.forward()
print(output_array2.shape)
'''
