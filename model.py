import torch.nn as nn
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
import torchvision.transforms as transforms

class CircleNet(nn.Module):    # nn.Module is parent class  
    def __init__(self):
        super(CircleNet, self).__init__()  #calls init of parent class
                

        #----------------------------------------------
        # implementation needed here 
        #----------------------------------------------
      
        
        # The convolution layers were chosen to keep dimensions of input image: (I-F+2P)/S +1= (128-3+2)/1 + 1 = 128
        
        # Our images are RGB, so input channels = 3. Use 12 filters for first 2 convolution layers, then double
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)

        
        
        #Pooling to reduce sizes, and dropout to prevent overfitting
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        
        self.drop = nn.Dropout2d(p=0.2)
#         self.norm1 = nn.BatchNorm2d(12)
#         self.norm2 = nn.BatchNorm2d(24)
        
        # There are 2 pooling layers, each with kernel size of 2. Output size: 128/(2*2) = 32
        # Feature tensors are now 32 x 32; With 24 output channels, this gives 32x32x24
        
        # Have 3 output features, corresponding to x-pos, y-pos, radius. 
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=3)
    

                
    def forward(self, x):
        """
        Feed forward through network
        Args:
            x - input to the network
            
        Returns "out", which is the network's output
        """
        
        #----------------------------------------------
        # implementation needed here 
        #----------------------------------------------
        
        #Convolution 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
#         out = self.norm1(out)

        #Convolution 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        
#         out = self.norm1(out)
        
        #Convolution 3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.drop(out)
#         out = self.norm2(out)
        
        #Convolution 4
        out = self.conv4(out)
        out = self.relu(out)
        out = F.dropout(out, training=self.training)
        
        
        out = out.view(-1, 32 * 32 * 32)
        out = self.fc(out)

        
        return out
    