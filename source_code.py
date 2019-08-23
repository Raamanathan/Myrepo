import numpy as np
from numpy import interp
#import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
from time import sleep
import spidev
import time
start =time.time()

spi = spidev.SpiDev()
spi.open(0,0)

def analogInput(channel):
  spi.max_speed_hz = 1350000
  adc = spi.xfer2([1,(8+channel)<<4,0])
  data = ((adc[1]&3) << 8) + adc[2]
  return data
r =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
ledpin = 33
GPIO.setwarnings(False)                 #disable warnings
GPIO.setmode(GPIO.BOARD)
GPIO.cleanup()          #set pin numbering system
GPIO.setup(ledpin,GPIO.OUT)
pwm = GPIO.PWM(ledpin,1000)
pwm.start(0)
sp=input("Enter the setpoint value (0-3.3)V")
#sp = np.array([sp])
# N is batch size(sample size); D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 1, 10, 1


# Create random input and output data
x = np.array([[0.5]])
f= int(30.303030 * x[0][0])
pwm.ChangeDutyCycle(f)
ontime = time.time()
sleep(1)
g=1
prev =0
#sleep(1)
count = 0
van = 0
while(abs(prev-g)>0.00002):
#while(count<10):
    prev = g
    output = analogInput(0)
#output1 = analogInput(1) # Reading from CH0
#output = interp(output, [0, 1023], [0, 100])
    g = (((output)*3.3)/1024)
 #   van = van + g
    count = count + 1
    sleep(1)

#y = np.array([[1] ])  #measure
    print (g)
#g = van/10
print(time.time()-ontime)
print ("Fully Charged")
y =np.array([[g]])
#sleep(2)
# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 0.00051
loss_col = []
y_pred = 0
a=0
sums=0

while True:
#for t in range(400):
   while(a<10):
    r[a] = (w1[0][a])*(x[0][0])
    r[a] = np.maximum(r[a],0)
    r[a] = r[a] * (w2[a][0])
    sums = sums + r[a]
    a=a+1
   sums = np.maximum(sums,0)
   a=0
   y_pred = sums
#   print (sums)
    sums =0
   while(abs(y-y_pred)>0.01):

    # Forward pass: compute predicted y
       h = x.dot(w1)
       h_relu = np.maximum(h, 0)  # using ReLU as activate function
       y_pred = h_relu.dot(w2)

    # Compute and print loss
       loss = np.square(y_pred - y).sum() # loss function
       loss_col.append(loss)
#       print( loss, y_pred)
       output = analogInput(0)
       g = (output*3.3)/1024
       print g

    # Backprop to compute gradients of w1 and w2 with respect to loss
       grad_y_pred = 2.0 * (y_pred - y) # the last layer's error
       grad_w2 = h_relu.T.dot(grad_y_pred)
       grad_h_relu = grad_y_pred.dot(w2.T) # the second laye's error
       grad_h = grad_h_relu.copy()
       grad_h[h < 0] = 0  # the derivate of ReLU
       grad_w1 = x.T.dot(grad_h)

    # Update weights
       w1 -= learning_rate * grad_w1
       w2 -= learning_rate * grad_w2
       flag=1
   func = 0
   prod = 0
   #if(flag==1):
   while(a<10):
       if(w1[0][a]>0):
         prod = w1[0][a]*w2[a][0]
       func =func + prod
       prod = 0
       a = a+1
   ##func = y_pred/x[0][0]
   a=0

#   print("func:")
#   print( func)
#   print(func[0][0])
   flag=0
   if(func==0):
     continue
   x[0][0] = (sp / func)
   #print(x[0][0])
   #x[0][0] = (x[0][0])
   e= int (x[0][0]* 30.303030)
#   print(x[0][0])
   if(e>100):
     e=100
   elif(e<0):
     e=0
   pwm.ChangeDutyCycle(e)
   sleep(0.2)
   #measure y here......
   output = analogInput(0) # Reading from CH0
   #output1 = analogInput(1)
   #output = interp(output, [0, 1023], [0, 100])
   g = float(((output)*3.3)/1024)
  # plt.plot(loss_col)
   # print(w1,w2)
   y[0][0] = g #+0.7
  # print(w1[0][9])
 #  g = g+0.7
   print ( g)
   #sleep(1)
   a=0
   while(abs (g - sp)<0.0005):
       output = analogInput(0)
       #output1 = analogInput(1)
       g = float (((output)*3.3)/1024)
#       g = g+0.7
       print(g)
 #  if(time.time()-start>20):
  #   sp = input("Enter new sp")
  #   start = time.time()

 #  plt.show()
