%%setup 
a = arduino('COM5','Uno','Libraries','I2C');
p  = scanI2CBus(a,0);
disp(p);
mpu = device(a,'I2CAddress','0x68');
writeRegister(mpu, hex2dec('B6'), hex2dec('00'), 'int16'); %reset
data=zeros(10000,14,'int8'); %prelocating
j=1;
%%loop
while(true)
  x=1;
  for i=59:72 % 14 Data Registers for Accel,Temp,Gyro
      data(j,x)= readRegister(mpu, i, 'int8');
      x=x+1;
  end
  y=swapbytes(typecast(data(j,:), 'int16'));
  acc_x(j)=double(y(1));
  acc_y(j)=double(y(2));
  acc_z(j)=double(y(3));
  j=j+1;
end%