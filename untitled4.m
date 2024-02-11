clear a;
a = arduino('COM5', 'Uno');
for k =  1:100
   writeDigitalPin(a, 'D11', 0);
   pause(2);
   writeDigitalPin(a, 'D11', 1);
   disp(k);
end
   disp(a);