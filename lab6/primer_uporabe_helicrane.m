

%zaèetni pogoj:
 x = [0 0];
 ts = 0.01;%tega se ne da spreminjati!
 ym(1) = 0;
 
for i = 1:2000

  

%vhod:
u = 0;
Fm = u;   
[fi_ fip_] = helicrane(Fm,x);
x = [fip_ fi_];

kot(i+1) = fi_; %fi_ je izhod procesa, ki nas zanima.

end
figure
plot(kot)