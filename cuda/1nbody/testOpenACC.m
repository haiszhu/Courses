% check correctness

!nvc -acc -gpu=cc80 testOpenACC.c
!./a.out

% 
mu = readmatrix('mu.txt');
src = readmatrix('src.txt');
targ = readmatrix('targ.txt');
pot = readmatrix('pot.txt');
N = size(src,2);
M = size(targ,2);

runtime = readmatrix('runtime.txt');
disp([' run time on gpu: ', num2str(runtime), ' milliseconds' ])

tic, 
pot2 = zeros(M,1);
for j=1:N
  pot2 = pot2 + mu(j)./sqrt((src(1,j) - targ(1,:)').^2+(src(2,j) - targ(2,:)').^2+(src(3,j) - targ(3,:)').^2); 
end
cpuruntime = toc;
disp([' run time on cpu: ',num2str(cpuruntime*1e+03),' milliseconds']); % 

diff = abs(pot-pot2);
max(diff)/max(abs(pot))

keyboard