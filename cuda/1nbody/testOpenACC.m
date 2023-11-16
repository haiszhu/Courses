% check correctness

!nvc -acc -gpu=cc80 testOpenACC.c
tic
!./a.out
toc

mu = readmatrix('mu.txt');
srcx = readmatrix('srcx.txt');
srcy = readmatrix('srcy.txt');
srcz = readmatrix('srcz.txt');
targx = readmatrix('targx.txt');
targy = readmatrix('targy.txt');
targz = readmatrix('targz.txt');
pot = readmatrix('pot.txt');
Amat = 1./sqrt((srcx(:)' - targx(:)).^2+(srcy(:)' - targy(:)).^2+(srcz(:)' - targz(:)).^2);
pot2 = Amat*mu;
diff = abs(pot-pot2);
max(diff)/max(abs(pot))

keyboard