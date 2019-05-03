clc; clear all;
%------ SPECIFY DATA ------------------------------------------------------
%%选择文件名称
stringname='111';
%选择你要处理的信号点数
points=1024; 
PATH= 'J:\20190428-ECG\MIT-BIH\MIT-BIH'; % path, where data are saved
HEADERFILE= strcat(stringname,'.hea');      % header-file in text format
ATRFILE= strcat(stringname,'.atr');        % attributes-file in binary format
DATAFILE=strcat(stringname,'.dat');        % data-file
SAMPLES2READ=points;         % number of samples to be read
                      % in case of more than one signal:
                            % 2*SAMPLES2READ samples are read
   
%------ LOAD HEADER DATA --------------------------------------------------
fprintf(1,'\\n$> WORKING ON %s ...\n', HEADERFILE);
signalh= fullfile(PATH, HEADERFILE);
fid1=fopen(signalh,'r');
z= fgetl(fid1);
A= sscanf(z, '%*s %d %d %d',[1,3]);
nosig= A(1);  % number of signals
sfreq=A(2);   % sample rate of data
clear A;
for k=1:nosig
    z= fgetl(fid1);
    A= sscanf(z, '%*s %d %d %d %d %d',[1,5]);
    dformat(k)= A(1);           % format; here only 212 is allowed
    gain(k)= A(2);              % number of integers per mV
    bitres(k)= A(3);            % bitresolution
    zerovalue(k)= A(4);         % integer value of ECG zero point
    firstvalue(k)= A(5);        % first integer value of signal (to test for errors)
end;
fclose(fid1);
clear A;

%------ LOAD BINARY DATA --------------------------------------------------
if dformat~= [212,212], error('this script does not apply binary formats different to 212.'); end;
signald= fullfile(PATH, DATAFILE);            % data in format 212
fid2=fopen(signald,'r');
A= fread(fid2, [3, SAMPLES2READ], 'uint8')';  % matrix with 3 rows, each 8 bits long, = 2*12bit
fclose(fid2);
M2H= bitshift(A(:,2), -4);
M1H= bitand(A(:,2), 15);
PRL=bitshift(bitand(A(:,2),8),9);     % sign-bit
PRR=bitshift(bitand(A(:,2),128),5);   % sign-bit
M( : , 1)= bitshift(M1H,8)+ A(:,1)-PRL;
M( : , 2)= bitshift(M2H,8)+ A(:,3)-PRR;
if M(1,:) ~= firstvalue, error('inconsistency in the first bit values'); end;
switch nosig
case 2
    M( : , 1)= (M( : , 1)- zerovalue(1))/gain(1);
    M( : , 2)= (M( : , 2)- zerovalue(2))/gain(2);
    TIME=(0:(SAMPLES2READ-1))/sfreq;
case 1
    M( : , 1)= (M( : , 1)- zerovalue(1));
    M( : , 2)= (M( : , 2)- zerovalue(1));
    M=M';
    M(1)=[];
    sM=size(M);
    sM=sM(2)+1;
    M(sM)=0;
    M=M';
    M=M/gain(1);
    TIME=(0:2*(SAMPLES2READ)-1)/sfreq;
otherwise  % this case did not appear up to now!
    % here M has to be sorted!!!
    disp('Sorting algorithm for more than 2 signals not programmed yet!');
end;
clear A M1H M2H PRR PRL;
fprintf(1,'\\n$> LOADING DATA FINISHED \n');

%------ LOAD ATTRIBUTES DATA ----------------------------------------------
atrd= fullfile(PATH, ATRFILE);      % attribute file with annotation data
fid3=fopen(atrd,'r');
A= fread(fid3, [2, inf], 'uint8')';
fclose(fid3);
ATRTIME=[];
ANNOT=[];
sa=size(A);
saa=sa(1);
i=1;
while i<=saa
    annoth=bitshift(A(i,2),-2);
    if annoth==59
        ANNOT=[ANNOT;bitshift(A(i+3,2),-2)];
        ATRTIME=[ATRTIME;A(i+2,1)+bitshift(A(i+2,2),8)+...
                bitshift(A(i+1,1),16)+bitshift(A(i+1,2),24)];
        i=i+3;
    elseif annoth==60
        % nothing to do!
    elseif annoth==61
        % nothing to do!
    elseif annoth==62
        % nothing to do!
    elseif annoth==63
        hilfe=bitshift(bitand(A(i,2),3),8)+A(i,1);
        hilfe=hilfe+mod(hilfe,2);
        i=i+hilfe/2;
    else
        ATRTIME=[ATRTIME;bitshift(bitand(A(i,2),3),8)+A(i,1)];
        ANNOT=[ANNOT;bitshift(A(i,2),-2)];
   end;
   i=i+1;
end;
ANNOT(length(ANNOT))=[];       % last line = EOF (=0)
ATRTIME(length(ATRTIME))=[];   % last line = EOF
clear A;
ATRTIME= (cumsum(ATRTIME))/sfreq;
ind= find(ATRTIME <= TIME(end));
ATRTIMED= ATRTIME(ind);
ANNOT=round(ANNOT);
ANNOTD= ANNOT(ind);

%------ DISPLAY DATA ------------------------------------------------------
figure(1); clf, box on, hold on ;grid on ;
%plot(TIME, M(:,1),'r');
if nosig==2
    plot(TIME, M(:,1),'b');
end;
for k=1:length(ATRTIMED)
    text(ATRTIMED(k),0,num2str(ANNOTD(k)));
end;
xlim([TIME(1), TIME(end)]);
xlabel('Time / s'); ylabel('Voltage / mV');
string=['ECG signal ',DATAFILE];
title(string);
fprintf(1,'\\n$> DISPLAYING DATA FINISHED \n');
% -------------------------------------------------------------------------
fprintf(1,'\\n$> ALL FINISHED \n');

%%%%%%%%%%%%%%%%%%%去除噪声和基线漂移%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
level=8; wavename='bior2.6';
ecgdata=M(:,1);
figure(2);
plot(ecgdata(1:points));grid on ;axis tight;axis([1,points,-1,1.5]);
title('原始ECG信号');
%%%%%%%%%%进行小波变换8层
[C,L]=wavedec(ecgdata,level,wavename);
%%%%%%%提取尺度系数，
A1=appcoef(C,L,wavename,1);
A2=appcoef(C,L,wavename,2);
A3=appcoef(C,L,wavename,3);
A4=appcoef(C,L,wavename,4);
A5=appcoef(C,L,wavename,5);
A6=appcoef(C,L,wavename,6);
A7=appcoef(C,L,wavename,7);
A8=appcoef(C,L,wavename,8);
%%%%%%%提取细节系数
D1=detcoef(C,L,1);
D2=detcoef(C,L,2);
D3=detcoef(C,L,3);
D4=detcoef(C,L,4);
D5=detcoef(C,L,5);
D6=detcoef(C,L,6);
D7=detcoef(C,L,7);
D8=detcoef(C,L,8);
%%%%%%%%%%%%重构
A8=zeros(length(A8),1); %去除基线漂移,8层低频信息
RA7=idwt(A8,D8,wavename);
RA6=idwt(RA7(1:length(D7)),D7,wavename);
RA5=idwt(RA6(1:length(D6)),D6,wavename);
RA4=idwt(RA5(1:length(D5)),D5,wavename);
RA3=idwt(RA4(1:length(D4)),D4,wavename);
RA2=idwt(RA3(1:length(D3)),D3,wavename);
D2=zeros(length(D2),1); %去除高频噪声，2层高频噪声
RA1=idwt(RA2(1:length(D2)),D2,wavename);
D1=zeros(length(D1),1);%去除高频噪声，1层高频噪声
DenoisingSignal=idwt(RA1,D1,wavename);
figure(3);
plot(DenoisingSignal);
title('去除噪声的ECG信号'); grid on; axis tight;axis([1,points,-1,1.5]);
clear ecgdata;


