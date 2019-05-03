clear;clc;
%% 加载数据
fprintf('Loading data...\n');
tic;
load('N_dat.mat');
load('L_dat.mat');
load('R_dat.mat');
load('V_dat.mat');
fprintf('Finished!\n');
toc;
fprintf('=============================================================\n');
%% 每个数据集都是值去相同的数据集数目，repmat可以赋值相同的数据值

fprintf('Data preprocessing...\n');
tic;
Nb=Nb(1:5000,:);Label1=ones(1,5000);%Label1=repmat([1;0;0;0],1,5000);
Vb=Vb(1:5000,:);Label2=ones(1,5000)*2;%Label2=repmat([0;1;0;0],1,5000);
Rb=Rb(1:5000,:);Label3=ones(1,5000)*3;%Label3=repmat([0;0;1;0],1,5000);
Lb=Lb(1:5000,:);Label4=ones(1,5000)*4;%Label4=repmat([0;0;0;1],1,5000);

Data=[Nb;Vb;Rb;Lb];
Label=[Label1,Label2,Label3,Label4];
Label=Label';

clear Nb;clear Label1;
clear Rb;clear Label2;
clear Lb;clear Label3;
clear Vb;clear Label4;
Data=Data-repmat(mean(Data,2),1,250); %数据减去平均值 mean(A,2) 求每行的平均值
% 热跑码头（A,m,n）将矩阵A复制M*N块
fprintf('Finished!\n');
toc;
fprintf('=============================================================\n');
%% 特征提取 使用小波变换数据
fprintf('Feature extracting and normalizing...\n');
tic;
Feature=[];
for i=1:size(Data,1)  %返回矩阵data的行数
    [C,L]=wavedec(Data(i,:),5,'db6');  %% db6小波函数进行5次小波分解
    Feature=[Feature;C(1:25)]; %选择前25个系数作为特征
end

Nums=randperm(20000);      %随机打乱数据
train_x=Feature(Nums(1:10000),:);
test_x=Feature(Nums(10001:end),:);
train_y=Label(Nums(1:10000));
test_y=Label(Nums(10001:end));

[train_x,ps]=mapminmax(train_x',0,1); % 全部数据最大最小归一化
test_x=mapminmax('apply',test_x',ps);
train_x=train_x';test_x=test_x';
fprintf('Finished!\n');
toc;
fprintf('=============================================================\n');
%% SVM进行预测分类
fprintf('SVM training and testing...\n');
tic;
model=svmtrain(train_y,train_x,'-c 2 -g 1'); %svmtrain函数训练
[ptest,~,~]=svmpredict(test_y,test_x,model); %svmpredict进行预测

Correct_Predict=zeros(1,4);                     %混淆矩阵
Class_Num=zeros(1,4);
Conf_Mat=zeros(4);
for i=1:10000
    Class_Num(test_y(i))=Class_Num(test_y(i))+1;
    Conf_Mat(test_y(i),ptest(i))=Conf_Mat(test_y(i),ptest(i))+1;
    if ptest(i)==test_y(i)
        Correct_Predict(test_y(i))= Correct_Predict(test_y(i))+1;
    end
end
ACCs=Correct_Predict./Class_Num;
fprintf('Accuracy_N = %.2f%%\n',ACCs(1)*100);
fprintf('Accuracy_V = %.2f%%\n',ACCs(2)*100);
fprintf('Accuracy_R = %.2f%%\n',ACCs(3)*100);
fprintf('Accuracy_L = %.2f%%\n',ACCs(4)*100);

toc;
