%% 该代码为基于BP神经网络的预测算法
%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化
%下载输入输出数据
load data1

%从1到605间随机排序
k=rand(1,605);
[m,n]=sort(k);

%找出训练数据和预测数据
input_train=input(n(1:454),:)';
output_train=output(n(1:454))';
input_test=input(n(455:605),:)';
output_test=output(n(455:605))';

%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP网络训练
% %初始化网络结构
net=newff(inputn,outputn,10);
net.trainParam.epochs=100;
net.trainParam.lr=0.005;
net.trainParam.goal=0.00004;

%网络训练
net=train(net,inputn,outputn);

%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
 
%网络预测输出
an=sim(net,inputn_test);
 
%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

%% 结果分析

figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('预测输出','期望输出')
title('BP网络预测输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
%预测误差
error=BPoutput-output_test;

figure(2)
plot(error,'-*')
title('BP网络预测误差','fontsize',12)
ylabel('误差','fontsize',12)
xlabel('样本','fontsize',12)

figure(3)
plot((output_test-BPoutput)./BPoutput,'-*');
title('神经网络预测误差百分比')

relative_error=abs(error)./output_test;
avg_relative_error=mean(relative_error);
disp('平均相对误差:')
disp(avg_relative_error)

% 用来记录预测结果
new_X=new_X';
[new_x,new_xout]=mapminmax(new_X);
new_x=mapminmax('apply',new_X,new_xout);
an1=sim(net,new_x);
predict_y=mapminmax('reverse',an1,outputps);
predict_y=predict_y';
disp('预测值为：')
disp(predict_y)

