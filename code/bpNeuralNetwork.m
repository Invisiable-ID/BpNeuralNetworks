%% �ô���Ϊ����BP�������Ԥ���㷨
%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��
%���������������
load data1

%��1��605���������
k=rand(1,605);
[m,n]=sort(k);

%�ҳ�ѵ�����ݺ�Ԥ������
input_train=input(n(1:454),:)';
output_train=output(n(1:454))';
input_test=input(n(455:605),:)';
output_test=output(n(455:605))';

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP����ѵ��
% %��ʼ������ṹ
net=newff(inputn,outputn,10);
net.trainParam.epochs=100;
net.trainParam.lr=0.005;
net.trainParam.goal=0.00004;

%����ѵ��
net=train(net,inputn,outputn);

%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����
an=sim(net,inputn_test);
 
%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

%% �������

figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('Ԥ�����','�������')
title('BP����Ԥ�����','fontsize',12)
ylabel('�������','fontsize',12)
xlabel('����','fontsize',12)
%Ԥ�����
error=BPoutput-output_test;

figure(2)
plot(error,'-*')
title('BP����Ԥ�����','fontsize',12)
ylabel('���','fontsize',12)
xlabel('����','fontsize',12)

figure(3)
plot((output_test-BPoutput)./BPoutput,'-*');
title('������Ԥ�����ٷֱ�')

relative_error=abs(error)./output_test;
avg_relative_error=mean(relative_error);
disp('ƽ��������:')
disp(avg_relative_error)

% ������¼Ԥ����
new_X=new_X';
[new_x,new_xout]=mapminmax(new_X);
new_x=mapminmax('apply',new_X,new_xout);
an1=sim(net,new_x);
predict_y=mapminmax('reverse',an1,outputps);
predict_y=predict_y';
disp('Ԥ��ֵΪ��')
disp(predict_y)

