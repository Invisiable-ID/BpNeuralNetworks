clc;clear;
%��������
load data.mat
%   X - input data.
%   Y - target data.

x = X';
t = Y';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.

% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% View the Network
view(net)
%sim������������ɵ�Ԥ�⺯�������ڼ����Ԥ��ģ��net�Ƿ���ȷ
sim(net, new_X(1,:)')

%дһ��ѭ��������605��ѵ�����ݵĲв��ƽ��������
output_y=zeros(605,0);
for i=1:605
     result1 = sim(net, X(i,:)');
     output_y(i)=result1;
end
output_y=output_y';
error=output_y-Y;
relative_error=abs(error)./Y;
avg_relative_error=mean(relative_error);
disp('ƽ��������:')
disp(avg_relative_error)

% дһ��ѭ����Ԥ��test��152������
predict_y = zeros(152,1); % ��ʼ��predict_y��������¼Ԥ����
for i = 1: 152
    result2 = sim(net, new_X(i,:)');
    predict_y(i) = result2;
end
disp('Ԥ��ֵΪ��')
disp(predict_y)