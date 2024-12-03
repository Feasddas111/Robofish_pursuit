clf;
wp1=0.5;wp2=3.5;
ws1=1.9;ws2=2.1;
rp=0.1; % 通带波纹最大衰减为0.1dB
rs=100; % 阻带衰减为60dB
fs=10; % 采样频率为400Hz
wp=[wp1,wp2]/(fs/2);% 通带截止频率/Hz
ws=[ws1,ws2]/(fs/2);% 阻带截止频率/Hz
% 巴特沃斯滤波器
[b_N,b_wc]=buttord(wp,ws,rp,rs,'z');
b_N = 5;
b_wc = [0.05, 0.95];
[b_b,b_a]=butter(b_N,b_wc,'stop');
[b_hw,b_w]=freqz(b_b,b_a,512); % 绘制频率响应曲线
A1 = b_w/(2*pi)*fs;
plot(A1,abs(b_hw));


% 原始数据
N=128;   %采样频率和数据点数
n=0:N-1;t=n/fs;   %时间序列
f=n*fs/N;    %频率序列
x=2*sin(2*pi*1.5*t); %信号
% x = temp(:, 6);
y=fft(x,N);    %对信号进行快速Fourier变换
mag=abs(y);     %求得Fourier变换后的振幅
subplot(2,2,1),plot(f,mag);   %绘出随频率变化的振幅
xlabel('频率/Hz');
ylabel('振幅');title('N=128');grid on;
% subplot(2,2,2),plot(f(1:N/2),mag(1:N/2)); %绘出Nyquist频率之前随频率变化的振幅
% xlabel('频率/Hz');
% ylabel('振幅');title('N=128');grid on;
subplot(2,2,2)
plot(x)
% 滤波数据
xf=filter(b_b,b_a, x); % 复合信号通过butter滤波器后的信号yn;
yf=fft(xf,N); % 对信号做M点FFT变换
mag=abs(yf);     %求得Fourier变换后的振幅
subplot(2,2,3),plot(f,mag);   %绘出随频率变化的振幅
xlabel('频率/Hz');
ylabel('振幅');title('N=128');grid on;
% subplot(2,2,2),plot(f(1:N/2),mag(1:N/2)); %绘出Nyquist频率之前随频率变化的振幅
% xlabel('频率/Hz');
% ylabel('振幅');title('N=128');grid on;
subplot(2,2,4)
plot(xf)
hold on
% 自己实现的滤波
prev_in_data = zeros(11,1);
prev_out_data = zeros(10,1);
xff = x;
for jj = 1:length(x)
    % 挪动输入队列
    for ii = length(prev_in_data)-1:-1:1
        prev_in_data(ii+1) = prev_in_data(ii);
    end
    prev_in_data(1) = x(jj);
    out = -b_a(2:end)*prev_out_data + b_b*prev_in_data;
    % 挪动输出队列
    for ii = length(prev_out_data)-1:-1:1
        prev_out_data(ii+1) = prev_out_data(ii);
    end
    prev_out_data(1) = out;
    xff = out;
end
plot(xff)
hold on

% %% IIR陷波器设计
% % 目的：设计一个陷波器阻带在50±1.5Hz以内,采样频率为400Hz的滤波器，
% % 并要求通带最大衰减为0.1dB，阻带最小衰减为60dB。
% clc;
% clear;close all;
% wp1=48.5;wp2=51.5;
% ws1=49.9;ws2=50.1;
% rp=0.1; % 通带波纹最大衰减为0.1dB
% rs=100; % 阻带衰减为60dB
% fs=400; % 采样频率为400Hz
% wp=[wp1,wp2]/(fs/2);% 通带截止频率/Hz
% ws=[ws1,ws2]/(fs/2);% 阻带截止频率/Hz
% 
% %% -------------------------------滤波器设计----------------------------------%%
% 
% % 巴特沃斯滤波器
% [b_N,b_wc]=buttord(wp,ws,rp,rs,'z');
% [b_b,b_a]=butter(b_N,b_wc,'stop');
% [b_hw,b_w]=freqz(b_b,b_a,512); % 绘制频率响应曲线
% 
% % 切比雪夫I型滤波器
% [c1_N,c1_wc]=cheb1ord(wp,ws,rp,rs,'z');
% [c1_b,c1_a]=cheby1(c1_N,rp,c1_wc,'stop');
% [c1_hw,c1_w]=freqz(c1_b,c1_a,512);% 绘制频率响应曲线
% 
% %切比雪夫II型滤波器
% [c2_N,c2_wc]=cheb2ord(wp,ws,rp,rs,'z');
% [c2_b,c2_a]=cheby2(c2_N,rs,c2_wc,'stop');
% [c2_hw,c2_w]=freqz(c2_b,c2_a,512);% 绘制频率响应曲线
% 
% % 椭圆滤波器
% [d_N,d_wc]=ellipord(wp,ws,rp,rs,'z');
% [d_b,d_a]=ellip(d_N,rp,rs,d_wc,'stop');
% [d_hw,d_w]=freqz(d_b,d_a,512); % 绘制频率响应曲线
% 
% %% -------------------- 各个滤波器的幅频响应对比分析---------------------%%
% A1 = b_w/(2*pi)*fs;
% A2 = c1_w/(2*pi)*fs;
% A3 = c2_w/(2*pi)*fs;
% A4 = d_w/(2*pi)*fs;
% 
% figure; % 画图
% subplot(2,1,1);
% plot(A1,abs(b_hw),A2,abs(c1_hw),A3,abs(c2_hw),A4,abs(d_hw));grid;
% title('H(ejw)'); xlabel('频率／Hz'); ylabel('频率响应幅度');
% legend('butter','cheby1','cheby2','ellip');
% % axis([47,53,0,1.1]);% 定义横坐标和纵坐标的范围
% subplot(2,1,2);
% plot(A1,20*log10(abs(b_hw))/max(abs(b_hw)),A2,20*log10(abs(c1_hw))/max(abs(c1_hw)),A3,20*log10(abs(c2_hw))/max(abs(c2_hw)),A4,20*log10(abs(d_hw))/max(abs(d_hw)));
% title('损耗函数'); xlabel('频率/Hz');ylabel('幅值/dB');
% legend('butter','cheby1','cheby2','ellip');
% % axis([47,53,-130,10]); % 定义横坐标和纵坐标的范围
% grid on;
% 
% %% ---------------------------- 产生信号-----------------------------%%
% 
% M=2038400;
% n=0:M-1;
% T=1/fs;
% xn=sin(2*pi*48*n*T)+sin(2*pi*48.5*n*T)+sin(2*pi*49*n*T)+sin(2*pi*50*n*T)+sin(2*pi*51*n*T)+sin(2*pi*51.5*n*T)+sin(2*pi*52*n*T); % 信号叠加
% %--------------------------- FFT分析信号频谱-------------------------------%
% xk=fft(xn,M); % 对信号做len点FFT变换
% yn_b=filter(b_b,b_a,xn); % 复合信号通过butter滤波器后的信号yn;
% yn_c1=filter(c1_b,c1_a,xn); % 复合信号通过cheby1滤波器后的信号yn;
% yn_c2=filter(c2_b,c2_a,xn); % 复合信号通过cheby2滤波器后的信号yn;
% yn_d=filter(d_b,d_a,xn); % 复合信号通过ellip滤波器后的信号yn;
% k=2*(0:M-1)/M;
% 
% yk_b=fft(yn_b,M); % 对信号做M点FFT变换
% yk_c1=fft(yn_c1,M); % 对信号做M点FFT变换
% yk_c2=fft(yn_c2,M); % 对信号做M点FFT变换
% yk_d=fft(yn_d,M);
% % 
% % yk_b_T=Tfft(yn_b,M); % 对信号做M点FFT变换
% % yk_c1_T=Tfft(yn_c1,M); % 对信号做M点FFT变换
% % yk_c2_T=Tfft(yn_c2,M); % 对信号做M点FFT变换
% % yk_d_T=Tfft(yn_d,M); % 对信号做M点FFT变换
% yk_b_T=fft(yn_b,M); % 对信号做M点FFT变换
% yk_c1_T=fft(yn_c1,M); % 对信号做M点FFT变换
% yk_c2_T=fft(yn_c2,M); % 对信号做M点FFT变换
% yk_d_T=fft(yn_d,M); % 对信号做M点FFT变换
% %% -------------------------------波形显示----------------------------------%%
% kfs = n*fs/M;
% % 原信号
% figure(2); % 画图
% subplot(2,1,1);
% plot(n,xn,'blue');
% title('原信号波形图'); xlabel('f/Hz');ylabel('幅度');
% axis([0,4000,-8,8]); % 定义横坐标和纵坐标的范围
% grid on;
% subplot(2,1,2);
% plot(kfs/2,abs(xk),'blue'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=k*fs/2
% title('滤波前输入信号x(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');%
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(xk))]);% 定义横坐标和纵坐标的范围
% grid on;
% 
% % 巴特沃斯滤波器
% figure; % 画图
% subplot(3,1,1);
% plot(n,yn_b,'r');
% title('butter滤波后波形图'); xlabel('f/Hz');ylabel('幅度');
% axis([0,4000,-8,8]); % 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_b),'b'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('fft变换');
% title('butter滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% %axis([40,60,0,max(abs(yk_b))]); % 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_b_T),'r'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('T*fft变换');
% title('butter滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_b_T))]); % 定义横坐标和纵坐标的范围
% grid on;
% 
% % 切比雪夫I型滤波器
% figure; % 画图
% subplot(3,1,1);
% plot(n,yn_c1,'m');
% title('cheby1滤波后波形图'); xlabel('f/Hz');ylabel('幅度');
% axis([0,4000,-8,8]); % 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_c1),'m'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('fft变换');
% title('cheby1滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c1))]);% 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_c1_T),'m'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('T*fft变换');
% title('cheby1滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c1_T))]);% 定义横坐标和纵坐标的范围
% grid on;
% 
% % 切比雪夫II型滤波器
% figure; % 画图
% subplot(3,1,1);
% plot(n,yn_c2,'g');
% title('cheby2滤波后波形图'); xlabel('f/Hz');ylabel('幅度');
% axis([0,4000,-8,8]); % 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_c2),'g');% k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('fft变换');
% title('cheby2滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c2))]);%定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_c2_T),'g');%画离散图 k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('T*fft变换');
% title('cheby2滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c2_T))]);% 定义横坐标和纵坐标的范围
% grid on;
% 
% % 3.4 椭圆滤波器
% figure; % 画图
% subplot(3,1,1);
% plot(n,yn_d);
% title('ellip滤波后波形图'); xlabel('f/Hz');ylabel('幅度');
% axis([0,4000,-8,8]); % 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_d));% k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('fft变换');
% title('ellip滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_d))]);% 定义横坐标和纵坐标的范围
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_d_T));% k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;得到f=kfs/2
% legend('T*fft变换');
% title('ellip滤波后输出信号y(n)频谱特性'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_d_T))]);% 定义横坐标和纵坐标的范围
% grid on;